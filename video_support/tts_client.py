from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import srt
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)
_download_local = threading.local()

TTS_INVALID_TEXT_CODES = {"40402002"}
TTS_INVALID_TEXT_MESSAGES = {"TTSInvalidText"}


@dataclass(frozen=True)
class TTSConfig:
    api_base: str
    voice: str
    resource_id: str
    rate: str = "0"
    timeout: int = 120
    request_retries: int = 1
    poll_interval: float = 5.0
    max_polls: int = 120
    pool_connections: int = 16
    pool_maxsize: int = 16


@dataclass(frozen=True)
class SrtEntry:
    index: int
    start: dt.timedelta
    end: dt.timedelta
    content: str

    @property
    def duration_ms(self) -> int:
        return max(0, int((self.end - self.start).total_seconds() * 1000))

    @property
    def start_ms(self) -> int:
        return max(0, int(self.start.total_seconds() * 1000))

    @property
    def end_ms(self) -> int:
        return max(0, int(self.end.total_seconds() * 1000))


@dataclass(frozen=True)
class BatchResult:
    batch_id: int
    items: list[dict[str, Any]]


class TTSError(RuntimeError):
    def __init__(self, message: str, result: dict[str, Any] | None = None):
        super().__init__(message)
        self.result = result or {}


class TTSInvalidTextError(TTSError):
    pass


class TTSTerminalError(TTSError):
    pass


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Khong thay file: {path}")
    if not path.is_file():
        raise ValueError(f"Khong phai file: {path}")
    return path


def clean_one_line(text: str) -> str:
    text = str(text or "").replace("\ufeff", "")
    text = unicodedata.normalize("NFKC", text)

    text = "".join(
        ch
        for ch in text
        if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )

    text = re.sub(r"<[^>]{1,120}>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_for_tts(text: str) -> str:
    text = clean_one_line(text)
    return text or "."


def atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")

    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return path


def save_json(path: str | Path, data: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=2),
    )


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    return data if isinstance(data, list) else []


def read_srt_entries(path: str | Path, keep_empty: bool = False) -> list[SrtEntry]:
    path = require_file(path)
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    entries: list[SrtEntry] = []

    for sub in srt.parse(raw):
        content = clean_one_line(sub.content)
        if not content and not keep_empty:
            continue

        entries.append(
            SrtEntry(
                index=int(sub.index or 0),
                start=sub.start,
                end=sub.end,
                content=content,
            )
        )

    entries.sort(key=lambda entry: (entry.start_ms, entry.index))
    validate_unique_indices(entries)
    return entries


def write_srt_entries(entries: list[SrtEntry], path: str | Path) -> Path:
    subtitles = [
        srt.Subtitle(
            index=i,
            start=entry.start,
            end=entry.end,
            content=entry.content,
        )
        for i, entry in enumerate(entries, start=1)
    ]
    return atomic_write_text(path, srt.compose(subtitles))


def validate_unique_indices(entries: list[SrtEntry]) -> None:
    seen: set[int] = set()
    duplicated: list[int] = []

    for entry in entries:
        if entry.index in seen:
            duplicated.append(entry.index)
        seen.add(entry.index)

    if duplicated:
        sample = ", ".join(map(str, duplicated[:20]))
        raise ValueError(f"SRT co index trung, khong the ghi audio an toan: {sample}")


def get_tts_output_path(output_dir: str | Path, index: int) -> Path:
    return Path(output_dir) / f"{int(index):05d}.mp3"


def is_audio_done(path: str | Path, min_bytes: int = 1024) -> bool:
    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def make_session(pool_connections: int, pool_maxsize: int) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=0,
        pool_block=True,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_download_session(pool_connections: int = 8, pool_maxsize: int = 8) -> requests.Session:
    session = getattr(_download_local, "session", None)
    if session is None:
        session = make_session(pool_connections, pool_maxsize)
        _download_local.session = session
    return session


def get_task_error(result: dict[str, Any]) -> tuple[str, str]:
    err_code = result.get("err_code")
    err_msg = result.get("err_msg")

    tasks = (((result or {}).get("response") or {}).get("data") or {}).get("tasks") or []
    if tasks and isinstance(tasks[0], dict):
        task = tasks[0]
        err_code = task.get("err_code", err_code)
        err_msg = task.get("err_msg", err_msg)

    return str(err_code or ""), str(err_msg or "")


def is_tts_invalid_text(result: dict[str, Any]) -> bool:
    err_code, err_msg = get_task_error(result)
    return err_code in TTS_INVALID_TEXT_CODES or err_msg in TTS_INVALID_TEXT_MESSAGES


def compact_error(result: dict[str, Any]) -> str:
    err_code, err_msg = get_task_error(result)
    task_id = result.get("task_id") or ""
    status = result.get("status") or ""
    return f"task_id={task_id} status={status} err_code={err_code} err_msg={err_msg}".strip()


class TTSClient:
    def __init__(self, config: TTSConfig):
        if not str(config.api_base or "").strip():
            raise ValueError("Thieu api_base")
        if not str(config.voice or "").strip():
            raise ValueError("Thieu voice")
        if not str(config.resource_id or "").strip():
            raise ValueError("Thieu resource_id")

        self.config = config
        self.api_base = config.api_base.rstrip("/")
        self.session = make_session(config.pool_connections, config.pool_maxsize)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> TTSClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        retries: int | None = None,
    ) -> dict[str, Any]:
        url = f"{self.api_base}{endpoint}"
        last_error: Exception | None = None
        total_retries = max(1, int(self.config.request_retries if retries is None else retries))

        for attempt in range(1, total_retries + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError(f"Response khong phai JSON object: {url}")
                return data
            except Exception as exc:
                last_error = exc
                if attempt >= total_retries:
                    break
                time.sleep(min(8.0, 0.75 * attempt))

        raise RuntimeError(f"POST {endpoint} failed: {last_error}")

    def create(self, texts: list[str]) -> dict[str, Any]:
        clean_texts = [text_for_tts(text) for text in texts]
        if not clean_texts:
            raise ValueError("texts rong")

        return self.post_json(
            "/tts/create",
            {
                "texts": clean_texts,
                "voice": self.config.voice,
                "resource_id": self.config.resource_id,
                "rate": str(self.config.rate),
            },
            retries=1,
        )

    def query(
        self,
        task_id: str,
        token: str | None,
        bind_id: str = "",
        identity_index: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": task_id,
            "token": token or "",
            "bind_id": bind_id or "",
        }

        if identity_index is not None:
            payload["identity_index"] = int(identity_index)

        return self.post_json("/tts/query", payload, retries=1)

    def wait(self, created: dict[str, Any]) -> dict[str, Any]:
        task_id = str(created.get("task_id") or "")
        token = created.get("token")
        bind_id = str(created.get("bind_id") or "")
        identity_index = created.get("identity_index")

        if not task_id:
            raise RuntimeError(f"TTS create response thieu task_id: {created}")

        estimated_ms = extract_estimated_ms(created)
        if estimated_ms > 0:
            time.sleep(min(12.0, max(1.0, estimated_ms / 1000.0)))

        interval = max(1.0, float(self.config.poll_interval))

        for attempt in range(1, self.config.max_polls + 1):
            result = self.query(
                task_id=task_id,
                token=token,
                bind_id=bind_id,
                identity_index=identity_index,
            )

            status = str(result.get("status") or "").lower()
            logger.info("TTS poll %s/%s task=%s status=%s", attempt, self.config.max_polls, task_id, status)

            if status == "succeed":
                return result

            if status in {"failed", "fail", "error"}:
                message = compact_error(result)
                if is_tts_invalid_text(result):
                    raise TTSInvalidTextError(message, result)
                raise TTSTerminalError(message, result)

            estimated_ms = extract_estimated_ms(result)
            if estimated_ms > 0:
                sleep_s = min(20.0, max(interval, estimated_ms / 1000.0))
            else:
                sleep_s = interval

            time.sleep(sleep_s)
            interval = min(20.0, interval * 1.2)

        raise TimeoutError(f"TTS polling timeout task_id={task_id}")


def extract_estimated_ms(data: dict[str, Any]) -> int:
    tasks = (((data or {}).get("response") or {}).get("data") or {}).get("tasks") or []
    if tasks and isinstance(tasks[0], dict):
        try:
            return int(tasks[0].get("estimated_time") or 0)
        except Exception:
            return 0

    try:
        return int(data.get("estimated_time") or 0)
    except Exception:
        return 0


def extract_speech_url(item: dict[str, Any]) -> str | None:
    return item.get("speech_url") or item.get("url") or item.get("audio_url")


def download_binary_once(
    url: str,
    output_path: str | Path,
    timeout: int = 120,
    chunk_size: int = 1024 * 512,
    min_bytes: int = 1024,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_audio_done(output_path, min_bytes=min_bytes):
        return output_path

    tmp_path = output_path.with_name(
        f"{output_path.name}.part.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
    )

    try:
        session = get_download_session()
        with session.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)

        if tmp_path.stat().st_size < min_bytes:
            raise RuntimeError(f"Downloaded file too small: {tmp_path.stat().st_size} bytes")

        os.replace(tmp_path, output_path)
        return output_path
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def download_binary_with_retry(
    url: str,
    output_path: str | Path,
    timeout: int = 120,
    retries: int = 3,
    min_bytes: int = 1024,
) -> Path:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            return download_binary_once(
                url=url,
                output_path=output_path,
                timeout=timeout,
                min_bytes=min_bytes,
            )
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(8.0, 0.5 * attempt))

    raise RuntimeError(f"Download failed: {output_path}. Error: {last_error}")


def download_tts_item(
    item: dict[str, Any],
    timeout: int = 120,
    retries: int = 3,
    min_bytes: int = 1024,
) -> dict[str, Any]:
    output_path = Path(item["output_path"])

    if is_audio_done(output_path, min_bytes=min_bytes):
        return {**item, "status": "skipped_existing"}

    try:
        download_binary_with_retry(
            url=item["speech_url"],
            output_path=output_path,
            timeout=timeout,
            retries=retries,
            min_bytes=min_bytes,
        )
        return {**item, "status": "downloaded"}
    except Exception as exc:
        return {**item, "status": "download_failed", "error": str(exc)}


def download_tts_items_concurrent(
    items: list[dict[str, Any]],
    max_workers: int = 8,
    timeout: int = 120,
    retries: int = 3,
    min_bytes: int = 1024,
) -> list[dict[str, Any]]:
    if not items:
        return []

    workers = max(1, min(int(max_workers), len(items)))
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_tts_item, item, timeout, retries, min_bytes)
            for item in items
        ]

        for future in as_completed(futures):
            results.append(future.result())

    return sorted(results, key=lambda item: int(item["index"]))


def build_tts_download_items(
    audio_subtitles: list[dict[str, Any]],
    entries: list[SrtEntry],
    output_dir: str | Path,
    min_bytes: int = 1024,
) -> list[dict[str, Any]]:
    output_dir = ensure_dir(output_dir)

    if len(audio_subtitles) < len(entries):
        raise RuntimeError(f"TTS response thieu audio: {len(audio_subtitles)}/{len(entries)}")

    items: list[dict[str, Any]] = []

    for entry, audio_item in zip(entries, audio_subtitles):
        output_path = get_tts_output_path(output_dir, entry.index)

        if is_audio_done(output_path, min_bytes=min_bytes):
            continue

        speech_url = extract_speech_url(audio_item)
        if not speech_url:
            raise RuntimeError(f"Missing speech_url for subtitle index {entry.index}")

        items.append(
            {
                "index": entry.index,
                "start_ms": entry.start_ms,
                "end_ms": entry.end_ms,
                "duration_ms": entry.duration_ms,
                "text": entry.content,
                "tts_text": text_for_tts(entry.content),
                "speech_url": speech_url,
                "output_path": str(output_path),
            }
        )

    return items


def result_for_existing(entry: SrtEntry, output_dir: str | Path) -> dict[str, Any]:
    return {
        "index": entry.index,
        "start_ms": entry.start_ms,
        "end_ms": entry.end_ms,
        "duration_ms": entry.duration_ms,
        "text": entry.content,
        "tts_text": text_for_tts(entry.content),
        "output_path": str(get_tts_output_path(output_dir, entry.index)),
        "status": "skipped_existing",
    }


def result_for_failed(
    entry: SrtEntry,
    output_dir: str | Path,
    error: str,
    error_kind: str = "tts_failed",
    response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "index": entry.index,
        "start_ms": entry.start_ms,
        "end_ms": entry.end_ms,
        "duration_ms": entry.duration_ms,
        "text": entry.content,
        "tts_text": text_for_tts(entry.content),
        "output_path": str(get_tts_output_path(output_dir, entry.index)),
        "status": "tts_failed",
        "error_kind": error_kind,
        "error": error,
        "response": response or {},
    }


def load_invalid_indices(output_dir: str | Path) -> set[int]:
    bad_items = load_json_list(Path(output_dir) / "bad_tts_items.json")
    indices: set[int] = set()

    for item in bad_items:
        if item.get("status") != "tts_failed":
            continue

        if item.get("error_kind") != "TTSInvalidText":
            continue

        try:
            indices.add(int(item["index"]))
        except Exception:
            pass

    return indices


def merge_unique_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}

    for item in results:
        try:
            index = int(item.get("index", 0))
        except Exception:
            continue

        old = by_index.get(index)
        if old is None:
            by_index[index] = item
            continue

        old_status = str(old.get("status") or "")
        new_status = str(item.get("status") or "")

        if old_status == "tts_failed" and new_status in {"downloaded", "skipped_existing"}:
            by_index[index] = item
        elif old_status not in {"downloaded", "skipped_existing", "tts_failed"}:
            by_index[index] = item

    return [by_index[index] for index in sorted(by_index)]


def save_manifest(output_dir: str | Path, results: list[dict[str, Any]], name: str = "manifest.json") -> Path:
    results = merge_unique_results(results)
    return save_json(Path(output_dir) / name, results)


def save_bad_items(output_dir: str | Path, results: list[dict[str, Any]]) -> Path | None:
    old_items = load_json_list(Path(output_dir) / "bad_tts_items.json")
    new_items = [
        item
        for item in results
        if item.get("status") in {"tts_failed", "download_failed"}
    ]

    merged = merge_unique_results(old_items + new_items)
    bad_items = [
        item
        for item in merged
        if item.get("status") in {"tts_failed", "download_failed"}
    ]

    if not bad_items:
        return None

    return save_json(Path(output_dir) / "bad_tts_items.json", bad_items)


def save_missing_audio(
    output_dir: str | Path,
    entries: list[SrtEntry],
    min_audio_bytes: int = 1024,
    skip_indices: set[int] | None = None,
) -> Path | None:
    skip_indices = skip_indices or set()
    missing = []

    for entry in entries:
        if entry.index in skip_indices:
            continue

        output_path = get_tts_output_path(output_dir, entry.index)
        if not is_audio_done(output_path, min_bytes=min_audio_bytes):
            missing.append(
                {
                    "index": entry.index,
                    "start_ms": entry.start_ms,
                    "end_ms": entry.end_ms,
                    "duration_ms": entry.duration_ms,
                    "text": entry.content,
                    "tts_text": text_for_tts(entry.content),
                    "output_path": str(output_path),
                }
            )

    if not missing:
        return None

    return save_json(Path(output_dir) / "missing_audio.json", missing)


def split_entries_missing_audio(
    entries: list[SrtEntry],
    output_dir: str | Path,
    batch_size: int = 100,
    min_bytes: int = 1024,
    skip_indices: set[int] | None = None,
) -> list[list[SrtEntry]]:
    if batch_size <= 0:
        raise ValueError("batch_size phai > 0")

    output_dir = ensure_dir(output_dir)
    skip_indices = skip_indices or set()

    missing = [
        entry
        for entry in entries
        if entry.index not in skip_indices
        and not is_audio_done(get_tts_output_path(output_dir, entry.index), min_bytes=min_bytes)
    ]

    return [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]


def generate_tts_once(
    entries: list[SrtEntry],
    output_dir: str | Path,
    client: TTSClient,
    download_workers: int = 8,
    download_timeout: int = 120,
    download_retries: int = 3,
    min_audio_bytes: int = 1024,
) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in entries
        if not is_audio_done(get_tts_output_path(output_dir, entry.index), min_bytes=min_audio_bytes)
    ]

    if not entries:
        return []

    print(
        f"[CREATE] count={len(entries)} from={entries[0].index} to={entries[-1].index}",
        flush=True,
    )

    created = client.create([entry.content for entry in entries])
    queried = client.wait(created)
    audio_subtitles = queried.get("audio_subtitles") or []

    if not isinstance(audio_subtitles, list) or not audio_subtitles:
        raise RuntimeError("Khong co audio_subtitles trong TTS response")

    items = build_tts_download_items(
        audio_subtitles=audio_subtitles,
        entries=entries,
        output_dir=output_dir,
        min_bytes=min_audio_bytes,
    )

    results = download_tts_items_concurrent(
        items=items,
        max_workers=download_workers,
        timeout=download_timeout,
        retries=download_retries,
        min_bytes=min_audio_bytes,
    )

    failed = [item for item in results if item.get("status") == "download_failed"]
    if failed:
        raise RuntimeError(f"Co {len(failed)} file TTS tai loi")

    return results


def generate_tts_split_invalid_text(
    entries: list[SrtEntry],
    output_dir: str | Path,
    client: TTSClient,
    download_workers: int = 8,
    download_timeout: int = 120,
    download_retries: int = 3,
    min_audio_bytes: int = 1024,
) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in entries
        if not is_audio_done(get_tts_output_path(output_dir, entry.index), min_bytes=min_audio_bytes)
    ]

    if not entries:
        return []

    try:
        return generate_tts_once(
            entries=entries,
            output_dir=output_dir,
            client=client,
            download_workers=download_workers,
            download_timeout=download_timeout,
            download_retries=download_retries,
            min_audio_bytes=min_audio_bytes,
        )

    except TTSInvalidTextError as exc:
        if len(entries) == 1:
            entry = entries[0]
            print(
                "\n[TTSInvalidText] BO QUA "
                f"index={entry.index} "
                f"start_ms={entry.start_ms} "
                f"text={json.dumps(entry.content, ensure_ascii=False)} "
                f"error={exc}\n",
                flush=True,
            )

            return [
                result_for_failed(
                    entry,
                    output_dir,
                    str(exc),
                    error_kind="TTSInvalidText",
                    response=exc.result,
                )
            ]

        mid = len(entries) // 2
        print(
            f"[TTSInvalidText] chia batch count={len(entries)} "
            f"left={len(entries[:mid])} right={len(entries[mid:])} "
            f"from={entries[0].index} to={entries[-1].index}",
            flush=True,
        )

        left = generate_tts_split_invalid_text(
            entries=entries[:mid],
            output_dir=output_dir,
            client=client,
            download_workers=download_workers,
            download_timeout=download_timeout,
            download_retries=download_retries,
            min_audio_bytes=min_audio_bytes,
        )

        right = generate_tts_split_invalid_text(
            entries=entries[mid:],
            output_dir=output_dir,
            client=client,
            download_workers=download_workers,
            download_timeout=download_timeout,
            download_retries=download_retries,
            min_audio_bytes=min_audio_bytes,
        )

        return left + right

    except TTSTerminalError as exc:
        print(
            f"[TTS_TERMINAL_ERROR] skip batch count={len(entries)} "
            f"from={entries[0].index} to={entries[-1].index} error={exc}",
            flush=True,
        )

        return [
            result_for_failed(
                entry,
                output_dir,
                str(exc),
                error_kind="terminal_tts_error",
                response=exc.result,
            )
            for entry in entries
        ]

    except Exception as exc:
        print(
            f"[TTS_BATCH_ERROR_NO_RECREATE] skip batch count={len(entries)} "
            f"from={entries[0].index} to={entries[-1].index} error={exc}",
            flush=True,
        )

        return [
            result_for_failed(
                entry,
                output_dir,
                str(exc),
                error_kind="transient_or_download_error",
            )
            for entry in entries
        ]


def process_tts_batch(
    batch_id: int,
    entries: list[SrtEntry],
    output_dir: str | Path,
    config: TTSConfig,
    download_workers: int,
    download_timeout: int,
    download_retries: int,
    min_audio_bytes: int,
) -> BatchResult:
    with TTSClient(config) as client:
        results = generate_tts_split_invalid_text(
            entries=entries,
            output_dir=output_dir,
            client=client,
            download_workers=download_workers,
            download_timeout=download_timeout,
            download_retries=download_retries,
            min_audio_bytes=min_audio_bytes,
        )

    return BatchResult(batch_id=batch_id, items=results)


def generate_tts_from_srt(
    srt_path: str | Path,
    output_dir: str | Path,
    config: TTSConfig,
    batch_size: int = 80,
    max_tts_workers: int = 1,
    download_workers_per_batch: int = 8,
    download_timeout: int = 120,
    download_retries: int = 3,
    min_audio_bytes: int = 1024,
    checkpoint_every: int = 1,
) -> list[dict[str, Any]]:
    output_dir = ensure_dir(output_dir)
    entries = read_srt_entries(srt_path)

    if not entries:
        raise ValueError("SRT rong hoac khong doc duoc")

    invalid_indices = load_invalid_indices(output_dir)

    batches = split_entries_missing_audio(
        entries=entries,
        output_dir=output_dir,
        batch_size=batch_size,
        min_bytes=min_audio_bytes,
        skip_indices=invalid_indices,
    )

    existing_results = [
        result_for_existing(entry, output_dir)
        for entry in entries
        if is_audio_done(get_tts_output_path(output_dir, entry.index), min_bytes=min_audio_bytes)
    ]

    skipped_invalid_results = [
        item
        for item in load_json_list(Path(output_dir) / "bad_tts_items.json")
        if int(item.get("index", -1)) in invalid_indices
    ]

    all_results: list[dict[str, Any]] = list(existing_results) + skipped_invalid_results

    print("Total entries:", len(entries), flush=True)
    print("Existing audio:", len(existing_results), flush=True)
    print("Skipped invalid text:", len(skipped_invalid_results), flush=True)
    print("TTS batches:", len(batches), flush=True)
    print("TTS workers:", min(max_tts_workers, max(1, len(batches))), flush=True)
    print("Download workers/batch:", download_workers_per_batch, flush=True)

    if not batches:
        final_results = merge_unique_results(all_results)
        save_manifest(output_dir, final_results)
        save_bad_items(output_dir, final_results)
        save_missing_audio(output_dir, entries, min_audio_bytes, invalid_indices)
        return final_results

    workers = max(1, min(int(max_tts_workers), len(batches)))
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_tts_batch,
                batch_id,
                batch,
                output_dir,
                config,
                download_workers_per_batch,
                download_timeout,
                download_retries,
                min_audio_bytes,
            )
            for batch_id, batch in enumerate(batches, start=1)
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="TTS batches"):
            result = future.result()
            all_results.extend(result.items)
            completed += 1

            if completed % max(1, checkpoint_every) == 0:
                checkpoint_results = merge_unique_results(all_results)
                save_manifest(output_dir, checkpoint_results, name="manifest.checkpoint.json")
                save_bad_items(output_dir, checkpoint_results)

    final_results = merge_unique_results(all_results)

    save_manifest(output_dir, final_results)
    save_bad_items(output_dir, final_results)

    invalid_indices = load_invalid_indices(output_dir)
    save_missing_audio(output_dir, entries, min_audio_bytes, invalid_indices)

    return final_results