from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
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


@dataclass(frozen=True)
class TTSConfig:
    api_base: str
    voice: str
    resource_id: str
    rate: str = "0"
    timeout: int = 120
    request_retries: int = 3
    poll_interval: float = 3.0
    max_polls: int = 120
    pool_connections: int = 32
    pool_maxsize: int = 32


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
    return " ".join(str(text or "").replace("\ufeff", "").split())


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
        pool_block=False,
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

    def post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_base}{endpoint}"
        last_error: Exception | None = None

        for attempt in range(1, self.config.request_retries + 1):
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
                if attempt >= self.config.request_retries:
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

        return self.post_json("/tts/query", payload)

    def wait(self, created: dict[str, Any]) -> dict[str, Any]:
        task_id = str(created.get("task_id") or "")
        token = str(created.get("token") or "")
        bind_id = str(created.get("bind_id") or "")
        identity_index = created.get("identity_index")

        if not task_id:
            raise RuntimeError(f"TTS create response thieu task_id: {created}")

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
                raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))

            time.sleep(self.config.poll_interval)

        raise TimeoutError(f"TTS polling timeout task_id={task_id}")


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


def split_entries_missing_audio(
    entries: list[SrtEntry],
    output_dir: str | Path,
    batch_size: int = 100,
    min_bytes: int = 1024,
) -> list[list[SrtEntry]]:
    if batch_size <= 0:
        raise ValueError("batch_size phai > 0")

    output_dir = ensure_dir(output_dir)
    missing = [
        entry
        for entry in entries
        if not is_audio_done(get_tts_output_path(output_dir, entry.index), min_bytes=min_bytes)
    ]

    return [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]


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


def result_for_failed(entry: SrtEntry, output_dir: str | Path, error: str) -> dict[str, Any]:
    return {
        "index": entry.index,
        "start_ms": entry.start_ms,
        "end_ms": entry.end_ms,
        "duration_ms": entry.duration_ms,
        "text": entry.content,
        "tts_text": text_for_tts(entry.content),
        "output_path": str(get_tts_output_path(output_dir, entry.index)),
        "status": "tts_failed",
        "error": error,
    }


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


def generate_tts_resilient(
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
    except Exception as exc:
        error = str(exc)

        if len(entries) == 1:
            return [result_for_failed(entries[0], output_dir, error)]

        mid = len(entries) // 2
        left = generate_tts_resilient(
            entries=entries[:mid],
            output_dir=output_dir,
            client=client,
            download_workers=download_workers,
            download_timeout=download_timeout,
            download_retries=download_retries,
            min_audio_bytes=min_audio_bytes,
        )
        right = generate_tts_resilient(
            entries=entries[mid:],
            output_dir=output_dir,
            client=client,
            download_workers=download_workers,
            download_timeout=download_timeout,
            download_retries=download_retries,
            min_audio_bytes=min_audio_bytes,
        )
        return left + right


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
        results = generate_tts_resilient(
            entries=entries,
            output_dir=output_dir,
            client=client,
            download_workers=download_workers,
            download_timeout=download_timeout,
            download_retries=download_retries,
            min_audio_bytes=min_audio_bytes,
        )

    return BatchResult(batch_id=batch_id, items=results)


def save_manifest(output_dir: str | Path, results: list[dict[str, Any]], name: str = "manifest.json") -> Path:
    results = sorted(results, key=lambda item: int(item.get("index", 0)))
    return save_json(Path(output_dir) / name, results)


def save_bad_items(output_dir: str | Path, results: list[dict[str, Any]]) -> Path | None:
    bad_items = [item for item in results if item.get("status") in {"tts_failed", "download_failed"}]
    if not bad_items:
        return None
    return save_json(Path(output_dir) / "bad_tts_items.json", bad_items)


def save_missing_audio(output_dir: str | Path, entries: list[SrtEntry], min_audio_bytes: int = 1024) -> Path | None:
    missing = []

    for entry in entries:
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


def generate_tts_from_srt(
    srt_path: str | Path,
    output_dir: str | Path,
    config: TTSConfig,
    batch_size: int = 80,
    max_tts_workers: int = 4,
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

    batches = split_entries_missing_audio(
        entries=entries,
        output_dir=output_dir,
        batch_size=batch_size,
        min_bytes=min_audio_bytes,
    )

    existing_results = [
        result_for_existing(entry, output_dir)
        for entry in entries
        if is_audio_done(get_tts_output_path(output_dir, entry.index), min_bytes=min_audio_bytes)
    ]

    all_results: list[dict[str, Any]] = list(existing_results)

    print("Total entries:", len(entries), flush=True)
    print("Existing audio:", len(existing_results), flush=True)
    print("TTS batches:", len(batches), flush=True)
    print("TTS workers:", min(max_tts_workers, max(1, len(batches))), flush=True)
    print("Download workers/batch:", download_workers_per_batch, flush=True)

    if not batches:
        save_manifest(output_dir, all_results)
        save_missing_audio(output_dir, entries, min_audio_bytes)
        return sorted(all_results, key=lambda item: int(item["index"]))

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

        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="TTS batches"):
                result = future.result()
                all_results.extend(result.items)
                completed += 1

                if completed % max(1, checkpoint_every) == 0:
                    save_manifest(output_dir, all_results, name="manifest.checkpoint.json")
        except Exception:
            for future in futures:
                future.cancel()
            save_manifest(output_dir, all_results, name="manifest.failed_checkpoint.json")
            save_bad_items(output_dir, all_results)
            save_missing_audio(output_dir, entries, min_audio_bytes)
            raise

    final_results = sorted(all_results, key=lambda item: int(item["index"]))
    save_manifest(output_dir, final_results)
    save_bad_items(output_dir, final_results)
    save_missing_audio(output_dir, entries, min_audio_bytes)

    return final_results


