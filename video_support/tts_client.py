from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import datetime as dt
import json
import logging
import time

import requests
import srt
from requests.adapters import HTTPAdapter


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTSConfig:
    api_base: str
    voice: str
    resource_id: str
    rate: str = "0"
    device: str | None = None
    timeout: int = 120
    poll_interval: int = 3
    max_polls: int = 120
    pool_connections: int = 64
    pool_maxsize: int = 64


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


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file: {path}")
    if not path.is_file():
        raise ValueError(f"Không phải file: {path}")
    return path


def read_srt_entries(path: str | Path) -> list[SrtEntry]:
    path = require_file(path)
    text = path.read_text(encoding="utf-8-sig", errors="ignore")

    entries: list[SrtEntry] = []

    for sub in srt.parse(text):
        content = str(sub.content or "").replace("\n", " ").strip()
        if not content:
            continue

        entries.append(
            SrtEntry(
                index=int(sub.index or 0),
                start=sub.start,
                end=sub.end,
                content=content,
            )
        )

    return sorted(entries, key=lambda e: (e.start_ms, e.index))


def write_srt_entries(entries: list[SrtEntry], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    subtitles = [
        srt.Subtitle(
            index=index,
            start=entry.start,
            end=entry.end,
            content=entry.content,
        )
        for index, entry in enumerate(entries, start=1)
    ]

    path.write_text(srt.compose(subtitles), encoding="utf-8")
    return path


def text_for_tts(text: str) -> str:
    text = str(text or "").replace("\n", " ").strip()
    return text or "."


class TTSClient:
    def __init__(self, config: TTSConfig):
        if not config.api_base:
            raise ValueError("Thiếu api_base")

        self.config = config
        self.api_base = config.api_base.rstrip("/")
        self.session = requests.Session()

        adapter = HTTPAdapter(
            pool_connections=config.pool_connections,
            pool_maxsize=config.pool_maxsize,
            max_retries=0,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> TTSClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout: int | None = None,
    ) -> dict[str, Any]:
        response = self.session.post(
            f"{self.api_base}{endpoint}",
            json=payload,
            timeout=timeout or self.config.timeout,
        )
        response.raise_for_status()
        return response.json()

    def create(
        self,
        texts: list[str],
        voice: str | None = None,
        resource_id: str | None = None,
        rate: str | None = None,
        device: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "texts": texts,
            "voice": voice or self.config.voice,
            "resource_id": resource_id or self.config.resource_id,
            "rate": str(rate if rate is not None else self.config.rate),
        }

        used_device = device if device is not None else self.config.device
        if used_device:
            payload["device"] = used_device

        return self.post("/tts/create", payload)

    def query(
        self,
        task_id: str,
        token: str | None,
        bind_id: str = "",
        device: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "task_id": task_id,
            "token": token,
            "bind_id": bind_id,
        }

        if device:
            payload["device"] = device

        return self.post("/tts/query", payload)

    def wait(
        self,
        created: dict[str, Any],
        poll_interval: int | None = None,
        max_polls: int | None = None,
    ) -> dict[str, Any]:
        task_id = created["task_id"]
        token = created.get("token")
        bind_id = created.get("bind_id") or ""
        device = created.get("device")

        poll_interval = poll_interval or self.config.poll_interval
        max_polls = max_polls or self.config.max_polls

        for attempt in range(1, max_polls + 1):
            result = self.query(
                task_id=task_id,
                token=token,
                bind_id=bind_id,
                device=device,
            )

            status = result.get("status")
            logger.info("TTS poll %s/%s: %s", attempt, max_polls, status)

            if status == "succeed":
                return result

            if status in {"failed", "fail", "error"}:
                raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))

            time.sleep(poll_interval)

        raise TimeoutError("TTS polling timeout")


def get_tts_output_path(output_dir: str | Path, index: int) -> Path:
    return Path(output_dir) / f"{int(index):05d}.mp3"


def is_audio_done(path: str | Path, min_bytes: int = 1024) -> bool:
    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


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

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    tmp_path.unlink(missing_ok=True)

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)

    tmp_path.replace(output_path)
    return output_path


def download_binary_with_retry(
    url: str,
    output_path: str | Path,
    timeout: int = 120,
    retries: int = 3,
    sleep_seconds: float = 0.5,
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
        except Exception as error:
            last_error = error
            tmp_path = Path(output_path).with_suffix(Path(output_path).suffix + ".part")
            tmp_path.unlink(missing_ok=True)

            if attempt < retries:
                time.sleep(sleep_seconds * attempt)

    raise RuntimeError(
        f"Download failed after {retries} attempts: {output_path}. Error: {last_error}"
    )


def download_tts_item(
    item: dict[str, Any],
    timeout: int = 120,
    retries: int = 3,
    min_bytes: int = 1024,
) -> dict[str, Any]:
    output_path = Path(item["output_path"])

    if is_audio_done(output_path, min_bytes=min_bytes):
        return {**item, "status": "skipped"}

    try:
        download_binary_with_retry(
            url=item["speech_url"],
            output_path=output_path,
            timeout=timeout,
            retries=retries,
            min_bytes=min_bytes,
        )
        return {**item, "status": "downloaded"}
    except Exception as error:
        return {**item, "status": "failed", "error": str(error)}


def download_tts_items_concurrent(
    items: list[dict[str, Any]],
    max_workers: int = 32,
    timeout: int = 120,
    retries: int = 3,
    min_bytes: int = 1024,
) -> list[dict[str, Any]]:
    if not items:
        return []

    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_tts_item, item, timeout, retries, min_bytes): item
            for item in items
        }

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
    items: list[dict[str, Any]] = []

    for local_index, item in enumerate(audio_subtitles):
        if local_index >= len(entries):
            break

        entry = entries[local_index]
        output_path = get_tts_output_path(output_dir, entry.index)

        if is_audio_done(output_path, min_bytes=min_bytes):
            continue

        speech_url = extract_speech_url(item)
        if not speech_url:
            logger.warning("Missing speech_url for subtitle index %s", entry.index)
            continue

        items.append(
            {
                "index": entry.index,
                "text": entry.content,
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
    output_dir = ensure_dir(output_dir)
    missing = [
        entry
        for entry in entries
        if not is_audio_done(get_tts_output_path(output_dir, entry.index), min_bytes=min_bytes)
    ]

    return [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]


def save_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_bad_items(output_dir: str | Path, bad_items: list[dict[str, Any]]) -> Path:
    return save_json(Path(output_dir) / "bad_tts_items.json", bad_items)


def generate_tts_for_entries_resilient(
    entries: list[SrtEntry],
    output_dir: str | Path,
    client: TTSClient,
    download_workers: int = 32,
    download_timeout: int = 120,
    download_retries: int = 3,
    wait_after_success_seconds: float = 5.0,
    min_existing_audio_bytes: int = 1024,
    bad_items: list[dict[str, Any]] | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    output_dir = ensure_dir(output_dir)
    bad_items = bad_items if bad_items is not None else []

    entries = [
        entry
        for entry in entries
        if not is_audio_done(
            get_tts_output_path(output_dir, entry.index),
            min_bytes=min_existing_audio_bytes,
        )
    ]

    if not entries:
        return []

    try:
        created = client.create([text_for_tts(entry.content) for entry in entries])
        queried = client.wait(created)

        audio_subtitles = queried.get("audio_subtitles") or []
        if not audio_subtitles:
            raise RuntimeError("Không có audio_subtitles trong TTS response")

        items = build_tts_download_items(
            audio_subtitles=audio_subtitles,
            entries=entries,
            output_dir=output_dir,
            min_bytes=min_existing_audio_bytes,
        )

        results = download_tts_items_concurrent(
            items=items,
            max_workers=download_workers,
            timeout=download_timeout,
            retries=download_retries,
            min_bytes=min_existing_audio_bytes,
        )

        failed = [item for item in results if item.get("status") == "failed"]
        if failed:
            failed_path = save_json(Path(output_dir) / "failed_downloads.json", failed)
            raise RuntimeError(f"Có {len(failed)} file TTS tải lỗi. Xem: {failed_path}")

        if wait_after_success_seconds > 0:
            time.sleep(wait_after_success_seconds)

        return results

    except Exception as exc:
        error = str(exc)

        if len(entries) == 1:
            entry = entries[0]

            bad_item = {
                "index": entry.index,
                "start_ms": entry.start_ms,
                "end_ms": entry.end_ms,
                "duration_ms": entry.duration_ms,
                "text": entry.content,
                "tts_text": text_for_tts(entry.content),
                "error": error,
            }

            bad_items.append(bad_item)
            save_bad_items(output_dir, bad_items)

            return [
                {
                    "index": entry.index,
                    "text": entry.content,
                    "tts_text": text_for_tts(entry.content),
                    "output_path": str(get_tts_output_path(output_dir, entry.index)),
                    "status": "bad_text_skipped",
                    "error": error,
                }
            ]

        mid = len(entries) // 2

        left = generate_tts_for_entries_resilient(
            entries=entries[:mid],
            output_dir=output_dir,
            client=client,
            download_workers=download_workers,
            download_timeout=download_timeout,
            download_retries=download_retries,
            wait_after_success_seconds=wait_after_success_seconds,
            min_existing_audio_bytes=min_existing_audio_bytes,
            bad_items=bad_items,
            depth=depth + 1,
        )

        right = generate_tts_for_entries_resilient(
            entries=entries[mid:],
            output_dir=output_dir,
            client=client,
            download_workers=download_workers,
            download_timeout=download_timeout,
            download_retries=download_retries,
            wait_after_success_seconds=wait_after_success_seconds,
            min_existing_audio_bytes=min_existing_audio_bytes,
            bad_items=bad_items,
            depth=depth + 1,
        )

        return left + right


def generate_tts_from_srt(
    srt_path: str | Path,
    output_dir: str | Path,
    config: TTSConfig | None = None,
    client: TTSClient | None = None,
    batch_size: int = 100,
    download_workers: int = 32,
    download_timeout: int = 120,
    download_retries: int = 3,
    wait_after_batch_seconds: float = 5.0,
    min_existing_audio_bytes: int = 1024,
) -> list[dict[str, Any]]:
    if client is None and config is None:
        raise ValueError("Cần truyền config hoặc client")

    output_dir = ensure_dir(output_dir)
    entries = read_srt_entries(srt_path)

    owns_client = client is None
    client = client or TTSClient(config)  # type: ignore[arg-type]

    batches = split_entries_missing_audio(
        entries=entries,
        output_dir=output_dir,
        batch_size=batch_size,
        min_bytes=min_existing_audio_bytes,
    )

    all_results: list[dict[str, Any]] = []
    bad_items: list[dict[str, Any]] = []

    try:
        for batch_entries in batches:
            all_results.extend(
                generate_tts_for_entries_resilient(
                    entries=batch_entries,
                    output_dir=output_dir,
                    client=client,
                    download_workers=download_workers,
                    download_timeout=download_timeout,
                    download_retries=download_retries,
                    wait_after_success_seconds=wait_after_batch_seconds,
                    min_existing_audio_bytes=min_existing_audio_bytes,
                    bad_items=bad_items,
                )
            )
    finally:
        if owns_client:
            client.close()

    done_indexes = {int(item["index"]) for item in all_results if "index" in item}
    existing_results = [
        {
            "index": entry.index,
            "text": entry.content,
            "output_path": str(get_tts_output_path(output_dir, entry.index)),
            "status": "skipped_existing",
        }
        for entry in entries
        if entry.index not in done_indexes
        and is_audio_done(
            get_tts_output_path(output_dir, entry.index),
            min_bytes=min_existing_audio_bytes,
        )
    ]

    final_results = sorted(all_results + existing_results, key=lambda item: int(item["index"]))

    save_json(Path(output_dir) / "manifest.json", final_results)

    if bad_items:
        save_bad_items(output_dir, bad_items)

    return final_results


__all__ = [
    "SrtEntry",
    "TTSConfig",
    "TTSClient",
    "build_tts_download_items",
    "download_binary_once",
    "download_binary_with_retry",
    "download_tts_item",
    "download_tts_items_concurrent",
    "ensure_dir",
    "extract_speech_url",
    "generate_tts_for_entries_resilient",
    "generate_tts_from_srt",
    "get_tts_output_path",
    "is_audio_done",
    "read_srt_entries",
    "require_file",
    "save_bad_items",
    "save_json",
    "split_entries_missing_audio",
    "text_for_tts",
    "write_srt_entries",
]