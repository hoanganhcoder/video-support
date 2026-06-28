from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import srt
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm


TTS_INVALID_CODES = {"40402002"}
TTS_INVALID_MESSAGES = {"TTSInvalidText"}
_download_local = threading.local()


@dataclass(frozen=True)
class TTSConfig:
    api_base: str
    voice: str
    resource_id: str
    rate: str = "0"
    timeout: int = 120
    poll_interval: float = 5.0
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
    def start_ms(self) -> int:
        return max(0, int(self.start.total_seconds() * 1000))

    @property
    def end_ms(self) -> int:
        return max(0, int(self.end.total_seconds() * 1000))

    @property
    def duration_ms(self) -> int:
        return max(0, int((self.end - self.start).total_seconds() * 1000))


@dataclass(frozen=True)
class TTSJob:
    job_id: str
    entries: list[SrtEntry]
    depth: int = 0


class TTSInvalidTextError(RuntimeError):
    def __init__(self, message: str, result: dict[str, Any] | None = None):
        super().__init__(message)
        self.result = result or {}


class TTSTerminalError(RuntimeError):
    def __init__(self, message: str, result: dict[str, Any] | None = None):
        super().__init__(message)
        self.result = result or {}


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


def atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return path


def save_json(path: str | Path, data: Any) -> Path:
    return atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def clean_one_line(text: str) -> str:
    text = str(text or "").replace("\ufeff", "")
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"<[^>]{1,120}>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_for_tts(text: str) -> str:
    return clean_one_line(text) or "."


def read_srt_entries(path: str | Path) -> list[SrtEntry]:
    path = require_file(path)
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    entries: list[SrtEntry] = []

    for sub in srt.parse(raw):
        content = clean_one_line(sub.content)
        if content:
            entries.append(SrtEntry(index=int(sub.index or 0), start=sub.start, end=sub.end, content=content))

    entries.sort(key=lambda x: (x.start_ms, x.index))
    validate_entries(entries)
    return entries


def validate_entries(entries: list[SrtEntry]) -> None:
    seen, duplicated = set(), []
    for entry in entries:
        if entry.index in seen:
            duplicated.append(entry.index)
        seen.add(entry.index)
    if duplicated:
        raise ValueError(f"SRT co index trung: {duplicated[:20]}")


def audio_path(output_dir: str | Path, index: int) -> Path:
    return Path(output_dir) / f"{int(index):05d}.mp3"


def is_audio_done(path: str | Path, min_bytes: int = 1024) -> bool:
    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def make_session(pool_connections: int, pool_maxsize: int) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize, max_retries=0, pool_block=True)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_session() -> requests.Session:
    session = getattr(_download_local, "session", None)
    if session is None:
        session = make_session(8, 8)
        _download_local.session = session
    return session


def task_payload(data: dict[str, Any]) -> dict[str, Any]:
    tasks = (((data or {}).get("response") or {}).get("data") or {}).get("tasks") or []
    if tasks and isinstance(tasks[0], dict):
        return tasks[0]
    tasks = (((data or {}).get("data") or {}).get("tasks") or [])
    if tasks and isinstance(tasks[0], dict):
        return tasks[0]
    return {}


def task_id_from(data: dict[str, Any]) -> str:
    return str(data.get("task_id") or task_payload(data).get("id") or "")


def task_token_from(data: dict[str, Any]) -> str:
    return str(data.get("token") or task_payload(data).get("token") or "")


def task_status(data: dict[str, Any]) -> str:
    return str(data.get("status") or task_payload(data).get("status") or "").lower()


def task_error(data: dict[str, Any]) -> tuple[str, str]:
    task = task_payload(data)
    code = data.get("err_code") or task.get("err_code") or ""
    msg = data.get("err_msg") or task.get("err_msg") or ""
    return str(code), str(msg)


def is_invalid_text(data: dict[str, Any]) -> bool:
    code, msg = task_error(data)
    return code in TTS_INVALID_CODES or msg in TTS_INVALID_MESSAGES


def compact_error(data: dict[str, Any]) -> str:
    code, msg = task_error(data)
    return f"task_id={task_id_from(data)} status={task_status(data)} err_code={code} err_msg={msg}"


def estimated_sleep(data: dict[str, Any]) -> float:
    task = task_payload(data)
    try:
        ms = int(task.get("estimated_time") or data.get("estimated_time") or 0)
    except Exception:
        ms = 0
    return min(12.0, max(1.0, ms / 1000.0)) if ms > 0 else 0.0


def speech_url(item: dict[str, Any]) -> str | None:
    return item.get("speech_url") or item.get("url") or item.get("audio_url")


class TTSClient:
    def __init__(self, config: TTSConfig):
        self.config = config
        self.api_base = config.api_base.rstrip("/")
        if not self.api_base:
            raise ValueError("Thieu api_base")
        if not config.voice:
            raise ValueError("Thieu voice")
        if not config.resource_id:
            raise ValueError("Thieu resource_id")
        self.session = make_session(config.pool_connections, config.pool_maxsize)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> TTSClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(f"{self.api_base}{endpoint}", json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Response khong phai JSON object: {endpoint}")
        return data

    def create(self, texts: list[str]) -> dict[str, Any]:
        clean_texts = [text_for_tts(x) for x in texts]
        if not clean_texts:
            raise ValueError("texts rong")
        payload = {"texts": clean_texts, "voice": self.config.voice, "resource_id": self.config.resource_id, "rate": str(self.config.rate)}
        return self.post("/tts/create", payload)

    def query(self, created: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {"task_id": task_id_from(created), "token": task_token_from(created), "bind_id": str(created.get("bind_id") or "")}
        if created.get("identity_index") is not None:
            payload["identity_index"] = int(created["identity_index"])
        if not payload["task_id"]:
            raise RuntimeError(f"TTS create response thieu task_id: {created}")
        return self.post("/tts/query", payload)

    def wait_done(self, created: dict[str, Any]) -> dict[str, Any]:
        first_sleep = estimated_sleep(created)
        if first_sleep > 0:
            time.sleep(first_sleep)

        interval = max(1.0, float(self.config.poll_interval))

        for _ in range(1, self.config.max_polls + 1):
            result = self.query(created)
            status = task_status(result)

            if status == "succeed":
                return result

            if status in {"failed", "fail", "error"}:
                err = compact_error(result)
                if is_invalid_text(result):
                    raise TTSInvalidTextError(err, result)
                raise TTSTerminalError(err, result)

            sleep_s = estimated_sleep(result) or interval
            time.sleep(sleep_s)
            interval = min(20.0, interval * 1.2)

        raise TimeoutError(f"TTS polling timeout task_id={task_id_from(created)}")


def download_binary(url: str, output_path: str | Path, timeout: int = 120, retries: int = 3, min_bytes: int = 1024) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_audio_done(output_path, min_bytes):
        return

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        tmp = output_path.with_name(f"{output_path.name}.part.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}")
        try:
            with download_session().get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with tmp.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=512 * 1024):
                        if chunk:
                            file.write(chunk)

            if tmp.stat().st_size < min_bytes:
                raise RuntimeError(f"Downloaded file too small: {tmp.stat().st_size} bytes")

            os.replace(tmp, output_path)
            return

        except Exception as exc:
            last_error = exc
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt < retries:
                time.sleep(min(8.0, 0.5 * attempt))

    raise RuntimeError(f"Download failed: {output_path}. Error: {last_error}")


def download_one(item: dict[str, Any], timeout: int, retries: int, min_bytes: int) -> dict[str, Any]:
    out = Path(item["output_path"])
    if is_audio_done(out, min_bytes):
        return {**item, "status": "skipped_existing"}
    try:
        download_binary(item["speech_url"], out, timeout, retries, min_bytes)
        return {**item, "status": "downloaded"}
    except Exception as exc:
        return {**item, "status": "download_failed", "error": str(exc)}


def download_many(items: list[dict[str, Any]], workers: int, timeout: int, retries: int, min_bytes: int) -> list[dict[str, Any]]:
    if not items:
        return []

    results: list[dict[str, Any]] = []
    workers = max(1, min(int(workers), len(items)))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(download_one, item, timeout, retries, min_bytes) for item in items}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())

    return sorted(results, key=lambda x: int(x["index"]))


def build_download_items(audio_subtitles: list[dict[str, Any]], entries: list[SrtEntry], output_dir: str | Path, min_bytes: int) -> list[dict[str, Any]]:
    if len(audio_subtitles) < len(entries):
        raise RuntimeError(f"TTS response thieu audio: {len(audio_subtitles)}/{len(entries)}")

    items: list[dict[str, Any]] = []

    for entry, audio in zip(entries, audio_subtitles):
        out = audio_path(output_dir, entry.index)
        if is_audio_done(out, min_bytes):
            continue

        url = speech_url(audio)
        if not url:
            raise RuntimeError(f"Missing speech_url for subtitle index {entry.index}")

        items.append({"index": entry.index, "start_ms": entry.start_ms, "end_ms": entry.end_ms, "duration_ms": entry.duration_ms, "text": entry.content, "tts_text": text_for_tts(entry.content), "speech_url": url, "output_path": str(out)})

    return items


def result_existing(entry: SrtEntry, output_dir: str | Path) -> dict[str, Any]:
    return {"index": entry.index, "start_ms": entry.start_ms, "end_ms": entry.end_ms, "duration_ms": entry.duration_ms, "text": entry.content, "tts_text": text_for_tts(entry.content), "output_path": str(audio_path(output_dir, entry.index)), "status": "skipped_existing"}


def result_bad(entry: SrtEntry, output_dir: str | Path, error: str, kind: str, raw: dict[str, Any] | None = None) -> dict[str, Any]:
    status = "bad_text_skipped" if kind == "TTSInvalidText" else "tts_failed"
    return {"index": entry.index, "start_ms": entry.start_ms, "end_ms": entry.end_ms, "duration_ms": entry.duration_ms, "text": entry.content, "tts_text": text_for_tts(entry.content), "output_path": str(audio_path(output_dir, entry.index)), "status": status, "error_kind": kind, "error": error, "response": raw or {}}


def merge_by_index(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {"downloaded": 5, "skipped_existing": 4, "bad_text_skipped": 3, "tts_failed": 2, "download_failed": 1}
    by_index: dict[int, dict[str, Any]] = {}

    for item in items:
        try:
            index = int(item["index"])
        except Exception:
            continue

        old = by_index.get(index)
        if old is None or rank.get(item.get("status"), 0) >= rank.get(old.get("status"), 0):
            by_index[index] = item

    return [by_index[i] for i in sorted(by_index)]


def save_manifest(output_dir: str | Path, items: list[dict[str, Any]], name: str = "manifest.json") -> Path:
    return save_json(Path(output_dir) / name, merge_by_index(items))


def save_bad_items(output_dir: str | Path, items: list[dict[str, Any]]) -> Path | None:
    path = Path(output_dir) / "bad_tts_items.json"
    old = load_json_list(path)
    bad = [x for x in items if x.get("status") in {"bad_text_skipped", "tts_failed", "download_failed"}]
    merged = [x for x in merge_by_index(old + bad) if x.get("status") in {"bad_text_skipped", "tts_failed", "download_failed"}]

    if not merged:
        return None

    return save_json(path, merged)


def bad_text_indices(output_dir: str | Path) -> set[int]:
    indices: set[int] = set()

    for item in load_json_list(Path(output_dir) / "bad_tts_items.json"):
        if item.get("status") != "bad_text_skipped":
            continue
        try:
            indices.add(int(item["index"]))
        except Exception:
            pass

    return indices


def split_batches(entries: list[SrtEntry], output_dir: str | Path, batch_size: int, min_bytes: int, skip_indices: set[int]) -> list[list[SrtEntry]]:
    missing = [e for e in entries if e.index not in skip_indices and not is_audio_done(audio_path(output_dir, e.index), min_bytes)]
    return [missing[i:i + batch_size] for i in range(0, len(missing), batch_size)]


def generate_once(entries: list[SrtEntry], output_dir: str | Path, config: TTSConfig, download_workers: int, download_timeout: int, download_retries: int, min_bytes: int) -> list[dict[str, Any]]:
    entries = [e for e in entries if not is_audio_done(audio_path(output_dir, e.index), min_bytes)]

    if not entries:
        return []

    print(f"[CREATE] count={len(entries)} range={entries[0].index}..{entries[-1].index}", flush=True)

    with TTSClient(config) as client:
        created = client.create([e.content for e in entries])
        queried = client.wait_done(created)

    audio_subtitles = queried.get("audio_subtitles") or []
    if not isinstance(audio_subtitles, list) or not audio_subtitles:
        raise RuntimeError("Khong co audio_subtitles trong TTS response")

    items = build_download_items(audio_subtitles, entries, output_dir, min_bytes)
    return download_many(items, download_workers, download_timeout, download_retries, min_bytes)


def run_job(job: TTSJob, output_dir: str | Path, config: TTSConfig, download_workers: int, download_timeout: int, download_retries: int, min_bytes: int) -> dict[str, Any]:
    try:
        items = generate_once(job.entries, output_dir, config, download_workers, download_timeout, download_retries, min_bytes)
        return {"kind": "done", "job": job, "items": items}

    except TTSInvalidTextError as exc:
        if len(job.entries) > 1:
            mid = len(job.entries) // 2
            left, right = job.entries[:mid], job.entries[mid:]
            print(f"[TTSInvalidText] split {job.job_id} count={len(job.entries)} -> {len(left)}/{len(right)} range={job.entries[0].index}..{job.entries[-1].index}", flush=True)
            return {"kind": "split", "job": job, "left": left, "right": right}

        entry = job.entries[0]
        item = result_bad(entry, output_dir, str(exc), "TTSInvalidText", exc.result)
        print(f"\n[TTSInvalidText] BO QUA index={entry.index} start_ms={entry.start_ms} text={json.dumps(entry.content, ensure_ascii=False)} error={exc}\n", flush=True)
        return {"kind": "done", "job": job, "items": [item]}

    except TTSTerminalError as exc:
        items = [result_bad(e, output_dir, str(exc), "terminal_tts_error", exc.result) for e in job.entries]
        print(f"[TTS_TERMINAL] skip {job.job_id}: {exc}", flush=True)
        return {"kind": "done", "job": job, "items": items}

    except Exception as exc:
        items = [result_bad(e, output_dir, str(exc), "transient_or_download_error") for e in job.entries]
        print(f"[TTS_ERROR] skip {job.job_id}: {exc}", flush=True)
        return {"kind": "done", "job": job, "items": items}


def save_missing_audio(output_dir: str | Path, entries: list[SrtEntry], min_bytes: int, skip_indices: set[int]) -> Path | None:
    missing = []

    for e in entries:
        if e.index in skip_indices:
            continue

        out = audio_path(output_dir, e.index)
        if not is_audio_done(out, min_bytes):
            missing.append({"index": e.index, "start_ms": e.start_ms, "end_ms": e.end_ms, "duration_ms": e.duration_ms, "text": e.content, "tts_text": text_for_tts(e.content), "output_path": str(out)})

    if not missing:
        return None

    return save_json(Path(output_dir) / "missing_audio.json", missing)


def generate_tts_from_srt(
    srt_path: str | Path,
    output_dir: str | Path,
    config: TTSConfig,
    batch_size: int = 80,
    max_tts_workers: int = 6,
    download_workers_per_job: int = 8,
    download_timeout: int = 120,
    download_retries: int = 3,
    min_audio_bytes: int = 1024,
    checkpoint_every: int = 5,
) -> list[dict[str, Any]]:
    output_dir = ensure_dir(output_dir)
    entries = read_srt_entries(srt_path)
    skip_indices = bad_text_indices(output_dir)

    batches = split_batches(entries, output_dir, batch_size, min_audio_bytes, skip_indices)
    existing = [result_existing(e, output_dir) for e in entries if is_audio_done(audio_path(output_dir, e.index), min_audio_bytes)]
    skipped_bad = [x for x in load_json_list(Path(output_dir) / "bad_tts_items.json") if int(x.get("index", -1)) in skip_indices]
    results: list[dict[str, Any]] = existing + skipped_bad

    print("Total entries:", len(entries), flush=True)
    print("Existing audio:", len(existing), flush=True)
    print("Skipped bad text:", len(skipped_bad), flush=True)
    print("TTS batches:", len(batches), flush=True)
    print("TTS workers:", max_tts_workers, flush=True)
    print("Download workers/job:", download_workers_per_job, flush=True)

    if not batches:
        final = merge_by_index(results)
        save_manifest(output_dir, final)
        save_missing_audio(output_dir, entries, min_audio_bytes, skip_indices)
        return final

    pending: dict[Any, TTSJob] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=max(1, int(max_tts_workers))) as executor:
        for i, batch in enumerate(batches, start=1):
            job = TTSJob(job_id=f"batch-{i}", entries=batch)
            pending[executor.submit(run_job, job, output_dir, config, download_workers_per_job, download_timeout, download_retries, min_audio_bytes)] = job

        pbar = tqdm(total=len(pending), desc="TTS jobs")

        while pending:
            done, _ = wait(set(pending), return_when=FIRST_COMPLETED)

            for future in done:
                job = pending.pop(future)
                outcome = future.result()

                if outcome["kind"] == "split":
                    parent = outcome["job"]

                    for name, part in (("L", outcome["left"]), ("R", outcome["right"])):
                        part = [e for e in part if not is_audio_done(audio_path(output_dir, e.index), min_audio_bytes) and e.index not in bad_text_indices(output_dir)]
                        if not part:
                            continue

                        child = TTSJob(job_id=f"{parent.job_id}-{name}", entries=part, depth=parent.depth + 1)
                        pending[executor.submit(run_job, child, output_dir, config, download_workers_per_job, download_timeout, download_retries, min_audio_bytes)] = child
                        pbar.total += 1

                    pbar.refresh()

                else:
                    items = outcome.get("items") or []
                    results.extend(items)
                    save_bad_items(output_dir, items)

                completed += 1
                pbar.update(1)

                if completed % max(1, checkpoint_every) == 0:
                    save_manifest(output_dir, results, "manifest.checkpoint.json")
                    save_bad_items(output_dir, results)

        pbar.close()

    final = merge_by_index(results)
    save_manifest(output_dir, final)
    save_bad_items(output_dir, final)

    skip_indices = bad_text_indices(output_dir)
    save_missing_audio(output_dir, entries, min_audio_bytes, skip_indices)

    print("Done:", len(final), flush=True)
    print("Bad text:", len(skip_indices), flush=True)
    return final