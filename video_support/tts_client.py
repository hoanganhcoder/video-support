from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import srt


BAD_TEXT_CODES = {"40402002"}
BAD_TEXT_MESSAGES = {"TTSInvalidText"}
BAD_STATUSES = {"bad_text_skipped", "skipped_no_audio", "tts_failed", "download_failed"}


@dataclass(frozen=True)
class TTSConfig:
    api_base: str
    voice: str
    resource_id: str
    rate: str = "1.0"
    timeout: int = 120
    poll_interval: float = 2.0
    max_polls: int = 120
    request_retries: int = 3
    retry_status_codes: tuple[int, ...] = (500, 502)
    retry_sleep: float = 1.0
    download_retries: int = 3
    max_connections: int = 128
    max_keepalive: int = 64


@dataclass(frozen=True)
class Entry:
    idx: int
    start: dt.timedelta
    end: dt.timedelta
    text: str

    @property
    def start_ms(self) -> int:
        return max(0, int(self.start.total_seconds() * 1000))

    @property
    def end_ms(self) -> int:
        return max(0, int(self.end.total_seconds() * 1000))

    @property
    def duration_ms(self) -> int:
        return max(0, int((self.end - self.start).total_seconds() * 1000))


class BadTextError(RuntimeError):
    def __init__(self, msg: str, data: dict[str, Any]):
        super().__init__(msg)
        self.data = data


class TerminalTTSError(RuntimeError):
    def __init__(self, msg: str, data: dict[str, Any]):
        super().__init__(msg)
        self.data = data


class MissingUrlError(RuntimeError):
    def __init__(self, entry: Entry, msg: str):
        super().__init__(msg)
        self.entry = entry


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_text(text: str) -> str:
    return " ".join(str(text or "").replace("\ufeff", " ").split())


def tts_text(text: str) -> str:
    return clean_text(text) or "."


def no_audio_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", clean_text(text))
    if not compact:
        return True
    if compact in {".", "..", "...", "…", "……", "。", "。。", "。。。", "-", "--", "---", "—", "——"}:
        return True
    return all(ch in ".…。·•・-—_~" for ch in compact)


def read_entries(path: str | Path) -> list[Entry]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Khong thay file: {path}")

    entries = []
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")

    for sub in srt.parse(raw):
        text = clean_text(sub.content)
        if text:
            entries.append(Entry(int(sub.index or 0), sub.start, sub.end, text))

    entries.sort(key=lambda e: (e.start_ms, e.idx))
    seen, dup = set(), []

    for e in entries:
        if e.idx in seen:
            dup.append(e.idx)
        seen.add(e.idx)

    if dup:
        raise ValueError(f"SRT co idx trung: {dup[:20]}")

    return entries


def audio_path(output_dir: str | Path, idx: int) -> Path:
    return Path(output_dir) / f"{int(idx):05d}.mp3"


def audio_done(path: str | Path, min_bytes: int) -> bool:
    path = Path(path)
    return path.is_file() and path.stat().st_size >= min_bytes


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json_list(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def task_obj(data: dict[str, Any]) -> dict[str, Any]:
    tasks = (((data or {}).get("response") or {}).get("data") or {}).get("tasks") or []
    if tasks and isinstance(tasks[0], dict):
        return tasks[0]
    tasks = (((data or {}).get("data") or {}).get("tasks") or [])
    if tasks and isinstance(tasks[0], dict):
        return tasks[0]
    return {}


def task_id(data: dict[str, Any]) -> str:
    return str(data.get("task_id") or task_obj(data).get("id") or "")


def task_token(data: dict[str, Any]) -> str:
    return str(data.get("token") or task_obj(data).get("token") or "")


def task_bind_id(data: dict[str, Any]) -> str:
    return str(data.get("bind_id") or task_obj(data).get("bind_id") or "")


def task_status(data: dict[str, Any]) -> str:
    return str(data.get("status") or task_obj(data).get("status") or "").lower()


def task_error(data: dict[str, Any]) -> tuple[str, str]:
    obj = task_obj(data)
    return str(data.get("err_code") or obj.get("err_code") or ""), str(data.get("err_msg") or obj.get("err_msg") or "")


def is_bad_text(data: dict[str, Any]) -> bool:
    code, msg = task_error(data)
    return code in BAD_TEXT_CODES or msg in BAD_TEXT_MESSAGES


def error_text(data: dict[str, Any]) -> str:
    code, msg = task_error(data)
    return f"task_id={task_id(data)} status={task_status(data)} err_code={code} err_msg={msg}"


def estimated_sleep(data: dict[str, Any]) -> float:
    try:
        ms = int(task_obj(data).get("estimated_time") or data.get("estimated_time") or 0)
    except Exception:
        ms = 0
    return min(12.0, max(1.0, ms / 1000.0)) if ms > 0 else 0.0


def speech_url(item: dict[str, Any]) -> str | None:
    return item.get("speech_url") or item.get("url") or item.get("audio_url")


def base_item(e: Entry, output_dir: str | Path) -> dict[str, Any]:
    return {
        "idx": e.idx,
        "start_ms": e.start_ms,
        "end_ms": e.end_ms,
        "duration_ms": e.duration_ms,
        "text": e.text,
        "tts_text": tts_text(e.text),
        "output_path": str(audio_path(output_dir, e.idx)),
    }


def make_item(e: Entry, output_dir: str | Path, status: str, error: str = "", response: dict[str, Any] | None = None) -> dict[str, Any]:
    item = {**base_item(e, output_dir), "status": status}
    if error:
        item["error"] = error
    if response:
        item["response"] = response
    return item


def merge_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {"downloaded": 6, "skipped_existing": 5, "skipped_no_audio": 4, "bad_text_skipped": 3, "tts_failed": 2, "download_failed": 1}
    out: dict[int, dict[str, Any]] = {}

    for item in items:
        try:
            idx = int(item["idx"])
        except Exception:
            continue

        old = out.get(idx)
        if old is None or rank.get(item.get("status"), 0) >= rank.get(old.get("status"), 0):
            out[idx] = item

    return [out[idx] for idx in sorted(out)]


def save_manifest(output_dir: str | Path, items: list[dict[str, Any]], name: str = "manifest.json") -> None:
    write_json(Path(output_dir) / name, merge_items(items))


def save_bad(output_dir: str | Path, items: list[dict[str, Any]]) -> None:
    path = Path(output_dir) / "bad_tts_items.json"
    bad = [x for x in items if x.get("status") in BAD_STATUSES]
    merged = [x for x in merge_items(read_json_list(path) + bad) if x.get("status") in BAD_STATUSES]
    if merged:
        write_json(path, merged)


def load_skip(output_dir: str | Path) -> set[int]:
    skip = set()
    for item in read_json_list(Path(output_dir) / "bad_tts_items.json"):
        if item.get("status") not in {"bad_text_skipped", "skipped_no_audio"}:
            continue
        try:
            skip.add(int(item["idx"]))
        except Exception:
            pass
    return skip


def split_batches(entries: list[Entry], output_dir: str | Path, batch_size: int, min_bytes: int, skip: set[int]) -> list[list[Entry]]:
    missing = [e for e in entries if e.idx not in skip and not audio_done(audio_path(output_dir, e.idx), min_bytes)]
    return [missing[i:i + batch_size] for i in range(0, len(missing), batch_size)]


class TTSClient:
    def __init__(self, config: TTSConfig):
        self.config = config
        self.base = config.api_base.rstrip("/")
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout),
            limits=httpx.Limits(max_connections=config.max_connections, max_keepalive_connections=config.max_keepalive),
        )

    async def __aenter__(self) -> "TTSClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.http.aclose()

    async def post(self, endpoint: str, body: dict[str, Any], idx: int | list[int] | None = None) -> dict[str, Any]:
        url = f"{self.base}{endpoint}"
        last: Exception | None = None
        retries = max(1, int(self.config.request_retries))

        for attempt in range(1, retries + 1):
            try:
                r = await self.http.post(url, json=body)
                if r.status_code in self.config.retry_status_codes and attempt < retries:
                    print(f"[retry] {endpoint} status={r.status_code} idx={idx} {attempt}/{retries}", flush=True)
                    await asyncio.sleep(min(10.0, self.config.retry_sleep * attempt))
                    continue

                r.raise_for_status()
                data = r.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"response not json object: {endpoint}")
                return data

            except httpx.HTTPStatusError as exc:
                last = exc
                code = exc.response.status_code if exc.response else 0
                if code in self.config.retry_status_codes and attempt < retries:
                    print(f"[retry] {endpoint} status={code} idx={idx} {attempt}/{retries}", flush=True)
                    await asyncio.sleep(min(10.0, self.config.retry_sleep * attempt))
                    continue
                raise

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                if attempt < retries:
                    print(f"[retry] {endpoint} network idx={idx} {attempt}/{retries}: {exc}", flush=True)
                    await asyncio.sleep(min(10.0, self.config.retry_sleep * attempt))
                    continue
                raise

        raise RuntimeError(f"{endpoint} failed: {last}")

    async def create(self, entries: list[Entry]) -> dict[str, Any]:
        idx = [e.idx for e in entries]
        body = {"texts": [tts_text(e.text) for e in entries], "voice": self.config.voice, "resource_id": self.config.resource_id, "rate": str(self.config.rate)}
        data = await self.post("/tts/create", body, idx=idx)
        return {"idx": idx, "task_id": task_id(data), "token": task_token(data), "bind_id": task_bind_id(data), "response": data}

    async def query(self, created: dict[str, Any]) -> dict[str, Any]:
        body = {"task_id": created.get("task_id") or "", "token": created.get("token") or "", "bind_id": created.get("bind_id") or ""}
        if not body["task_id"]:
            raise TerminalTTSError("create response missing task_id", created)
        data = await self.post("/tts/query", body, idx=created.get("idx"))
        data["_idx"] = created.get("idx")
        return data

    async def wait_done(self, created: dict[str, Any]) -> dict[str, Any]:
        first_sleep = estimated_sleep(created.get("response") or {})
        if first_sleep > 0:
            await asyncio.sleep(first_sleep)

        interval = max(1.0, float(self.config.poll_interval))

        for _ in range(max(1, int(self.config.max_polls))):
            data = await self.query(created)
            status = task_status(data)

            if status == "succeed":
                return data

            if status in {"failed", "fail", "error"}:
                if is_bad_text(data):
                    raise BadTextError(error_text(data), data)
                raise TerminalTTSError(error_text(data), data)

            await asyncio.sleep(estimated_sleep(data) or interval)
            interval = min(20.0, interval * 1.2)

        raise TimeoutError(f"poll timeout task_id={created.get('task_id')} idx={created.get('idx')}")


async def download_file(http: httpx.AsyncClient, url: str, path: str | Path, config: TTSConfig, min_bytes: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if audio_done(path, min_bytes):
        return

    last: Exception | None = None

    for attempt in range(1, max(1, int(config.download_retries)) + 1):
        tmp = path.with_name(f"{path.name}.part.{os.getpid()}.{time.time_ns()}")

        try:
            async with http.stream("GET", url) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    async for chunk in r.aiter_bytes(512 * 1024):
                        if chunk:
                            f.write(chunk)

            if tmp.stat().st_size < min_bytes:
                raise RuntimeError(f"file too small: {tmp.stat().st_size}")

            os.replace(tmp, path)
            return

        except Exception as exc:
            last = exc
            tmp.unlink(missing_ok=True)
            if attempt < config.download_retries:
                await asyncio.sleep(min(8.0, 0.5 * attempt))

    raise RuntimeError(str(last))


async def download_many(client: TTSClient, rows: list[dict[str, Any]], workers: int, min_bytes: int) -> list[dict[str, Any]]:
    if not rows:
        return []

    sem = asyncio.Semaphore(max(1, int(workers)))
    results: list[dict[str, Any]] = []

    async def one(row: dict[str, Any]) -> None:
        async with sem:
            if audio_done(row["output_path"], min_bytes):
                results.append({**row, "status": "skipped_existing"})
                return

            try:
                await download_file(client.http, row["speech_url"], row["output_path"], client.config, min_bytes)
                results.append({**row, "status": "downloaded"})
            except Exception as exc:
                results.append({**row, "status": "download_failed", "error": str(exc)})

    await asyncio.gather(*(one(row) for row in rows))
    return sorted(results, key=lambda x: int(x["idx"]))


def build_audio_rows(audios: list[dict[str, Any]], entries: list[Entry], output_dir: str | Path, min_bytes: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(audios) < len(entries):
        raise RuntimeError(f"tts response missing audio: {len(audios)}/{len(entries)}")

    rows, skipped = [], []

    for e, audio in zip(entries, audios):
        out = audio_path(output_dir, e.idx)
        if audio_done(out, min_bytes):
            continue

        url = speech_url(audio)
        if not url:
            if no_audio_text(e.text):
                skipped.append(make_item(e, output_dir, "skipped_no_audio", "missing speech_url"))
                print(f"[no-audio] idx={e.idx} text={json.dumps(e.text, ensure_ascii=False)}", flush=True)
                continue
            raise MissingUrlError(e, f"missing speech_url idx={e.idx}")

        rows.append({**base_item(e, output_dir), "speech_url": url})

    return rows, skipped


class TTSEngine:
    def __init__(self, client: TTSClient, output_dir: str | Path, tts_limit: int, download_limit: int, min_bytes: int):
        self.client = client
        self.output_dir = Path(output_dir)
        self.tts_sem = asyncio.Semaphore(max(1, int(tts_limit)))
        self.download_limit = max(1, int(download_limit))
        self.min_bytes = min_bytes

    async def run_group(self, entries: list[Entry], name: str) -> list[dict[str, Any]]:
        entries = [e for e in entries if not audio_done(audio_path(self.output_dir, e.idx), self.min_bytes)]
        if not entries:
            return []

        try:
            async with self.tts_sem:
                print(f"[create] {name} n={len(entries)} idx={entries[0].idx}..{entries[-1].idx}", flush=True)
                created = await self.client.create(entries)
                done = await self.client.wait_done(created)
                audios = done.get("audio_subtitles") or []

                if not isinstance(audios, list) or not audios:
                    raise RuntimeError("missing audio_subtitles")

                rows, skipped = build_audio_rows(audios, entries, self.output_dir, self.min_bytes)
                downloaded = await download_many(self.client, rows, self.download_limit, self.min_bytes)
                return downloaded + skipped

        except BadTextError as exc:
            if len(entries) > 1:
                return await self.split(entries, name, "bad-text")
            e = entries[0]
            print(f"[bad-text] idx={e.idx} text={json.dumps(e.text, ensure_ascii=False)}", flush=True)
            return [make_item(e, self.output_dir, "bad_text_skipped", str(exc), exc.data)]

        except MissingUrlError as exc:
            if no_audio_text(exc.entry.text):
                e = exc.entry
                print(f"[no-audio] idx={e.idx} text={json.dumps(e.text, ensure_ascii=False)}", flush=True)
                return [make_item(e, self.output_dir, "skipped_no_audio", str(exc))]

            if len(entries) > 1:
                return await self.split(entries, name, f"missing-url bad_idx={exc.entry.idx}")

            e = exc.entry
            print(f"[missing-url] idx={e.idx} text={json.dumps(e.text, ensure_ascii=False)}", flush=True)
            return [make_item(e, self.output_dir, "tts_failed", str(exc))]

        except TerminalTTSError as exc:
            print(f"[tts-fail] {name}: {exc}", flush=True)
            return [make_item(e, self.output_dir, "tts_failed", str(exc), exc.data) for e in entries]

        except Exception as exc:
            print(f"[error] {name}: {exc}", flush=True)
            return [make_item(e, self.output_dir, "tts_failed", str(exc)) for e in entries]

    async def split(self, entries: list[Entry], name: str, reason: str) -> list[dict[str, Any]]:
        mid = len(entries) // 2
        left, right = entries[:mid], entries[mid:]
        print(f"[split] {reason} {name} {len(entries)} -> {len(left)}/{len(right)}", flush=True)
        a, b = await asyncio.gather(self.run_group(left, f"{name}-L"), self.run_group(right, f"{name}-R"))
        return a + b


async def generate_tts_from_srt_async(
    srt_path: str | Path,
    output_dir: str | Path,
    config: TTSConfig,
    batch_size: int = 80,
    max_tts_workers: int = 8,
    download_workers_per_job: int = 8,
    min_audio_bytes: int = 1024,
) -> list[dict[str, Any]]:
    output_dir = ensure_dir(output_dir)
    entries = read_entries(srt_path)
    skip = load_skip(output_dir)
    batches = split_batches(entries, output_dir, batch_size, min_audio_bytes, skip)
    results = [make_item(e, output_dir, "skipped_existing") for e in entries if audio_done(audio_path(output_dir, e.idx), min_audio_bytes)]
    results += [x for x in read_json_list(Path(output_dir) / "bad_tts_items.json") if int(x.get("idx", -1)) in skip]

    print(f"entries={len(entries)} existing={len(results)} batches={len(batches)} tts_workers={max_tts_workers} download_workers={download_workers_per_job}", flush=True)

    if not batches:
        final = merge_items(results)
        save_manifest(output_dir, final)
        return final

    async with TTSClient(config) as client:
        engine = TTSEngine(client, output_dir, max_tts_workers, download_workers_per_job, min_audio_bytes)
        tasks = [engine.run_group(batch, f"batch-{i}") for i, batch in enumerate(batches, 1)]
        chunks = await asyncio.gather(*tasks)

    for chunk in chunks:
        results.extend(chunk)

    final = merge_items(results)
    save_manifest(output_dir, final)
    save_bad(output_dir, final)

    skip = load_skip(output_dir)
    missing = [base_item(e, output_dir) for e in entries if e.idx not in skip and not audio_done(audio_path(output_dir, e.idx), min_audio_bytes)]
    if missing:
        write_json(Path(output_dir) / "missing_audio.json", missing)

    stats: dict[str, int] = {}
    for row in final:
        stats[row["status"]] = stats.get(row["status"], 0) + 1

    print(
        f"done={len(final)}/{len(entries)} "
        f"downloaded={stats.get('downloaded', 0)} "
        f"existing={stats.get('skipped_existing', 0)} "
        f"no_audio={stats.get('skipped_no_audio', 0)} "
        f"bad_text={stats.get('bad_text_skipped', 0)} "
        f"failed={stats.get('tts_failed', 0)} "
        f"download_failed={stats.get('download_failed', 0)}",
        flush=True,
    )

    return final


def generate_tts_from_srt(
    srt_path: str | Path,
    output_dir: str | Path,
    config: TTSConfig,
    batch_size: int = 80,
    max_tts_workers: int = 8,
    download_workers_per_job: int = 8,
    min_audio_bytes: int = 1024,
) -> list[dict[str, Any]]:
    return asyncio.run(generate_tts_from_srt_async(
        srt_path=srt_path,
        output_dir=output_dir,
        config=config,
        batch_size=batch_size,
        max_tts_workers=max_tts_workers,
        download_workers_per_job=download_workers_per_job,
        min_audio_bytes=min_audio_bytes,
    ))

