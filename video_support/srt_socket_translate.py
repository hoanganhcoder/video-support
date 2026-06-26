import re
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
import datetime as dt

import srt
import requests


@dataclass
class SrtEntry:
    index: int
    start: dt.timedelta
    end: dt.timedelta
    content: str


def clean_one_line(text: str) -> str:
    text = str(text or "")
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def read_srt(path: str | Path) -> list[SrtEntry]:
    raw = Path(path).read_text(encoding="utf-8-sig", errors="ignore")
    entries = []

    for sub in srt.parse(raw):
        entries.append(
            SrtEntry(
                index=int(sub.index),
                start=sub.start,
                end=sub.end,
                content=clean_one_line(sub.content),
            )
        )

    return entries


def write_srt(entries: list[SrtEntry], path: str | Path) -> None:
    subtitles = [
        srt.Subtitle(
            index=i + 1,
            start=e.start,
            end=e.end,
            content=e.content.strip(),
        )
        for i, e in enumerate(entries)
    ]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(srt.compose(subtitles), encoding="utf-8")


def make_batches(entries: list[SrtEntry], batch_size: int = 100):
    for i in range(0, len(entries), batch_size):
        yield i, entries[i:i + batch_size]


def build_batch_input(batch: list[SrtEntry]) -> str:
    lines = []

    for i, entry in enumerate(batch, start=1):
        lines.append(f"{i}. {entry.content}")

    return "\n".join(lines)


def send_job(server: str, text: str) -> str:
    url = f"{server.rstrip('/')}/ask"

    res = requests.post(
        url,
        json={"text": text},
        timeout=60,
    )

    res.raise_for_status()
    data = res.json()

    return data["job_id"]


def poll_job(
    server: str,
    job_id: str,
    interval: float = 2.0,
    timeout: int = 600,
) -> dict:
    url = f"{server.rstrip('/')}/jobs/{job_id}"
    start = time.time()

    while True:
        if time.time() - start > timeout:
            raise TimeoutError(f"Job timeout: {job_id}")

        res = requests.get(url, timeout=30)
        res.raise_for_status()

        data = res.json()
        status = data.get("status")

        if status == "done":
            return data

        if status == "error":
            raise RuntimeError(data.get("error") or f"Job lỗi: {job_id}")

        print(f"Polling {job_id}: {status}")
        time.sleep(interval)


def parse_numbered_result(text: str) -> dict[int, str]:
    """
    Parse output dạng:
    1. text
    2. text
    3. text

    Có hỗ trợ trường hợp text nhiều dòng.
    """
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    pattern = re.compile(
        r"(?m)^\s*(\d+)\s*[\.\)]\s*(.*?)(?=^\s*\d+\s*[\.\)]\s*|\Z)",
        re.S,
    )

    result = {}

    for match in pattern.finditer(text):
        num = int(match.group(1))
        value = match.group(2).strip()
        value = re.sub(r"\n{3,}", "\n\n", value)
        result[num] = value

    return result


def validate_batch_result(
    batch: list[SrtEntry],
    parsed: dict[int, str],
) -> tuple[bool, list[str]]:
    errors = []
    expected = len(batch)

    for i in range(1, expected + 1):
        if i not in parsed:
            errors.append(f"Thiếu dòng {i}")

    extra = sorted(k for k in parsed.keys() if k < 1 or k > expected)

    for k in extra:
        errors.append(f"Thừa dòng {k}")

    return len(errors) == 0, errors


def translate_srt_via_socket(
    input_srt: str | Path,
    output_srt: str | Path,
    server: str = "http://127.0.0.1:8000",
    batch_size: int = 100,
    poll_interval: float = 2.0,
    job_timeout: int = 600,
    retry: int = 2,
):
    entries = read_srt(input_srt)

    if not entries:
        raise ValueError("SRT rỗng hoặc không đọc được")

    output_entries = [
        SrtEntry(
            index=e.index,
            start=e.start,
            end=e.end,
            content="",
        )
        for e in entries
    ]

    total_batches = (len(entries) + batch_size - 1) // batch_size

    print(f"Total lines: {len(entries)}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")

    for batch_no, (offset, batch) in enumerate(
        make_batches(entries, batch_size=batch_size),
        start=1,
    ):
        print(f"\nBatch {batch_no}/{total_batches}")
        question = build_batch_input(batch)

        last_error = None
        parsed = None

        for attempt in range(1, retry + 2):
            try:
                print(f"Send attempt {attempt}")

                job_id = send_job(server, question)
                print("Job:", job_id)

                job = poll_job(
                    server=server,
                    job_id=job_id,
                    interval=poll_interval,
                    timeout=job_timeout,
                )

                result_text = job.get("text") or ""
                parsed = parse_numbered_result(result_text)

                ok, errors = validate_batch_result(batch, parsed)

                if not ok:
                    raise ValueError("; ".join(errors))

                break

            except Exception as e:
                last_error = e
                print("Batch error:", repr(e))

                if attempt >= retry + 1:
                    raise RuntimeError(
                        f"Batch {batch_no} thất bại sau {attempt} lần: {last_error}"
                    )

                time.sleep(3)

        for local_no, entry in enumerate(batch, start=1):
            output_entries[offset + local_no - 1].content = parsed.get(local_no, "")

        write_srt(output_entries, output_srt)
        print("Saved partial:", output_srt)

    write_srt(output_entries, output_srt)
    print("\nDone:", output_srt)


