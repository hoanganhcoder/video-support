import os
import re
import json
import time
import hashlib
import datetime as dt
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import srt
from tqdm.auto import tqdm
from openai import OpenAI


@dataclass
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


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff]")
KOREAN_RE = re.compile(r"[\uac00-\ud7af]")
VIETNAMESE_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
    r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨ"
    r"ÒÓỌỎÕÔỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
)
SQUARE_RE = re.compile(r"[□�▯▢■◆◇]+")
ONLY_SYMBOL_RE = re.compile(r"^[\W_]+$", re.UNICODE)


DEFAULT_MODEL = "gpt-4.1-mini"


# =====================
# Text utils
# =====================

def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_one_line(text: str) -> str:
    return clean_text(str(text or "").replace("\n", " "))


def clean_json_response(text: str) -> str:
    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()

    first = text.find("{")
    last = text.rfind("}")

    if first != -1 and last != -1 and last > first:
        text = text[first:last + 1]

    return text


def source_hash(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:12]


def contains_asian_non_vi(text: str) -> bool:
    text = str(text or "")
    return bool(
        CJK_RE.search(text)
        or JAPANESE_RE.search(text)
        or KOREAN_RE.search(text)
    )


def has_vietnamese_char(text: str) -> bool:
    return bool(VIETNAMESE_RE.search(str(text or "")))


def is_ocr_junk(text: str) -> bool:
    text = clean_one_line(text)
    if not text:
        return True

    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True

    if SQUARE_RE.fullmatch(compact):
        return True

    square_count = sum(1 for ch in compact if SQUARE_RE.match(ch))
    if len(compact) >= 2 and square_count / max(1, len(compact)) >= 0.5:
        return True

    if ONLY_SYMBOL_RE.fullmatch(compact):
        return True

    if not contains_asian_non_vi(compact) and not has_vietnamese_char(compact):
        if re.fullmatch(r"[A-Za-z0-9|IlO0]+", compact) and len(compact) <= 3:
            return True

    if re.fullmatch(r"\d{1,4}", compact):
        return True

    return False


def normalize_model_text(text: str) -> str:
    text = clean_one_line(text)
    if text.lower() in {"null", "none", "undefined"}:
        return ""
    return text


# =====================
# SRT IO
# =====================

def read_srt(path: str | Path) -> list[SrtEntry]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
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


def write_srt(entries: list[SrtEntry], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    subtitles = [
        srt.Subtitle(
            index=int(e.index),
            start=e.start,
            end=e.end,
            content=str(e.content or ""),
        )
        for e in entries
    ]

    path.write_text(srt.compose(subtitles), encoding="utf-8")
    return path


# =====================
# Cache
# =====================

def load_cache(path: str | Path) -> dict[int, dict[str, str]]:
    path = Path(path)

    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out = {}

    for k, v in raw.items():
        try:
            idx = int(k)
        except Exception:
            continue

        if isinstance(v, dict):
            out[idx] = {
                "src_hash": str(v.get("src_hash", "")),
                "text_vi": str(v.get("text_vi", "")),
            }
        else:
            out[idx] = {
                "src_hash": "",
                "text_vi": str(v),
            }

    return out


def save_cache(path: str | Path, cache: dict[int, dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        str(k): {
            "src_hash": str(v.get("src_hash", "")),
            "text_vi": str(v.get("text_vi", "")),
        }
        for k, v in sorted(cache.items())
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_cached(cache: dict[int, dict[str, str]], entry: SrtEntry) -> str | None:
    item = cache.get(int(entry.index))
    if not item:
        return None

    if item.get("src_hash") and item.get("src_hash") != source_hash(entry.content):
        return None

    text_vi = clean_text(item.get("text_vi", ""))

    if is_ocr_junk(entry.content):
        return "" if text_vi == "" else None

    if contains_asian_non_vi(text_vi):
        return None

    return text_vi


def set_cached(cache: dict[int, dict[str, str]], entry: SrtEntry, text_vi: str) -> None:
    cache[int(entry.index)] = {
        "src_hash": source_hash(entry.content),
        "text_vi": str(text_vi or ""),
    }


# =====================
# API
# =====================

def make_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY", "")

    if not key:
        try:
            from google.colab import userdata
            key = userdata.get("OPENAI_API_KEY") or ""
        except Exception:
            pass

    if not key:
        raise ValueError("Thiếu API key. Truyền api_key=... hoặc set OPENAI_API_KEY.")

    kwargs = {"api_key": key}

    final_base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
    if final_base_url:
        kwargs["base_url"] = final_base_url

    return OpenAI(**kwargs)


def call_json(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    retries: int = 4,
) -> dict[str, Any]:
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            data = json.loads(clean_json_response(content))

            if not isinstance(data.get("items"), list):
                raise ValueError("JSON output sai schema: thiếu items list")

            return data

        except Exception as exc:
            last_error = exc
            sleep_s = min(20, 1.5 * attempt)
            print(f"api retry {attempt}/{retries}: {repr(exc)} | sleep {sleep_s}s")
            time.sleep(sleep_s)

    raise RuntimeError(f"API thất bại: {last_error}")


# =====================
# Prompt builders
# =====================

def build_translate_messages(
    items: list[dict[str, Any]],
    system_prompt: str,
    user_prompt: str = "",
) -> list[dict[str, str]]:
    if not system_prompt or not str(system_prompt).strip():
        raise ValueError("Thiếu system_prompt.")

    prompt = str(system_prompt).strip()

    if user_prompt:
        prompt += "\n\nYÊU CẦU BỔ SUNG TỪ NGƯỜI DÙNG:\n"
        prompt += str(user_prompt).strip()

    payload = {
        "task": "translate_subtitle_to_vietnamese",
        "items": items,
    }

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_repair_messages(
    items: list[dict[str, Any]],
    repair_prompt: str,
) -> list[dict[str, str]]:
    if not repair_prompt or not str(repair_prompt).strip():
        raise ValueError("Thiếu repair_prompt.")

    payload = {
        "task": "repair_remaining_cjk_in_vietnamese_subtitles",
        "items": items,
    }

    return [
        {"role": "system", "content": str(repair_prompt).strip()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


# =====================
# Batching / workers
# =====================

def split_batches(
    entries: list[SrtEntry],
    batch_size: int,
    max_chars: int,
) -> list[list[SrtEntry]]:
    batches = []
    current = []
    chars = 0

    for entry in entries:
        n = len(entry.content or "")

        if current and (len(current) >= batch_size or chars + n > max_chars):
            batches.append(current)
            current = []
            chars = 0

        current.append(entry)
        chars += n

    if current:
        batches.append(current)

    return batches


def split_dict_batches(
    items: list[dict[str, Any]],
    batch_size: int,
    max_chars: int,
) -> list[list[dict[str, Any]]]:
    batches = []
    current = []
    chars = 0

    for item in items:
        n = (
            len(item.get("source_text", ""))
            + len(item.get("bad_text_vi", ""))
            + len(item.get("text", ""))
        )

        if current and (len(current) >= batch_size or chars + n > max_chars):
            batches.append(current)
            current = []
            chars = 0

        current.append(item)
        chars += n

    if current:
        batches.append(current)

    return batches


def translate_worker(
    batch_id: int,
    items: list[dict[str, Any]],
    entries_by_idx: dict[int, SrtEntry],
    model: str,
    temperature: float,
    api_key: str | None,
    base_url: str | None,
    system_prompt: str,
    user_prompt: str,
) -> tuple[int, dict[int, str]]:
    client = make_client(api_key=api_key, base_url=base_url)

    data = call_json(
        client=client,
        model=model,
        messages=build_translate_messages(
            items=items,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ),
        temperature=temperature,
    )

    raw_map = {}

    for x in data.get("items", []):
        if "index" in x:
            raw_map[int(x["index"])] = normalize_model_text(x.get("text_vi", ""))

    result = {}

    for item in items:
        idx = int(item["index"])
        entry = entries_by_idx[idx]

        text_vi = raw_map.get(idx, "")

        if is_ocr_junk(entry.content):
            text_vi = ""

        result[idx] = text_vi

    return batch_id, result


def repair_worker(
    batch_id: int,
    items: list[dict[str, Any]],
    model: str,
    api_key: str | None,
    base_url: str | None,
    repair_prompt: str,
) -> tuple[int, dict[int, str]]:
    client = make_client(api_key=api_key, base_url=base_url)

    data = call_json(
        client=client,
        model=model,
        messages=build_repair_messages(
            items=items,
            repair_prompt=repair_prompt,
        ),
        temperature=0.0,
    )

    raw_map = {}

    for x in data.get("items", []):
        if "index" in x:
            raw_map[int(x["index"])] = normalize_model_text(x.get("text_vi", ""))

    result = {}

    for item in items:
        idx = int(item["index"])
        fixed = raw_map.get(idx, "")

        if contains_asian_non_vi(fixed):
            fixed = ""

        result[idx] = fixed

    return batch_id, result


# =====================
# Public API
# =====================

def translate_srt(
    input_srt: str | Path,
    output_srt: str | Path,
    system_prompt: str,
    repair_prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    user_prompt: str = "",
    cache_path: str | Path = "translation_cache_vi.json",
    resume: bool = True,
    repair_cjk: bool = True,
    batch_size: int = 12,
    max_chars: int = 2400,
    max_workers: int = 8,
    temperature: float = 0.1,
    save_every: int = 5,
) -> Path:
    _ = make_client(api_key=api_key, base_url=base_url)

    if not system_prompt or not str(system_prompt).strip():
        raise ValueError("Thiếu system_prompt.")

    if repair_cjk and (not repair_prompt or not str(repair_prompt).strip()):
        raise ValueError("Thiếu repair_prompt.")

    input_srt = Path(input_srt)
    output_srt = Path(output_srt)
    cache_path = Path(cache_path)

    entries = read_srt(input_srt)

    if not entries:
        raise ValueError("SRT rỗng hoặc không đọc được.")

    entries_by_idx = {e.index: e for e in entries}
    cache = load_cache(cache_path) if resume else {}
    translated: dict[int, str] = {}
    cache_lock = Lock()
    save_counter = 0

    for e in entries:
        if is_ocr_junk(e.content):
            translated[e.index] = ""
            set_cached(cache, e, "")
            continue

        cached = get_cached(cache, e)
        if cached is not None:
            translated[e.index] = cached

    raw_batches = split_batches(
        entries,
        batch_size=batch_size,
        max_chars=max_chars,
    )

    job_batches = []

    for batch in raw_batches:
        items = []

        for e in batch:
            if e.index in translated:
                continue

            if is_ocr_junk(e.content):
                translated[e.index] = ""
                set_cached(cache, e, "")
                continue

            items.append({
                "index": e.index,
                "start_ms": e.start_ms,
                "end_ms": e.end_ms,
                "duration_ms": e.duration_ms,
                "text": e.content,
            })

        if items:
            job_batches.append(items)

    print("Total entries:", len(entries))
    print("Raw batches:", len(raw_batches))
    print("Translate jobs:", len(job_batches))
    print("Cached/filled:", len(translated))
    print("Max workers:", max_workers)

    if job_batches:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    translate_worker,
                    batch_id,
                    items,
                    entries_by_idx,
                    model,
                    temperature,
                    api_key,
                    base_url,
                    system_prompt,
                    user_prompt,
                )
                for batch_id, items in enumerate(job_batches, start=1)
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
                batch_id, result_map = future.result()

                with cache_lock:
                    for idx, text_vi in result_map.items():
                        entry = entries_by_idx[idx]
                        translated[idx] = text_vi
                        set_cached(cache, entry, text_vi)

                    save_counter += 1
                    if save_counter % save_every == 0:
                        save_cache(cache_path, cache)

        save_cache(cache_path, cache)

    if repair_cjk:
        repair_items = []

        for e in entries:
            text_vi = translated.get(e.index, "")

            if is_ocr_junk(e.content):
                translated[e.index] = ""
                set_cached(cache, e, "")
                continue

            if contains_asian_non_vi(text_vi):
                repair_items.append({
                    "index": e.index,
                    "source_text": e.content,
                    "bad_text_vi": text_vi,
                })

        print("Need CJK repair:", len(repair_items))

        if repair_items:
            repair_batches = split_dict_batches(
                repair_items,
                batch_size=batch_size,
                max_chars=max_chars,
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        repair_worker,
                        batch_id,
                        items,
                        model,
                        api_key,
                        base_url,
                        repair_prompt,
                    )
                    for batch_id, items in enumerate(repair_batches, start=1)
                ]

                for future in tqdm(as_completed(futures), total=len(futures), desc="Repairing"):
                    batch_id, result_map = future.result()

                    with cache_lock:
                        for idx, fixed in result_map.items():
                            entry = entries_by_idx[idx]
                            translated[idx] = fixed
                            set_cached(cache, entry, fixed)

                        save_cache(cache_path, cache)

    for e in entries:
        text_vi = translated.get(e.index, "")

        if is_ocr_junk(e.content) or contains_asian_non_vi(text_vi):
            translated[e.index] = ""
            set_cached(cache, e, "")

    save_cache(cache_path, cache)

    output_entries = [
        SrtEntry(
            index=e.index,
            start=e.start,
            end=e.end,
            content=translated.get(e.index, ""),
        )
        for e in entries
    ]

    out = write_srt(output_entries, output_srt)
    print("SRT saved:", out)
    return out


