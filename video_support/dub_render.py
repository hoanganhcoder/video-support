from pathlib import Path
import subprocess
import shlex
import json
import math
import os
import time
import wave
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import srt


class DubRenderConfig:
    def __init__(
        self,
        render_dir="/content/render",
        original_volume=0.045,
        dub_volume=2.35,
        final_mix_volume=1.45,
        dub_target_peak=30500,
        max_dub_gain=4.0,
        limiter_limit=0.98,
        min_gap_ms=0,
        max_timeline_scale=1.30,
        scale_percentile=0.90,
        dub_sample_rate=24000,
        block_seconds=600,
        duration_workers=8,
        wav_convert_workers=4,
        ffmpeg_threads=4,
        faststart=True,
        normalize_dub=True,
        keep_cache=True,
        keep_blocks=False,
        silence_threshold_db=-70.0,
    ):
        self.render_dir = Path(render_dir)
        self.render_dir.mkdir(parents=True, exist_ok=True)

        self.original_volume = float(original_volume)
        self.dub_volume = float(dub_volume)
        self.final_mix_volume = float(final_mix_volume)
        self.dub_target_peak = int(dub_target_peak)
        self.max_dub_gain = float(max_dub_gain)
        self.limiter_limit = float(limiter_limit)

        self.min_gap_ms = int(min_gap_ms)
        self.max_timeline_scale = float(max_timeline_scale)
        self.scale_percentile = float(scale_percentile)

        self.dub_sample_rate = int(dub_sample_rate)
        self.block_seconds = int(block_seconds)

        self.duration_workers = int(duration_workers)
        self.wav_convert_workers = int(wav_convert_workers)
        self.ffmpeg_threads = str(ffmpeg_threads)

        self.faststart = bool(faststart)
        self.normalize_dub = bool(normalize_dub)
        self.keep_cache = bool(keep_cache)
        self.keep_blocks = bool(keep_blocks)
        self.silence_threshold_db = float(silence_threshold_db)


def _now():
    return time.perf_counter()


def _elapsed(label, t0):
    print(f"{label}: {time.perf_counter() - t0:.2f}s", flush=True)


def _run(cmd, check=True):
    cmd = list(map(str, cmd))
    print("Running:", " ".join(shlex.quote(x) for x in cmd), flush=True)

    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print("RETURN CODE:", r.returncode, flush=True)

    if r.stdout:
        print(r.stdout[-800:], flush=True)

    if r.stderr:
        print(r.stderr[-2500:], flush=True)

    if check and r.returncode != 0:
        raise RuntimeError("Command failed")

    return r


def _require_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_file():
        raise ValueError(path)

    return path


def _safe_unlink(path):
    try:
        Path(path).unlink()
    except Exception:
        pass


def _clean_dir(path):
    path = Path(path)

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def _load_json(path, default):
    path = Path(path)

    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _file_signature(path):
    st = Path(path).stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _ffprobe_duration_seconds(path):
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(_require_file(path)),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return float(r.stdout.strip())


def _ffprobe_duration_ms(path):
    return int(math.ceil(_ffprobe_duration_seconds(path) * 1000))


def _video_has_audio(path):
    path = Path(path)

    if not path.exists():
        return False

    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return bool(r.stdout.strip())


def _audio_max_volume_db(path):
    r = subprocess.run(
        [
            "ffmpeg", "-hide_banner",
            "-nostdin",
            "-i", str(_require_file(path)),
            "-af", "volumedetect",
            "-f", "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    text = r.stderr or ""

    for line in text.splitlines():
        if "max_volume:" in line:
            try:
                return float(line.split("max_volume:")[1].split("dB")[0].strip())
            except Exception:
                pass

    return None


def _assert_audio_not_silent(path, label, config):
    max_db = _audio_max_volume_db(path)
    print(f"{label} max_volume:", max_db, "dB", flush=True)

    if max_db is None:
        raise RuntimeError(f"{label} không đọc được volume: {path}")

    if max_db <= config.silence_threshold_db:
        raise RuntimeError(f"{label} gần như silent: {path}")


def _atempo_chain(speed):
    speed = float(speed)
    parts = []

    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5

    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0

    parts.append(f"atempo={speed:.8f}")
    return ",".join(parts)


def _read_srt_entries(path):
    text = _require_file(path).read_text(encoding="utf-8-sig", errors="ignore")
    out = []

    for sub in srt.parse(text):
        out.append({
            "index": int(sub.index),
            "start_ms": max(0, int(sub.start.total_seconds() * 1000)),
            "end_ms": max(0, int(sub.end.total_seconds() * 1000)),
            "content": str(sub.content or "").replace("\n", " ").strip(),
        })

    out.sort(key=lambda x: (x["start_ms"], x["index"]))
    return out


def _read_wav_info(path):
    with wave.open(str(_require_file(path)), "rb") as wf:
        return {
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "sample_rate": wf.getframerate(),
            "frames": wf.getnframes(),
        }


def _validate_pcm16_mono_wav(path, sample_rate):
    info = _read_wav_info(path)

    if info["channels"] != 1:
        raise RuntimeError(f"WAV không phải mono: {path}")

    if info["sample_width"] != 2:
        raise RuntimeError(f"WAV không phải pcm_s16le: {path}")

    if info["sample_rate"] != sample_rate:
        raise RuntimeError(f"WAV sai sample_rate {info['sample_rate']} != {sample_rate}: {path}")

    if info["frames"] <= 0:
        raise RuntimeError(f"WAV rỗng: {path}")

    return info


def _read_pcm16_wav(path):
    with wave.open(str(_require_file(path)), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()

        if channels != 1 or sample_width != 2:
            raise ValueError(f"Invalid WAV format: {path}")

        data = wf.readframes(frames)

    return sample_rate, np.frombuffer(data, dtype=np.int16)


def _write_int16_wav_stream(output_path, raw_block_files, gain, sample_rate):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))

        for raw_path in raw_block_files:
            arr = np.fromfile(str(raw_path), dtype=np.int32)

            if arr.size == 0:
                continue

            if gain != 1.0:
                arr = np.rint(arr.astype(np.float64) * gain)

            pcm = np.clip(arr, -32768, 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())

    return _require_file(output_path)


def collect_tts_items(vi_srt, tts_dir, config):
    t0 = _now()

    vi_srt = Path(vi_srt)
    tts_dir = Path(tts_dir)
    cache_file = config.render_dir / "tts_duration_cache.json"

    entries = _read_srt_entries(vi_srt)
    cache = _load_json(cache_file, {})

    items = []
    missing = []
    need_probe = []

    for e in entries:
        idx = int(e["index"])
        mp3 = tts_dir / f"{idx:05d}.mp3"

        if not mp3.exists() or mp3.stat().st_size < 1024:
            missing.append(idx)
            continue

        item = {
            "index": idx,
            "start_ms": int(e["start_ms"]),
            "path": mp3,
        }

        key = str(mp3)
        sig = _file_signature(mp3)
        rec = cache.get(key)

        if rec and rec.get("sig") == sig and rec.get("duration_ms"):
            item["duration_ms"] = int(rec["duration_ms"])
        else:
            need_probe.append((item, key, sig))

        items.append(item)

    print("TTS clips found:", len(items), flush=True)
    print("Need ffprobe:", len(need_probe), flush=True)
    print("Missing:", len(missing), flush=True)

    if missing:
        (config.render_dir / "missing_tts_for_dub.json").write_text(
            json.dumps(missing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if need_probe:
        key_to_item = {key: item for item, key, sig in need_probe}
        failed_probe = []
        changed = False

        def probe_job(item, key, sig):
            duration_ms = _ffprobe_duration_ms(item["path"])
            return key, sig, duration_ms

        with ThreadPoolExecutor(max_workers=config.duration_workers) as ex:
            futures = [
                ex.submit(probe_job, item, key, sig)
                for item, key, sig in need_probe
            ]

            for n, fut in enumerate(as_completed(futures), start=1):
                try:
                    key, sig, duration_ms = fut.result()
                    key_to_item[key]["duration_ms"] = int(duration_ms)
                    cache[key] = {
                        "sig": sig,
                        "duration_ms": int(duration_ms),
                    }
                    changed = True
                except Exception as e:
                    failed_probe.append(str(e))

                if n == 1 or n % 50 == 0 or n == len(futures):
                    print(f"ffprobe {n}/{len(futures)}", flush=True)

        if changed:
            _save_json_atomic(cache_file, cache)

        if failed_probe:
            (config.render_dir / "duration_probe_failed.json").write_text(
                json.dumps(failed_probe, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    items = [x for x in items if "duration_ms" in x]
    items.sort(key=lambda x: (x["start_ms"], x["index"]))

    print("TTS clips usable:", len(items), flush=True)
    _elapsed("collect_tts_items", t0)

    return items


def compute_timeline_scale(items, config):
    ratios = []
    items = sorted(items, key=lambda x: (x["start_ms"], x["index"]))

    for a, b in zip(items, items[1:]):
        gap = int(b["start_ms"]) - int(a["start_ms"])
        need = int(a["duration_ms"]) + int(config.min_gap_ms)

        if gap > 0:
            ratios.append(need / gap)

    if not ratios:
        scale = 1.0
    else:
        ratios.sort()
        pos = int(len(ratios) * float(config.scale_percentile))
        pos = min(max(pos, 0), len(ratios) - 1)
        scale = ratios[pos]

    scale = min(scale, float(config.max_timeline_scale))
    scale = max(1.0, scale)

    over_count = sum(1 for r in ratios if r > scale)

    report = {
        "timeline_scale": scale,
        "tts_speedup": scale,
        "scale_percentile": config.scale_percentile,
        "min_gap_ms": config.min_gap_ms,
        "max_timeline_scale": config.max_timeline_scale,
        "allowed_overlap_pairs": over_count,
        "total_pairs": len(ratios),
        "design": "speedup_tts_only_no_slow_timeline",
    }

    (config.render_dir / "timeline_scale_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("timeline_scale:", scale, flush=True)
    print("tts_speedup:", scale, flush=True)
    print("allowed overlap pairs:", over_count, "/", len(ratios), flush=True)

    return scale


def extract_original_audio(rendered_video, config, source_video=None):
    t0 = _now()

    rendered_video = Path(rendered_video)
    source_video = Path(source_video) if source_video else None
    out_path = config.render_dir / "original_audio_low.wav"

    if _video_has_audio(rendered_video):
        audio_source = rendered_video
    elif source_video and _video_has_audio(source_video):
        audio_source = source_video
    else:
        audio_source = None

    if audio_source:
        _run([
            "ffmpeg", "-y",
            "-hide_banner",
            "-nostdin",
            "-threads", config.ffmpeg_threads,
            "-i", str(_require_file(audio_source)),
            "-vn",
            "-af", f"volume={config.original_volume}",
            "-ac", "2",
            "-ar", "44100",
            "-c:a", "pcm_s16le",
            str(out_path),
        ])
    else:
        total_sec = _ffprobe_duration_seconds(rendered_video)

        _run([
            "ffmpeg", "-y",
            "-hide_banner",
            "-nostdin",
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", f"{total_sec:.3f}",
            "-c:a", "pcm_s16le",
            str(out_path),
        ])

    _elapsed("extract_original_audio", t0)
    return _require_file(out_path)


def cache_tts_speed_wavs(items, scale, config):
    t0 = _now()

    scale_key = f"{float(scale):.6f}".replace(".", "_")
    cache_dir = config.render_dir / "tts_speed_wav_cache" / f"scale_{scale_key}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = config.render_dir / f"tts_speed_wav_cache_scale_{scale_key}.json"
    cache = _load_json(cache_file, {})

    jobs = []
    usable = []
    failed = []

    atempo = _atempo_chain(scale)

    for item in items:
        mp3 = Path(item["path"])
        idx = int(item["index"])
        sig = _file_signature(mp3)
        wav_path = cache_dir / f"{idx:05d}.wav"

        key = str(mp3)
        rec = cache.get(key)

        ok = (
            rec
            and rec.get("sig") == sig
            and abs(float(rec.get("scale", 0)) - float(scale)) < 1e-6
            and rec.get("wav")
            and Path(rec["wav"]).exists()
            and Path(rec["wav"]).stat().st_size > 44
        )

        new_item = dict(item)

        if ok:
            try:
                info = _validate_pcm16_mono_wav(rec["wav"], config.dub_sample_rate)
                new_item["wav_path"] = Path(rec["wav"])
                new_item["speed_frames"] = int(info["frames"])
                usable.append(new_item)
            except Exception:
                jobs.append((new_item, key, sig, wav_path))
        else:
            jobs.append((new_item, key, sig, wav_path))

    print("Speed WAV cache ready:", len(usable), flush=True)
    print("Need convert/speed mp3->wav:", len(jobs), flush=True)

    def convert_job(new_item, key, sig, wav_path):
        tmp = wav_path.with_suffix(".tmp.wav")

        _safe_unlink(tmp)

        r = subprocess.run(
            [
                "ffmpeg", "-y",
                "-v", "error",
                "-nostdin",
                "-threads", "1",
                "-i", str(_require_file(new_item["path"])),
                "-af", atempo,
                "-ac", "1",
                "-ar", str(config.dub_sample_rate),
                "-c:a", "pcm_s16le",
                str(tmp),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if r.returncode != 0:
            raise RuntimeError(r.stderr[-1000:])

        tmp.replace(wav_path)

        info = _validate_pcm16_mono_wav(wav_path, config.dub_sample_rate)

        new_item["wav_path"] = wav_path
        new_item["speed_frames"] = int(info["frames"])

        return new_item, key, sig, wav_path

    if jobs:
        with ThreadPoolExecutor(max_workers=config.wav_convert_workers) as ex:
            futures = [
                ex.submit(convert_job, new_item, key, sig, wav_path)
                for new_item, key, sig, wav_path in jobs
            ]

            for n, fut in enumerate(as_completed(futures), start=1):
                try:
                    new_item, key, sig, wav_path = fut.result()
                    usable.append(new_item)
                    cache[key] = {
                        "sig": sig,
                        "scale": float(scale),
                        "wav": str(wav_path),
                    }
                except Exception as e:
                    failed.append(str(e))

                if n == 1 or n % 50 == 0 or n == len(futures):
                    print(f"convert speed wav {n}/{len(futures)}", flush=True)

        _save_json_atomic(cache_file, cache)

    if failed:
        (config.render_dir / "tts_speed_wav_failed.json").write_text(
            json.dumps(failed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    usable.sort(key=lambda x: (x["start_ms"], x["index"]))

    print("Speed WAV usable:", len(usable), flush=True)
    print("Speed WAV failed:", len(failed), flush=True)

    if not usable:
        raise RuntimeError("Không có speed WAV usable.")

    _elapsed("cache_tts_speed_wavs", t0)

    return usable


def _assign_items_to_blocks(items, total_frames, block_frames, sample_rate):
    block_count = max(1, int(math.ceil(total_frames / block_frames)))
    blocks = [[] for _ in range(block_count)]

    skipped = 0

    for item in items:
        start_frame = int(round(int(item["start_ms"]) * sample_rate / 1000.0))
        frames = int(item.get("speed_frames") or 0)
        end_frame = start_frame + frames

        if frames <= 0 or end_frame <= 0 or start_frame >= total_frames:
            skipped += 1
            continue

        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

        first_block = start_frame // block_frames
        last_block = max(first_block, (end_frame - 1) // block_frames)

        for b in range(first_block, last_block + 1):
            if 0 <= b < block_count:
                blocks[b].append(item)

    return blocks, skipped


def build_dub_wav_timeline_original(items, rendered_video, config):
    t0 = _now()

    rendered_video = Path(rendered_video)

    block_dir = config.render_dir / "dub_blocks_raw"
    final_dub = config.render_dir / "dub_timeline.wav"

    _clean_dir(block_dir)

    sample_rate = int(config.dub_sample_rate)
    total_sec = _ffprobe_duration_seconds(rendered_video)
    total_frames = int(math.ceil(total_sec * sample_rate))
    block_frames = int(config.block_seconds * sample_rate)
    block_count = max(1, int(math.ceil(total_frames / block_frames)))

    print("Dub timeline seconds:", total_sec, flush=True)
    print("Dub timeline frames:", total_frames, flush=True)
    print("Block seconds:", config.block_seconds, flush=True)
    print("Block count:", block_count, flush=True)

    blocks, skipped = _assign_items_to_blocks(
        items=items,
        total_frames=total_frames,
        block_frames=block_frames,
        sample_rate=sample_rate,
    )

    raw_block_files = []
    global_peak = 0
    failed = []
    mixed_count = 0
    non_empty_blocks = 0

    for block_index, block_items in enumerate(blocks):
        block_start_frame = block_index * block_frames
        frames_this_block = min(block_frames, total_frames - block_start_frame)

        if frames_this_block <= 0:
            continue

        raw_path = block_dir / f"block_{block_index:05d}.s32le"
        mix = np.zeros(frames_this_block, dtype=np.int32)

        print(
            f"mix block {block_index + 1}/{block_count} items={len(block_items)}",
            flush=True,
        )

        block_mixed = 0

        for item in block_items:
            try:
                rate, samples = _read_pcm16_wav(item["wav_path"])

                if rate != sample_rate:
                    raise ValueError(f"Bad wav sample rate: {rate} != {sample_rate}")

                target_frame = int(round(int(item["start_ms"]) * sample_rate / 1000.0))
                local_start = target_frame - block_start_frame
                src_start = 0

                if local_start < 0:
                    src_start = -local_start
                    local_start = 0

                if src_start >= samples.size:
                    continue

                local_end = min(frames_this_block, local_start + samples.size - src_start)

                if local_end <= local_start:
                    continue

                src_end = src_start + (local_end - local_start)

                mix[local_start:local_end] += samples[src_start:src_end].astype(np.int32)
                mixed_count += 1
                block_mixed += 1

            except Exception as e:
                failed.append({
                    "index": int(item.get("index", -1)),
                    "error": str(e),
                })

        if block_mixed > 0:
            non_empty_blocks += 1

        peak = int(np.max(np.abs(mix))) if mix.size else 0
        global_peak = max(global_peak, peak)

        mix.tofile(str(raw_path))
        raw_block_files.append(raw_path)

    if failed:
        (config.render_dir / "dub_mix_failed.json").write_text(
            json.dumps(failed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if mixed_count <= 0:
        raise RuntimeError("Không mix được TTS nào vào dub timeline.")

    if global_peak <= 0:
        raise RuntimeError("Dub timeline silent, peak = 0.")

    gain = 1.0

    if config.normalize_dub:
        gain = config.dub_target_peak / global_peak
        gain = min(gain, config.max_dub_gain)

    print("Dub raw peak:", global_peak, flush=True)
    print("Dub normalize gain:", gain, flush=True)
    print("Mixed count:", mixed_count, flush=True)
    print("Non-empty blocks:", non_empty_blocks, flush=True)
    print("Skipped items:", skipped, flush=True)

    _write_int16_wav_stream(
        output_path=final_dub,
        raw_block_files=raw_block_files,
        gain=gain,
        sample_rate=sample_rate,
    )

    _assert_audio_not_silent(final_dub, "DUB_WAV", config)

    report = {
        "method": "tts_speedup_then_block_pcm_mix_on_original_timeline",
        "timeline_seconds": total_sec,
        "timeline_frames": total_frames,
        "sample_rate": sample_rate,
        "block_seconds": config.block_seconds,
        "block_count": block_count,
        "items_count": len(items),
        "mixed_count_block_occurrences": mixed_count,
        "non_empty_blocks": non_empty_blocks,
        "skipped_items": skipped,
        "global_peak_before_normalize": global_peak,
        "normalize_enabled": config.normalize_dub,
        "normalize_gain": gain,
        "failed_count": len(failed),
    }

    (config.render_dir / "dub_mix_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not config.keep_blocks:
        shutil.rmtree(block_dir, ignore_errors=True)

    _elapsed("build_dub_wav", t0)
    return _require_file(final_dub)


def mux_final_audio_video(rendered_video, original_audio, dub_wav, output_video, config):
    t0 = _now()

    _assert_audio_not_silent(dub_wav, "DUB_BEFORE_MUX", config)

    filter_complex = (
        "[1:a]"
        "aresample=44100,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo[orig];"

        "[2:a]"
        f"volume={config.dub_volume},"
        "aresample=44100,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo[dub];"

        "[orig][dub]"
        "amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
        f"volume={config.final_mix_volume},"
        f"alimiter=limit={config.limiter_limit}[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-nostdin",
        "-threads", config.ffmpeg_threads,
        "-i", str(_require_file(rendered_video)),
        "-i", str(_require_file(original_audio)),
        "-i", str(_require_file(dub_wav)),
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
    ]

    if config.faststart:
        cmd += ["-movflags", "+faststart"]

    cmd += [str(output_video)]

    _run(cmd)

    final_video = _require_file(output_video)
    _assert_audio_not_silent(final_video, "FINAL_VIDEO_AUDIO", config)

    _elapsed("mux_final_audio_video", t0)
    return final_video


def render_dubbed_video(
    rendered_video,
    vi_srt,
    tts_dir,
    output_video,
    source_video=None,
    config=None,
):
    config = config or DubRenderConfig()
    total_t0 = _now()

    rendered_video = Path(rendered_video)
    vi_srt = Path(vi_srt)
    tts_dir = Path(tts_dir)
    output_video = Path(output_video)
    source_video = Path(source_video) if source_video else None

    print("RENDERED_VIDEO:", rendered_video, flush=True)
    print("SOURCE_VIDEO:", source_video, flush=True)
    print("VI_SRT:", vi_srt, flush=True)
    print("TTS_DIR:", tts_dir, flush=True)
    print("OUTPUT_VIDEO:", output_video, flush=True)

    print("MAX_TIMELINE_SCALE:", config.max_timeline_scale, flush=True)
    print("SCALE_PERCENTILE:", config.scale_percentile, flush=True)
    print("BLOCK_SECONDS:", config.block_seconds, flush=True)
    print("DUB_SAMPLE_RATE:", config.dub_sample_rate, flush=True)

    _require_file(rendered_video)
    _require_file(vi_srt)

    items = collect_tts_items(vi_srt, tts_dir, config)

    if not items:
        raise RuntimeError("No TTS clips found. Check TTS_DIR and VI_SRT indices.")

    scale = compute_timeline_scale(items, config)

    original_audio = extract_original_audio(
        rendered_video=rendered_video,
        source_video=source_video,
        config=config,
    )

    speed_items = cache_tts_speed_wavs(items, scale, config)

    dub_wav = build_dub_wav_timeline_original(
        items=speed_items,
        rendered_video=rendered_video,
        config=config,
    )

    final_video = mux_final_audio_video(
        rendered_video=rendered_video,
        original_audio=original_audio,
        dub_wav=dub_wav,
        output_video=output_video,
        config=config,
    )

    print("DONE:", final_video, flush=True)
    print("exists:", final_video.exists(), flush=True)
    print("size_mb:", final_video.stat().st_size / 1024 / 1024 if final_video.exists() else None, flush=True)

    _elapsed("TOTAL", total_t0)

    return {
        "output_video": str(final_video),
        "scale": scale,
        "original_audio": str(original_audio),
        "dub_wav": str(dub_wav),
    }