%%writefile dub_render.py
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
        block_seconds=900,
        wav_convert_workers=4,
        duration_workers=8,
        ffmpeg_threads=4,
        faststart=True,
        normalize_dub=True,
        keep_cache=True,
        keep_blocks=False,
    ):
        self.render_dir = Path(render_dir)
        self.render_dir.mkdir(parents=True, exist_ok=True)

        self.original_volume = original_volume
        self.dub_volume = dub_volume
        self.final_mix_volume = final_mix_volume
        self.dub_target_peak = dub_target_peak
        self.max_dub_gain = max_dub_gain
        self.limiter_limit = limiter_limit

        self.min_gap_ms = min_gap_ms
        self.max_timeline_scale = max_timeline_scale
        self.scale_percentile = scale_percentile

        self.dub_sample_rate = int(dub_sample_rate)
        self.block_seconds = int(block_seconds)

        self.wav_convert_workers = int(wav_convert_workers)
        self.duration_workers = int(duration_workers)
        self.ffmpeg_threads = str(ffmpeg_threads)

        self.faststart = bool(faststart)
        self.normalize_dub = bool(normalize_dub)
        self.keep_cache = bool(keep_cache)
        self.keep_blocks = bool(keep_blocks)


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
        print(r.stdout[-1000:], flush=True)

    if r.stderr:
        print(r.stderr[-3000:], flush=True)

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


def _read_pcm16_wav(path):
    with wave.open(str(_require_file(path)), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.getnframes()

        if channels != 1 or sampwidth != 2:
            raise ValueError(f"Invalid WAV format: {path}")

        data = wf.readframes(frames)

    return rate, np.frombuffer(data, dtype=np.int16)


def _write_pcm16_wav(path, samples, sample_rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(samples)
    samples = np.clip(samples, -32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(samples.tobytes())

    return path


def _concat_wavs_copy(wav_files, output_path, list_path, config):
    list_path = Path(list_path)
    output_path = Path(output_path)

    with list_path.open("w", encoding="utf-8") as f:
        for p in wav_files:
            f.write(f"file '{Path(p).resolve().as_posix()}'\n")

    _run([
        "ffmpeg", "-y",
        "-hide_banner",
        "-nostdin",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(output_path),
    ])

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
        "slow_audio_speed": 1.0 / scale,
        "final_speedback": scale,
        "scale_percentile": config.scale_percentile,
        "min_gap_ms": config.min_gap_ms,
        "max_timeline_scale": config.max_timeline_scale,
        "allowed_overlap_pairs": over_count,
        "total_pairs": len(ratios),
    }

    (config.render_dir / "timeline_scale_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("timeline_scale:", scale, flush=True)
    print("slow_audio_speed:", 1.0 / scale, flush=True)
    print("final_speedback:", scale, flush=True)
    print("allowed overlap pairs:", over_count, "/", len(ratios), flush=True)

    return scale


def make_slow_original_audio(rendered_video, scale, config, source_video=None):
    t0 = _now()

    rendered_video = Path(rendered_video)
    source_video = Path(source_video) if source_video else None
    out_path = config.render_dir / "slow_original_low.wav"

    if _video_has_audio(rendered_video):
        audio_source = rendered_video
    elif source_video and _video_has_audio(source_video):
        audio_source = source_video
    else:
        audio_source = None

    speed = 1.0 / float(scale)

    if audio_source:
        _run([
            "ffmpeg", "-y",
            "-hide_banner",
            "-nostdin",
            "-threads", config.ffmpeg_threads,
            "-i", str(_require_file(audio_source)),
            "-vn",
            "-af", f"{_atempo_chain(speed)},volume={config.original_volume}",
            "-ac", "2",
            "-ar", "44100",
            "-c:a", "pcm_s16le",
            str(out_path),
        ])
    else:
        total_sec = _ffprobe_duration_seconds(rendered_video) * float(scale)

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

    _elapsed("make_slow_original_audio", t0)
    return _require_file(out_path)


def cache_tts_wavs(items, config):
    t0 = _now()

    cache_dir = config.render_dir / "tts_wav_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = config.render_dir / "tts_wav_cache.json"
    cache = _load_json(cache_file, {})

    jobs = []
    usable = []
    failed = []

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
            and rec.get("wav")
            and Path(rec["wav"]).exists()
            and Path(rec["wav"]).stat().st_size > 44
        )

        new_item = dict(item)
        new_item["wav_path"] = wav_path

        if ok:
            new_item["wav_path"] = Path(rec["wav"])
            usable.append(new_item)
        else:
            jobs.append((new_item, key, sig, wav_path))

    print("WAV cache ready:", len(usable), flush=True)
    print("Need convert mp3->wav:", len(jobs), flush=True)

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

        new_item["wav_path"] = wav_path

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
                        "wav": str(wav_path),
                    }
                except Exception as e:
                    failed.append(str(e))

                if n == 1 or n % 50 == 0 or n == len(futures):
                    print(f"convert wav {n}/{len(futures)}", flush=True)

        _save_json_atomic(cache_file, cache)

    if failed:
        (config.render_dir / "tts_wav_convert_failed.json").write_text(
            json.dumps(failed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    usable.sort(key=lambda x: (x["start_ms"], x["index"]))

    print("WAV usable:", len(usable), flush=True)
    print("WAV failed:", len(failed), flush=True)
    _elapsed("cache_tts_wavs", t0)

    return usable


def _item_scaled_range_frames(item, scale, sample_rate):
    start_ms = int(round(int(item["start_ms"]) * float(scale)))
    duration_ms = int(item["duration_ms"])

    start_frame = int(round(start_ms * sample_rate / 1000.0))
    approx_frames = int(math.ceil(duration_ms * sample_rate / 1000.0))

    return start_frame, start_frame + approx_frames


def _assign_items_to_blocks(items, scale, total_frames, block_frames, sample_rate):
    block_count = max(1, int(math.ceil(total_frames / block_frames)))
    blocks = [[] for _ in range(block_count)]

    for item in items:
        start_frame, end_frame = _item_scaled_range_frames(item, scale, sample_rate)

        if end_frame <= 0 or start_frame >= total_frames:
            continue

        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

        first_block = start_frame // block_frames
        last_block = max(first_block, (end_frame - 1) // block_frames)

        for b in range(first_block, last_block + 1):
            if 0 <= b < block_count:
                blocks[b].append(item)

    return blocks


def build_dub_wav_block_mix(items, rendered_video, scale, config):
    t0 = _now()

    rendered_video = Path(rendered_video)
    block_dir = config.render_dir / "dub_blocks"
    final_dub = config.render_dir / "dub_slow_timeline.wav"
    raw_dub = config.render_dir / "dub_slow_timeline_raw.wav"
    concat_list = config.render_dir / "dub_blocks.txt"

    _clean_dir(block_dir)

    sample_rate = int(config.dub_sample_rate)
    total_sec = _ffprobe_duration_seconds(rendered_video) * float(scale)
    total_frames = int(math.ceil(total_sec * sample_rate))
    block_frames = int(config.block_seconds * sample_rate)
    block_count = max(1, int(math.ceil(total_frames / block_frames)))

    print("Dub timeline seconds:", total_sec, flush=True)
    print("Dub timeline frames:", total_frames, flush=True)
    print("Block seconds:", config.block_seconds, flush=True)
    print("Block frames:", block_frames, flush=True)
    print("Block count:", block_count, flush=True)

    blocks = _assign_items_to_blocks(
        items=items,
        scale=scale,
        total_frames=total_frames,
        block_frames=block_frames,
        sample_rate=sample_rate,
    )

    block_files = []
    global_peak = 0
    mixed_count = 0
    failed = []

    for block_index, block_items in enumerate(blocks):
        block_start_frame = block_index * block_frames
        frames_this_block = min(block_frames, total_frames - block_start_frame)

        if frames_this_block <= 0:
            continue

        block_path = block_dir / f"block_{block_index:05d}.wav"
        mix = np.zeros(frames_this_block, dtype=np.int32)

        if block_index == 0 or block_index % 5 == 0 or block_index == block_count - 1:
            print(
                f"mix block {block_index + 1}/{block_count} "
                f"items={len(block_items)}",
                flush=True,
            )

        seen_in_block = 0

        for item in block_items:
            try:
                rate, samples = _read_pcm16_wav(item["wav_path"])

                if rate != sample_rate:
                    raise ValueError(f"Bad wav sample rate: {rate} != {sample_rate}")

                scaled_start_ms = int(round(int(item["start_ms"]) * float(scale)))
                target_frame = int(round(scaled_start_ms * sample_rate / 1000.0))

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
                seen_in_block += 1

            except Exception as e:
                failed.append({
                    "index": int(item.get("index", -1)),
                    "error": str(e),
                })

        if mix.size:
            peak = int(np.max(np.abs(mix)))
            global_peak = max(global_peak, peak)

        mixed_count += seen_in_block

        _write_pcm16_wav(block_path, mix, sample_rate)
        block_files.append(block_path)

    if failed:
        (config.render_dir / "dub_block_mix_failed.json").write_text(
            json.dumps(failed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    _concat_wavs_copy(block_files, raw_dub, concat_list, config)

    gain = 1.0

    if config.normalize_dub and global_peak > 0:
        gain = config.dub_target_peak / global_peak
        gain = min(gain, config.max_dub_gain)

    print("Dub raw peak:", global_peak, flush=True)
    print("Dub normalize gain:", gain, flush=True)

    if gain != 1.0:
        _run([
            "ffmpeg", "-y",
            "-hide_banner",
            "-nostdin",
            "-threads", config.ffmpeg_threads,
            "-i", str(_require_file(raw_dub)),
            "-af", f"volume={gain:.8f}",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(final_dub),
        ])
    else:
        shutil.copyfile(raw_dub, final_dub)

    report = {
        "method": "wav_cache_block_pcm_mix",
        "timeline_seconds": total_sec,
        "timeline_frames": total_frames,
        "sample_rate": sample_rate,
        "block_seconds": config.block_seconds,
        "block_count": block_count,
        "mixed_count_block_occurrences": mixed_count,
        "items_count": len(items),
        "global_peak_before_normalize": global_peak,
        "normalize_enabled": config.normalize_dub,
        "normalize_gain": gain,
        "failed_count": len(failed),
    }

    (config.render_dir / "dub_block_mix_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not config.keep_blocks:
        shutil.rmtree(block_dir, ignore_errors=True)

    _elapsed("build_dub_wav_block_mix", t0)
    return _require_file(final_dub)


def mux_final_audio_video(rendered_video, slow_original_audio, dub_wav, output_video, scale, config):
    t0 = _now()

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
        f"{_atempo_chain(scale)},"
        f"volume={config.final_mix_volume},"
        f"alimiter=limit={config.limiter_limit}[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-nostdin",
        "-threads", config.ffmpeg_threads,
        "-i", str(_require_file(rendered_video)),
        "-i", str(_require_file(slow_original_audio)),
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

    _elapsed("mux_final_audio_video", t0)
    return _require_file(output_video)


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

    print("ORIGINAL_VOLUME:", config.original_volume, flush=True)
    print("DUB_VOLUME:", config.dub_volume, flush=True)
    print("FINAL_MIX_VOLUME:", config.final_mix_volume, flush=True)
    print("DUB_TARGET_PEAK:", config.dub_target_peak, flush=True)

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

    slow_original_audio = make_slow_original_audio(
        rendered_video=rendered_video,
        source_video=source_video,
        scale=scale,
        config=config,
    )

    wav_items = cache_tts_wavs(items, config)

    dub_wav = build_dub_wav_block_mix(
        items=wav_items,
        rendered_video=rendered_video,
        scale=scale,
        config=config,
    )

    final_video = mux_final_audio_video(
        rendered_video=rendered_video,
        slow_original_audio=slow_original_audio,
        dub_wav=dub_wav,
        output_video=output_video,
        scale=scale,
        config=config,
    )

    print("DONE:", final_video, flush=True)
    print("exists:", final_video.exists(), flush=True)
    print("size_mb:", final_video.stat().st_size / 1024 / 1024 if final_video.exists() else None, flush=True)

    _elapsed("TOTAL", total_t0)

    return {
        "output_video": str(final_video),
        "scale": scale,
        "dub_wav": str(dub_wav),
        "slow_original_audio": str(slow_original_audio),
    }