from pathlib import Path
import subprocess
import shlex
import json
import math
import time
import shutil
import re


class DubRenderConfig:
    def __init__(
        self,
        render_dir="/content/render",
        original_volume=0.045,
        dub_volume=2.35,
        final_mix_volume=1.45,
        limiter_limit=0.98,
        target_overlap=0.15,
        max_video_stretch=1.29,
        scale_percentile=0.90,
        dub_mix_inputs_per_pass=96,
        dub_sample_rate=24000,
        ffmpeg_threads=4,
        video_encoder="h264_nvenc",
        video_preset="p4",
        video_cq=23,
        faststart=False,
        keep_work=False,
    ):
        self.render_dir = Path(render_dir)
        self.render_dir.mkdir(parents=True, exist_ok=True)

        self.original_volume = float(original_volume)
        self.dub_volume = float(dub_volume)
        self.final_mix_volume = float(final_mix_volume)
        self.limiter_limit = float(limiter_limit)

        self.target_overlap = float(target_overlap)
        self.max_video_stretch = float(max_video_stretch)
        self.scale_percentile = float(scale_percentile)

        self.dub_mix_inputs_per_pass = int(dub_mix_inputs_per_pass)
        self.dub_sample_rate = int(dub_sample_rate)
        self.ffmpeg_threads = str(ffmpeg_threads)

        self.video_encoder = str(video_encoder)
        self.video_preset = str(video_preset)
        self.video_cq = int(video_cq)
        self.faststart = bool(faststart)
        self.keep_work = bool(keep_work)


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


def _clean_dir(path):
    path = Path(path)

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def _wav_duration_ms(path):
    path = _require_file(path)

    with path.open("rb") as f:
        riff = f.read(12)

        if len(riff) < 12 or riff[0:4] not in (b"RIFF", b"RF64") or riff[8:12] != b"WAVE":
            raise RuntimeError(f"Not a RIFF/WAVE file: {path}")

        byte_rate = None
        block_align = None
        sample_rate = None
        data_size = None

        while True:
            header = f.read(8)

            if len(header) < 8:
                break

            chunk_id = header[0:4]
            chunk_size = int.from_bytes(header[4:8], "little", signed=False)
            chunk_start = f.tell()

            if chunk_id == b"fmt ":
                fmt = f.read(min(chunk_size, 32))

                if len(fmt) < 16:
                    raise RuntimeError(f"Invalid WAV fmt chunk: {path}")

                sample_rate = int.from_bytes(fmt[4:8], "little", signed=False)
                byte_rate = int.from_bytes(fmt[8:12], "little", signed=False)
                block_align = int.from_bytes(fmt[12:14], "little", signed=False)
            elif chunk_id == b"data":
                data_size = chunk_size
                break

            f.seek(chunk_start + chunk_size + (chunk_size % 2))

    if data_size is None:
        raise RuntimeError(f"WAV has no data chunk: {path}")

    if byte_rate and byte_rate > 0:
        return int(math.ceil(data_size * 1000.0 / byte_rate))

    if sample_rate and block_align and sample_rate > 0 and block_align > 0:
        frames = data_size / block_align
        return int(math.ceil(frames * 1000.0 / sample_rate))

    raise RuntimeError(f"WAV duration is not readable: {path}")


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


def _srt_time_to_ms(value):
    match = re.match(
        r"^\s*(\d+):(\d+):(\d+)[,.](\d+)\s*$",
        str(value),
    )

    if not match:
        raise ValueError(f"Invalid SRT time: {value}")

    hours, minutes, seconds, millis = match.groups()
    millis = (millis + "000")[:3]

    return (
        int(hours) * 3600000
        + int(minutes) * 60000
        + int(seconds) * 1000
        + int(millis)
    )


def _read_srt_entries(path):
    text = _require_file(path).read_text(encoding="utf-8-sig", errors="ignore")
    entries = []

    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [line.strip() for line in block.splitlines() if line.strip()]

        if len(lines) < 2:
            continue

        try:
            index = int(lines[0])
            timing = lines[1]
            content_lines = lines[2:]
        except ValueError:
            index = len(entries) + 1
            timing = lines[0]
            content_lines = lines[1:]

        if "-->" not in timing:
            continue

        start_text, end_text = timing.split("-->", 1)

        entries.append({
            "index": index,
            "start_ms": max(0, _srt_time_to_ms(start_text)),
            "end_ms": max(0, _srt_time_to_ms(end_text)),
            "content": " ".join(content_lines).strip(),
        })

    entries.sort(key=lambda x: (x["start_ms"], x["index"]))
    return entries


def _chunked(seq, size):
    size = max(1, int(size))

    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def collect_tts_items(vi_srt, tts_dir, config):
    t0 = _now()

    tts_dir = Path(tts_dir)
    items = []
    missing = []
    failed = []

    for entry in _read_srt_entries(vi_srt):
        index = int(entry["index"])
        wav_path = tts_dir / f"{index:05d}.wav"

        if not wav_path.exists() or wav_path.stat().st_size <= 44:
            missing.append(index)
            continue

        try:
            duration_ms = _wav_duration_ms(wav_path)
            items.append({
                "index": index,
                "start_ms": int(entry["start_ms"]),
                "duration_ms": duration_ms,
                "path": wav_path,
            })
        except Exception as exc:
            failed.append({
                "index": index,
                "path": str(wav_path),
                "error": str(exc),
            })

    print("TTS clips found:", len(items), flush=True)
    print("Missing:", len(missing), flush=True)
    print("Bad WAV:", len(failed), flush=True)

    if missing:
        (config.render_dir / "missing_tts_for_dub.json").write_text(
            json.dumps(missing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if failed:
        (config.render_dir / "bad_tts_wav_for_dub.json").write_text(
            json.dumps(failed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    items.sort(key=lambda x: (x["start_ms"], x["index"]))

    print("TTS clips usable:", len(items), flush=True)
    _elapsed("collect_tts_items", t0)
    return items


def compute_timing_plan(items, config):
    target_overlap = min(max(float(config.target_overlap), 0.0), 0.95)
    required_scales = []

    for current, nxt in zip(items, items[1:]):
        gap = int(nxt["start_ms"]) - int(current["start_ms"])
        duration = int(current["duration_ms"])

        if gap <= 0 or duration <= 0:
            continue

        required_scales.append((duration * (1.0 - target_overlap)) / gap)

    if required_scales:
        required_scales.sort()
        pos = int(len(required_scales) * float(config.scale_percentile))
        pos = min(max(pos, 0), len(required_scales) - 1)
        wanted_stretch = required_scales[pos]
    else:
        wanted_stretch = 1.0

    video_stretch = min(float(config.max_video_stretch), max(1.0, wanted_stretch))

    overlap_ratios = []
    overlap_pairs = 0

    for current, nxt in zip(items, items[1:]):
        gap = (int(nxt["start_ms"]) - int(current["start_ms"])) * video_stretch
        duration = int(current["duration_ms"])

        if duration <= 0:
            continue

        overlap = max(0.0, duration - gap)
        ratio = overlap / duration
        overlap_ratios.append(ratio)

        if ratio > target_overlap:
            overlap_pairs += 1

    avg_overlap = sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0.0
    max_overlap = max(overlap_ratios) if overlap_ratios else 0.0

    report = {
        "video_stretch": video_stretch,
        "wanted_stretch": wanted_stretch,
        "max_video_stretch": config.max_video_stretch,
        "target_overlap": target_overlap,
        "scale_percentile": config.scale_percentile,
        "pairs": len(overlap_ratios),
        "pairs_over_target": overlap_pairs,
        "avg_overlap": avg_overlap,
        "max_overlap": max_overlap,
    }

    (config.render_dir / "timing_plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("video_stretch:", video_stretch, flush=True)
    print("target_overlap:", target_overlap, flush=True)
    print("pairs_over_target:", overlap_pairs, "/", len(overlap_ratios), flush=True)

    return report


def _dub_filter_script(events, sample_rate, base_start_ms=0, batch_size=32):
    lines = []
    batch_labels = []

    for batch_index, batch in enumerate(_chunked(events, batch_size)):
        item_labels = []

        for event in batch:
            input_index = int(event["input_index"])
            delay_ms = max(0, int(event["start_ms"]) - int(base_start_ms))
            label = f"c{input_index}"

            lines.append(
                f"[{input_index}:a]"
                f"aresample={sample_rate},"
                "aformat=sample_fmts=fltp:channel_layouts=mono,"
                f"adelay={delay_ms}"
                f"[{label}]"
            )
            item_labels.append(f"[{label}]")

        if len(item_labels) == 1:
            batch_labels.append(item_labels[0])
        else:
            label = f"b{batch_index}"
            lines.append(
                "".join(item_labels)
                + f"amix=inputs={len(item_labels)}:duration=longest:dropout_transition=0:normalize=0"
                + f"[{label}]"
            )
            batch_labels.append(f"[{label}]")

    if len(batch_labels) == 1:
        lines.append(f"{batch_labels[0]}anull[outa]")
    else:
        lines.append(
            "".join(batch_labels)
            + f"amix=inputs={len(batch_labels)}:duration=longest:dropout_transition=0:normalize=0"
            + "[outa]"
        )

    return ";\n".join(lines)


def _run_audio_mix(events, output_path, script_path, config, base_start_ms=0):
    events = [
        {
            **event,
            "input_index": input_index,
        }
        for input_index, event in enumerate(events)
    ]

    script_path.write_text(
        _dub_filter_script(
            events=events,
            sample_rate=config.dub_sample_rate,
            base_start_ms=base_start_ms,
        ),
        encoding="utf-8",
    )

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-nostats",
        "-nostdin",
        "-threads", config.ffmpeg_threads,
        "-filter_complex_threads", config.ffmpeg_threads,
    ]

    for event in events:
        cmd += ["-i", str(_require_file(event["path"]))]

    cmd += [
        "-filter_complex_script", str(script_path),
        "-map", "[outa]",
        "-ac", "1",
        "-ar", str(config.dub_sample_rate),
        "-c:a", "pcm_s16le",
        str(output_path),
    ]

    _run(cmd)
    return _require_file(output_path)


def build_dub_wav(items, video_stretch, config):
    t0 = _now()
    work_dir = config.render_dir / "dub_mix_work"
    final_dub = config.render_dir / "dub_timeline.wav"

    _clean_dir(work_dir)

    events = [
        {
            "index": int(item["index"]),
            "start_ms": int(round(int(item["start_ms"]) * float(video_stretch))),
            "path": Path(item["path"]),
        }
        for item in items
    ]
    events.sort(key=lambda x: (x["start_ms"], x["index"]))

    if not events:
        raise RuntimeError("No TTS clips to mix.")

    max_inputs = max(1, int(config.dub_mix_inputs_per_pass))
    batch_items = []

    if len(events) <= max_inputs:
        _run_audio_mix(
            events=events,
            output_path=final_dub,
            script_path=config.render_dir / "dub_mix.filter.txt",
            config=config,
        )
    else:
        total_batches = math.ceil(len(events) / max_inputs)

        for batch_index, batch in enumerate(_chunked(events, max_inputs)):
            batch_start_ms = min(int(event["start_ms"]) for event in batch)
            batch_path = work_dir / f"batch_{batch_index:05d}.wav"

            print(
                f"dub batch {batch_index + 1}/{total_batches} items={len(batch)}",
                flush=True,
            )

            _run_audio_mix(
                events=batch,
                output_path=batch_path,
                script_path=work_dir / f"batch_{batch_index:05d}.filter.txt",
                config=config,
                base_start_ms=batch_start_ms,
            )

            batch_items.append({
                "index": batch_index,
                "start_ms": batch_start_ms,
                "path": batch_path,
            })

        _run_audio_mix(
            events=batch_items,
            output_path=final_dub,
            script_path=config.render_dir / "dub_mix.filter.txt",
            config=config,
        )

    if not config.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    _elapsed("build_dub_wav", t0)
    return _require_file(final_dub)


def mux_final_video(rendered_video, dub_wav, output_video, video_stretch, config, source_video=None):
    t0 = _now()
    rendered_video = Path(rendered_video)
    source_video = Path(source_video) if source_video else None
    output_video = Path(output_video)

    audio_source = source_video or rendered_video

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-nostats",
        "-nostdin",
        "-threads", config.ffmpeg_threads,
        "-i", str(_require_file(rendered_video)),
    ]

    if audio_source.resolve() == rendered_video.resolve():
        orig_label = "0:a"
        dub_label = "1:a"
    else:
        orig_label = "1:a"
        dub_label = "2:a"
        cmd += ["-i", str(_require_file(audio_source))]

    cmd += ["-i", str(_require_file(dub_wav))]

    audio_speed = 1.0 / float(video_stretch)
    filters = []

    if video_stretch > 1.0001:
        filters.append(f"[0:v]setpts={video_stretch:.8f}*PTS[vout]")

    filters.append(
        f"[{orig_label}]"
        f"volume={config.original_volume},"
        f"{_atempo_chain(audio_speed)},"
        "aresample=44100,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo[orig]"
    )
    filters.append(
        f"[{dub_label}]"
        f"volume={config.dub_volume},"
        "aresample=44100,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo[dub]"
    )
    filters.append(
        "[orig][dub]"
        "amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
        f"volume={config.final_mix_volume},"
        f"alimiter=limit={config.limiter_limit}[aout]"
    )

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]" if video_stretch > 1.0001 else "0:v:0",
        "-map", "[aout]",
    ]

    if video_stretch > 1.0001:
        cmd += ["-c:v", config.video_encoder]

        if config.video_encoder == "h264_nvenc":
            cmd += ["-preset", config.video_preset, "-cq", str(config.video_cq)]
        elif config.video_encoder == "libx264":
            cmd += ["-preset", "veryfast", "-crf", str(config.video_cq)]
    else:
        cmd += ["-c:v", "copy"]

    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]

    if config.faststart and rendered_video.stat().st_size <= 2 * 1024 * 1024 * 1024:
        cmd += ["-movflags", "+faststart"]
    elif config.faststart:
        print("Skip faststart for large video.", flush=True)

    cmd += [str(output_video)]

    _run(cmd)

    final_video = _require_file(output_video)
    _elapsed("mux_final_video", t0)
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

    _require_file(rendered_video)
    _require_file(vi_srt)

    items = collect_tts_items(vi_srt, tts_dir, config)

    if not items:
        raise RuntimeError("No TTS clips found. Check TTS_DIR and VI_SRT indices.")

    timing = compute_timing_plan(items, config)
    video_stretch = float(timing["video_stretch"])

    dub_wav = build_dub_wav(
        items=items,
        video_stretch=video_stretch,
        config=config,
    )

    final_video = mux_final_video(
        rendered_video=rendered_video,
        dub_wav=dub_wav,
        output_video=output_video,
        video_stretch=video_stretch,
        config=config,
        source_video=source_video,
    )

    print("DONE:", final_video, flush=True)
    print("exists:", final_video.exists(), flush=True)
    print(
        "size_mb:",
        final_video.stat().st_size / 1024 / 1024 if final_video.exists() else None,
        flush=True,
    )

    _elapsed("TOTAL", total_t0)

    return {
        "output_video": str(final_video),
        "video_stretch": video_stretch,
        "target_overlap": config.target_overlap,
        "dub_wav": str(dub_wav),
    }
