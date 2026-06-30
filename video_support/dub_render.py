from pathlib import Path
import subprocess, shlex, json, math, os, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        duration_workers=None,
        ffmpeg_threads=None,
        tts_batch_size=120,
        part_mix_batch_size=80,
        faststart=True,
        normalize_dub=True,
        keep_parts=False,
    ):
        cpu = os.cpu_count() or 2
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

        self.dub_sample_rate = dub_sample_rate
        self.duration_workers = duration_workers or max(2, min(8, cpu * 2))
        self.ffmpeg_threads = str(ffmpeg_threads or max(2, min(4, cpu)))

        self.tts_batch_size = tts_batch_size
        self.part_mix_batch_size = part_mix_batch_size
        self.faststart = faststart
        self.normalize_dub = normalize_dub
        self.keep_parts = keep_parts


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


def _clean_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for p in path.rglob("*"):
        if p.is_file():
            p.unlink()


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


def _chunked(items, size):
    for i in range(0, len(items), size):
        yield i // size, items[i:i + size]


def _write_filter_script(path, text):
    path = Path(path)
    path.write_text(text, encoding="utf-8")
    return path


def _parse_max_volume_db(text):
    m = re.search(r"max_volume:\s*([\-0-9.]+)\s*dB", text)
    if not m:
        return None
    return float(m.group(1))


def _detect_max_volume_db(path):
    r = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin",
            "-i", str(_require_file(path)),
            "-af", "volumedetect",
            "-f", "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return _parse_max_volume_db(r.stderr or "")


def collect_tts_items(vi_srt, tts_dir, config=None):
    config = config or DubRenderConfig()
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


def compute_timeline_scale(items, config=None):
    config = config or DubRenderConfig()

    ratios = []
    items = sorted(items, key=lambda x: (x["start_ms"], x["index"]))

    for a, b in zip(items, items[1:]):
        gap = int(b["start_ms"]) - int(a["start_ms"])
        need = int(a["duration_ms"]) + config.min_gap_ms

        if gap > 0:
            ratios.append(need / gap)

    if not ratios:
        scale = 1.0
    else:
        ratios.sort()
        pos = int(len(ratios) * config.scale_percentile)
        pos = min(max(pos, 0), len(ratios) - 1)
        scale = ratios[pos]

    scale = min(scale, config.max_timeline_scale)
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


def make_slow_original_audio(rendered_video, source_video=None, scale=1.0, config=None):
    config = config or DubRenderConfig()
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

    speed = 1.0 / scale

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
        total_sec = _ffprobe_duration_seconds(rendered_video) * scale

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


def _build_tts_batch_part(batch_index, batch_items, scale, part_dir, config):
    inputs = []
    filters = []
    labels = []

    for i, item in enumerate(batch_items):
        inputs += ["-i", str(_require_file(item["path"]))]

        delay_ms = int(round(int(item["start_ms"]) * scale))
        label = f"a{i}"

        filters.append(
            f"[{i}:a]"
            f"aresample={config.dub_sample_rate},"
            f"aformat=sample_fmts=fltp:channel_layouts=mono,"
            f"adelay={delay_ms}:all=1"
            f"[{label}]"
        )

        labels.append(f"[{label}]")

    if len(labels) == 1:
        filters.append(f"{labels[0]}anull[mix]")
    else:
        filters.append(
            f"{''.join(labels)}"
            f"amix=inputs={len(labels)}:duration=longest:dropout_transition=0:normalize=0"
            f"[mix]"
        )

    script = part_dir / f"batch_{batch_index:05d}.filter"
    part = part_dir / f"batch_{batch_index:05d}.wav"

    _write_filter_script(script, ";\n".join(filters))

    _run([
        "ffmpeg", "-y",
        "-hide_banner",
        "-nostdin",
        "-threads", config.ffmpeg_threads,
        *inputs,
        "-filter_complex_script", str(script),
        "-map", "[mix]",
        "-ar", str(config.dub_sample_rate),
        "-ac", "1",
        "-c:a", "pcm_f32le",
        str(part),
    ])

    return _require_file(part)


def _mix_audio_files_to_one(inputs, output, work_dir, prefix, sample_rate, channels, codec, config):
    inputs = [Path(x) for x in inputs]

    if not inputs:
        raise RuntimeError("No audio files to mix")

    if len(inputs) == 1:
        _run([
            "ffmpeg", "-y",
            "-hide_banner",
            "-nostdin",
            "-threads", config.ffmpeg_threads,
            "-i", str(_require_file(inputs[0])),
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-c:a", codec,
            str(output),
        ])
        return _require_file(output)

    round_no = 0
    current = inputs

    while len(current) > 1:
        next_round = []
        round_dir = Path(work_dir) / f"{prefix}_round_{round_no:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        for group_index, group in _chunked(current, config.part_mix_batch_size):
            cmd_inputs = []
            filters = []
            labels = []

            for i, audio_path in enumerate(group):
                cmd_inputs += ["-i", str(_require_file(audio_path))]
                label = f"a{i}"
                layout = "mono" if channels == 1 else "stereo"

                filters.append(
                    f"[{i}:a]"
                    f"aresample={sample_rate},"
                    f"aformat=sample_fmts=fltp:channel_layouts={layout}"
                    f"[{label}]"
                )

                labels.append(f"[{label}]")

            if len(labels) == 1:
                filters.append(f"{labels[0]}anull[mix]")
            else:
                filters.append(
                    f"{''.join(labels)}"
                    f"amix=inputs={len(labels)}:duration=longest:dropout_transition=0:normalize=0"
                    f"[mix]"
                )

            is_final_single = len(current) <= config.part_mix_batch_size
            out_path = Path(output) if is_final_single else round_dir / f"mix_{group_index:05d}.wav"
            script = round_dir / f"mix_{group_index:05d}.filter"

            _write_filter_script(script, ";\n".join(filters))

            _run([
                "ffmpeg", "-y",
                "-hide_banner",
                "-nostdin",
                "-threads", config.ffmpeg_threads,
                *cmd_inputs,
                "-filter_complex_script", str(script),
                "-map", "[mix]",
                "-ar", str(sample_rate),
                "-ac", str(channels),
                "-c:a", codec,
                str(out_path),
            ])

            next_round.append(_require_file(out_path))

        current = next_round
        round_no += 1

    if current[0] != Path(output):
        _run([
            "ffmpeg", "-y",
            "-hide_banner",
            "-nostdin",
            "-threads", config.ffmpeg_threads,
            "-i", str(_require_file(current[0])),
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-c:a", codec,
            str(output),
        ])

    return _require_file(output)


def _normalize_dub_audio(raw_path, out_path, config):
    if not config.normalize_dub:
        _run([
            "ffmpeg", "-y",
            "-hide_banner",
            "-nostdin",
            "-threads", config.ffmpeg_threads,
            "-i", str(_require_file(raw_path)),
            "-ar", str(config.dub_sample_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(out_path),
        ])
        return 1.0

    max_db = _detect_max_volume_db(raw_path)

    if max_db is None:
        gain = 1.0
    else:
        target_ratio = config.dub_target_peak / 32767.0
        target_db = 20.0 * math.log10(target_ratio)
        gain_db = target_db - max_db
        max_gain_db = 20.0 * math.log10(config.max_dub_gain)
        gain_db = min(gain_db, max_gain_db)
        gain = 10.0 ** (gain_db / 20.0)

    print("Dub normalize gain:", gain, flush=True)

    _run([
        "ffmpeg", "-y",
        "-hide_banner",
        "-nostdin",
        "-threads", config.ffmpeg_threads,
        "-i", str(_require_file(raw_path)),
        "-af", f"volume={gain:.8f}",
        "-ar", str(config.dub_sample_rate),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out_path),
    ])

    return gain


def build_dub_wav(items, scale, config=None):
    config = config or DubRenderConfig()
    t0 = _now()

    part_dir = config.render_dir / "dub_parts"
    raw_dub_wav = config.render_dir / "dub_slow_timeline_raw_f32.wav"
    dub_wav = config.render_dir / "dub_slow_timeline.wav"

    _clean_dir(part_dir)

    items = sorted(items, key=lambda x: (x["start_ms"], x["index"]))

    print("TTS batch size:", config.tts_batch_size, flush=True)
    print("TTS total:", len(items), flush=True)

    parts = []

    total_batches = math.ceil(len(items) / config.tts_batch_size)

    for batch_index, batch_items in _chunked(items, config.tts_batch_size):
        first_idx = batch_items[0]["index"]
        last_idx = batch_items[-1]["index"]

        print(
            f"build dub batch {batch_index + 1}/{total_batches} "
            f"idx={first_idx}-{last_idx}",
            flush=True,
        )

        part = _build_tts_batch_part(batch_index, batch_items, scale, part_dir, config)
        parts.append(part)

    print("Dub parts:", len(parts), flush=True)

    _mix_audio_files_to_one(
        inputs=parts,
        output=raw_dub_wav,
        work_dir=part_dir,
        prefix="parts",
        sample_rate=config.dub_sample_rate,
        channels=1,
        codec="pcm_f32le",
        config=config,
    )

    gain = _normalize_dub_audio(raw_dub_wav, dub_wav, config)

    report = {
        "mixed_count": len(items),
        "tts_batch_size": config.tts_batch_size,
        "part_count": len(parts),
        "normalize_enabled": config.normalize_dub,
        "normalize_gain": gain,
        "dub_target_peak": config.dub_target_peak,
        "max_dub_gain": config.max_dub_gain,
        "method": "ffmpeg_adelay_amix_batch",
    }

    (config.render_dir / "dub_overlap_mix_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not config.keep_parts:
        for p in part_dir.rglob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass

    _elapsed("build_dub_wav", t0)
    return _require_file(dub_wav)


def mux_final_audio_video(rendered_video, slow_original_audio, dub_wav, output_video, scale, config=None):
    config = config or DubRenderConfig()
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

    print("FFMPEG_THREADS:", config.ffmpeg_threads, flush=True)
    print("DURATION_WORKERS:", config.duration_workers, flush=True)
    print("TTS_BATCH_SIZE:", config.tts_batch_size, flush=True)

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

    dub_wav = build_dub_wav(items, scale, config)

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