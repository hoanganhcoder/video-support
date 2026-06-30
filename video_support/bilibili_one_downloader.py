import json
import re
import subprocess
import time
from pathlib import Path
from http.cookiejar import MozillaCookieJar

import requests


TIMEOUT = 30
CONNECTIONS = 4


def extract_bvid(text):
    m = re.search(r"(BV[0-9A-Za-z]+)", text)
    if not m:
        raise ValueError("Không tìm thấy BV id")
    return m.group(1)


def load_cookies_txt(cookie_path="cookies.txt"):
    cookie_path = Path(cookie_path)
    if not cookie_path.exists():
        return {}

    jar = MozillaCookieJar(str(cookie_path))
    jar.load(ignore_discard=True, ignore_expires=True)

    cookies = {}
    for c in jar:
        d = c.domain.lower()
        if "bilibili.com" in d or "biliapi.net" in d or "biliapi.com" in d:
            cookies[c.name] = c.value

    return cookies


def cookies_to_header(cookies):
    return "; ".join(f"{k}={v}" for k, v in cookies.items())


def quality_name(qid):
    return {
        127: "8K",
        126: "Dolby Vision",
        125: "HDR",
        120: "4K",
        116: "1080P60",
        112: "1080P+",
        80: "1080P",
        74: "720P60",
        64: "720P",
        32: "480P",
        16: "360P",
        6: "240P",
    }.get(qid, str(qid))


def codec_label(codec):
    c = (codec or "").lower()
    if c.startswith("avc1"):
        return "H.264/AVC"
    if c.startswith(("hev1", "hvc1")):
        return "H.265/HEVC"
    if c.startswith("av01"):
        return "AV1"
    if c.startswith("mp4a"):
        return "AAC"
    return codec or "?"


def codec_rank(codec):
    c = (codec or "").lower()
    if c.startswith("avc1"):
        return 300
    if c.startswith(("hev1", "hvc1")):
        return 200
    if c.startswith("av01"):
        return 50
    return 100


def fmt_bytes(n):
    if not n:
        return "?"
    n = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024
        i += 1
    return f"{n:.2f} {units[i]}"


def stream_urls(item):
    urls = []

    for k in ("baseUrl", "base_url"):
        u = item.get(k)
        if u:
            urls.append(u)

    for k in ("backupUrl", "backup_url"):
        for u in item.get(k) or []:
            if u:
                urls.append(u)

    return list(dict.fromkeys(urls))


def request_json(url, params, headers, cookies):
    r = requests.get(url, params=params, headers=headers, cookies=cookies, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 0:
        raise RuntimeError(data)
    return data


def build_headers(bvid):
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Referer": f"https://www.bilibili.com/video/{bvid}/",
        "Origin": "https://www.bilibili.com",
    }


def info(video_url, cookie_path="cookies.txt", target_qid=80):
    cookies = load_cookies_txt(cookie_path)
    bvid = extract_bvid(video_url)
    headers = build_headers(bvid)

    view = request_json(
        "https://api.bilibili.com/x/web-interface/view",
        params={"bvid": bvid},
        headers=headers,
        cookies=cookies,
    )

    data = view["data"]
    title = data["title"]
    cid = data["cid"]

    play = request_json(
        "https://api.bilibili.com/x/player/playurl",
        params={
            "bvid": bvid,
            "cid": cid,
            "qn": target_qid,
            "fnval": 4048,
            "fourk": 1,
        },
        headers=headers,
        cookies=cookies,
    )

    play_data = play.get("data") or {}
    dash = play_data.get("dash") or {}

    videos = dash.get("video") or []
    audios = dash.get("audio") or []

    if not videos:
        raise RuntimeError("Không có video stream")
    if not audios:
        raise RuntimeError("Không có audio stream")

    video_choices = []
    for i, v in enumerate(sorted(
        videos,
        key=lambda x: (
            x.get("id", 0) or 0,
            codec_rank(x.get("codecs")),
            x.get("bandwidth", 0) or 0,
        ),
        reverse=True,
    )):
        video_choices.append({
            "index": i,
            "id": v.get("id"),
            "quality": quality_name(v.get("id")),
            "codec": v.get("codecs"),
            "codec_name": codec_label(v.get("codecs")),
            "width": v.get("width"),
            "height": v.get("height"),
            "bandwidth": v.get("bandwidth"),
            "size": v.get("size") or 0,
            "size_text": fmt_bytes(v.get("size") or 0),
            "urls": stream_urls(v),
            "raw": v,
        })

    audio_choices = []
    for i, a in enumerate(sorted(audios, key=lambda x: x.get("bandwidth", 0) or 0, reverse=True)):
        audio_choices.append({
            "index": i,
            "id": a.get("id"),
            "codec": a.get("codecs"),
            "codec_name": codec_label(a.get("codecs")),
            "bandwidth": a.get("bandwidth"),
            "size": a.get("size") or 0,
            "size_text": fmt_bytes(a.get("size") or 0),
            "urls": stream_urls(a),
            "raw": a,
        })

    return {
        "bvid": bvid,
        "cid": cid,
        "title": title,
        "cookies": cookies,
        "headers": headers,
        "support": play_data.get("accept_description") or [],
        "videos": video_choices,
        "audios": audio_choices,
    }


def aria2_download(url, output_path, headers, cookies, connections=CONNECTIONS):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aria2c",
        "-x", str(connections),
        "-s", str(connections),
        "--split", str(connections),
        "--max-connection-per-server", str(connections),
        "--min-split-size=4M",
        "-j", "1",
        "--continue=true",
        "--file-allocation=none",
        "--retry-wait=3",
        "--max-tries=5",
        "--timeout=60",
        "--connect-timeout=20",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--console-log-level=notice",
        "--summary-interval=1",
        f"--user-agent={headers.get('User-Agent')}",
        f"--referer={headers.get('Referer')}",
        "--header", f"Origin: {headers.get('Origin')}",
        "--check-certificate=false",
    ]

    cookie_header = cookies_to_header(cookies)
    if cookie_header:
        cmd += ["--header", f"Cookie: {cookie_header}"]

    cmd += ["-d", str(output_path.parent), "-o", output_path.name, url]

    subprocess.run(cmd, check=True)


def download_any(urls, output_path, headers, cookies):
    last_error = None

    for i, url in enumerate(urls, 1):
        try:
            print(f"[download] try {i}/{len(urls)}")
            aria2_download(url, output_path, headers, cookies)

            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                return

        except subprocess.CalledProcessError as e:
            last_error = e
            print(f"[download failed] code={e.returncode}")
            Path(output_path).unlink(missing_ok=True)
            time.sleep(1)

    raise last_error or RuntimeError("Download failed")


def ffprobe_info(file_path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries",
                "format=format_name,duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,avg_frame_rate,bit_rate,sample_rate,channels",
                "-of", "json",
                str(file_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except Exception as e:
        print("ffprobe skipped:", e)
        return None

    return json.loads(result.stdout)


def download(info_data, video_index, output_path, audio_index=0, temp_dir="downloads", only_audio=False):
    output_path = Path(output_path)
    temp_dir = Path(temp_dir)

    headers = info_data["headers"]
    cookies = info_data["cookies"]

    video = info_data["videos"][video_index]
    audio = info_data["audios"][audio_index]

    if not video["urls"]:
        raise RuntimeError("Không lấy được video URL")
    if not audio["urls"]:
        raise RuntimeError("Không lấy được audio URL")

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_tmp = temp_dir / f"{output_path.stem}.video.m4s"
    audio_tmp = temp_dir / f"{output_path.stem}.audio.m4s"

    if only_audio:
        print("========== SELECTED ==========")
        print("Title       :", info_data["title"])
        print("Audio codec :", audio["codec_name"], audio["codec"])
        print("Audio bitrate:", audio["bandwidth"])
        print("Audio file  :", audio["size_text"])

        print("\nDownloading audio...")
        download_any(audio["urls"], audio_tmp, headers, cookies)

        print("\nMerging...")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-nostats",
                "-y",
                "-i", str(audio_tmp),
                "-c", "copy",
                str(output_path),
            ],
            check=True,
        )

        audio_tmp.unlink(missing_ok=True)

        print("\nDone:", output_path)

        probe = ffprobe_info(output_path)
        if probe:
            fmt = probe.get("format", {})
            print("Duration:", fmt.get("duration"))
            print("Size    :", fmt.get("size"))
            print("Bitrate :", fmt.get("bit_rate"))

        return str(output_path)

    video_tmp.unlink(missing_ok=True)
    audio_tmp.unlink(missing_ok=True)

    print("========== SELECTED ==========")
    print("Title        :", info_data["title"])
    print("Video quality:", video["quality"])
    print("Video codec  :", video["codec_name"], video["codec"])
    print("Video size   :", f"{video['width']}x{video['height']}")
    print("Video bitrate:", video["bandwidth"])
    print("Video file   :", video["size_text"])
    print("Audio codec  :", audio["codec_name"], audio["codec"])
    print("Audio bitrate:", audio["bandwidth"])
    print("Audio file   :", audio["size_text"])

    print("\nDownloading video...")
    download_any(video["urls"], video_tmp, headers, cookies)

    print("\nDownloading audio...")
    download_any(audio["urls"], audio_tmp, headers, cookies)

    print("\nMerging...")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-nostats",
            "-y",
            "-i", str(video_tmp),
            "-i", str(audio_tmp),
            "-c", "copy",
            str(output_path),
        ],
        check=True,
    )

    video_tmp.unlink(missing_ok=True)
    audio_tmp.unlink(missing_ok=True)

    print("\nDone:", output_path)

    probe = ffprobe_info(output_path)
    if probe:
        fmt = probe.get("format", {})
        print("Duration:", fmt.get("duration"))
        print("Size    :", fmt.get("size"))
        print("Bitrate :", fmt.get("bit_rate"))

    return str(output_path)