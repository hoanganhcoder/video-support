import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

TIMEOUT = 30
PROBE_BYTES = 2 * 1024 * 1024
PROBE_TIMEOUT = 8
ARIA_CONNECTIONS_PER_STREAM = 8

PREFERRED_CDN_HOSTS = [
    "upos-sz-mirrorcos.bilivideo.com",
    "upos-sz-mirroraliov.bilivideo.com",
    "upos-sz-mirrorhw.bilivideo.com",
    "upos-sz-mirror08c.bilivideo.com",
    "upos-hz-mirrorakam.akamaized.net",
]


def extract_bvid(text):
    m = re.search(r"(BV[0-9A-Za-z]+)", str(text))
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
    for cookie in jar:
        domain = cookie.domain.lower()
        if (
            "bilibili.com" in domain
            or "biliapi.net" in domain
            or "biliapi.com" in domain
            or "bilivideo.com" in domain
        ):
            cookies[cookie.name] = cookie.value
    return cookies


def cookies_to_header(cookies):
    return "; ".join(f"{key}={value}" for key, value in cookies.items())


def safe_name(name):
    return re.sub(r'[\\/:*?"<>|]', "_", str(name)).strip()


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
    codec = (codec or "").lower()
    if codec.startswith("avc1"):
        return "H.264/AVC"
    if codec.startswith(("hev1", "hvc1")):
        return "H.265/HEVC"
    if codec.startswith("av01"):
        return "AV1"
    if codec.startswith("mp4a"):
        return "AAC"
    return codec or "?"


def codec_rank(codec):
    codec = (codec or "").lower()
    if codec.startswith("avc1"):
        return 300
    if codec.startswith(("hev1", "hvc1")):
        return 200
    if codec.startswith("av01"):
        return 50
    return 100


def fmt_bytes(value):
    if not value:
        return "?"
    value = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    return f"{value:.2f} {units[index]}"


def replace_url_host(url, host):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme or "https", host, parts.path, parts.query, parts.fragment))


def original_stream_urls(item):
    urls = []
    for key in ("baseUrl", "base_url"):
        url = item.get(key)
        if url:
            urls.append(url)

    for key in ("backupUrl", "backup_url"):
        for url in item.get(key) or []:
            if url:
                urls.append(url)

    return list(dict.fromkeys(urls))


def stream_urls(item):
    originals = original_stream_urls(item)
    rewritten = []

    for original in originals:
        for host in PREFERRED_CDN_HOSTS:
            rewritten.append(replace_url_host(original, host))

    return list(dict.fromkeys(rewritten + originals))


def request_json(url, params, headers, cookies):
    response = requests.get(
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 0:
        raise RuntimeError(data)
    return data


def build_headers(bvid):
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/150.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://www.bilibili.com/video/{bvid}/",
        "Origin": "https://www.bilibili.com",
        "Accept": "*/*",
        "Accept-Encoding": "identity",
    }


def print_pages(info_data):
    print("========== PAGES ==========")
    for page in info_data["pages"]:
        mark = "*" if page["page"] == info_data["page"] else " "
        print(f"{mark} P{page['page']} | cid={page['cid']} | {page['title']}")


def info(video_url, cookie_path="cookies.txt", target_qid=80, page=1):
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
    pages_raw = data.get("pages") or []

    if not pages_raw:
        raise RuntimeError("Không lấy được danh sách page")
    if page < 1 or page > len(pages_raw):
        raise ValueError(f"Page không hợp lệ: {page}. Video có {len(pages_raw)} page")

    selected_page = pages_raw[page - 1]
    cid = selected_page["cid"]
    part = selected_page.get("part") or f"P{page}"

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

    sorted_videos = sorted(
        videos,
        key=lambda item: (
            item.get("id", 0) or 0,
            codec_rank(item.get("codecs")),
            item.get("bandwidth", 0) or 0,
        ),
        reverse=True,
    )

    video_choices = []
    for index, item in enumerate(sorted_videos):
        video_choices.append(
            {
                "index": index,
                "id": item.get("id"),
                "quality": quality_name(item.get("id")),
                "codec": item.get("codecs"),
                "codec_name": codec_label(item.get("codecs")),
                "width": item.get("width"),
                "height": item.get("height"),
                "bandwidth": item.get("bandwidth"),
                "size": item.get("size") or 0,
                "size_text": fmt_bytes(item.get("size") or 0),
                "urls": stream_urls(item),
                "raw": item,
            }
        )

    sorted_audios = sorted(
        audios,
        key=lambda item: item.get("bandwidth", 0) or 0,
        reverse=True,
    )

    audio_choices = []
    for index, item in enumerate(sorted_audios):
        audio_choices.append(
            {
                "index": index,
                "id": item.get("id"),
                "codec": item.get("codecs"),
                "codec_name": codec_label(item.get("codecs")),
                "bandwidth": item.get("bandwidth"),
                "size": item.get("size") or 0,
                "size_text": fmt_bytes(item.get("size") or 0),
                "urls": stream_urls(item),
                "raw": item,
            }
        )

    return {
        "bvid": bvid,
        "cid": cid,
        "title": title,
        "page": page,
        "part": part,
        "pages": [
            {
                "page": item.get("page"),
                "cid": item.get("cid"),
                "title": item.get("part") or f"P{item.get('page')}",
                "duration": item.get("duration"),
            }
            for item in pages_raw
        ],
        "cookies": cookies,
        "headers": headers,
        "support": play_data.get("accept_description") or [],
        "videos": video_choices,
        "audios": audio_choices,
    }


def _probe_one(url, headers, cookies):
    request_headers = dict(headers)
    request_headers["Range"] = f"bytes=0-{PROBE_BYTES - 1}"

    started = time.perf_counter()
    downloaded = 0
    try:
        with requests.get(
            url,
            headers=request_headers,
            cookies=cookies,
            stream=True,
            timeout=(4, PROBE_TIMEOUT),
            allow_redirects=True,
        ) as response:
            if response.status_code not in (200, 206):
                return url, 0.0

            for chunk in response.iter_content(chunk_size=256 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded >= PROBE_BYTES:
                    break

        elapsed = max(time.perf_counter() - started, 0.001)
        return url, downloaded / elapsed
    except Exception:
        return url, 0.0


def rank_urls(urls, headers, cookies, max_candidates=10):
    candidates = list(dict.fromkeys(urls))[:max_candidates]
    if len(candidates) <= 1:
        return candidates

    print(f"[CDN] Đang đo {len(candidates)} URL...")
    results = []

    with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as executor:
        futures = [executor.submit(_probe_one, url, headers, cookies) for url in candidates]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item[1], reverse=True)

    ranked = []
    for position, (url, speed) in enumerate(results, 1):
        host = urlsplit(url).netloc
        if speed > 0:
            print(f"[CDN] {position:>2}. {host:<45} {speed / 1024 / 1024:>7.2f} MB/s")
            ranked.append(url)
        else:
            print(f"[CDN] {position:>2}. {host:<45} lỗi/chậm")

    for url in urls:
        if url not in ranked:
            ranked.append(url)

    return ranked


def aria2_download(url, output_path, headers, cookies, connections=ARIA_CONNECTIONS_PER_STREAM):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aria2c",
        "-x", str(connections),
        "-s", str(connections),
        "-k", "1M",
        "--max-connection-per-server", str(connections),
        "--min-split-size=1M",
        "--file-allocation=none",
        "--continue=true",
        "--max-tries=3",
        "--retry-wait=1",
        "--timeout=30",
        "--connect-timeout=10",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--remote-time=true",
        "--check-certificate=false",
        "--console-log-level=warn",
        "--summary-interval=1",
        "--download-result=hide",
        f"--user-agent={headers.get('User-Agent', '')}",
        f"--referer={headers.get('Referer', '')}",
        "--header", f"Origin: {headers.get('Origin', '')}",
        "--header", "Accept: */*",
        "--header", "Accept-Encoding: identity",
    ]

    cookie_header = cookies_to_header(cookies)
    if cookie_header:
        cmd += ["--header", f"Cookie: {cookie_header}"]

    cmd += ["-d", str(output_path.parent), "-o", output_path.name, url]
    subprocess.run(cmd, check=True)


def download_any(urls, output_path, headers, cookies, label):
    output_path = Path(output_path)
    ranked_urls = rank_urls(urls, headers, cookies)
    last_error = None

    for attempt, url in enumerate(ranked_urls, 1):
        host = urlsplit(url).netloc
        try:
            print(f"[{label}] thử {attempt}/{len(ranked_urls)}: {host}")
            output_path.unlink(missing_ok=True)
            aria2_download(url, output_path, headers, cookies)

            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"[{label}] hoàn tất: {fmt_bytes(output_path.stat().st_size)}")
                return str(output_path)
        except subprocess.CalledProcessError as error:
            last_error = error
            print(f"[{label}] CDN lỗi, chuyển URL khác")
            output_path.unlink(missing_ok=True)

    raise last_error or RuntimeError(f"Không tải được {label}")


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
        return json.loads(result.stdout)
    except Exception as error:
        print("ffprobe skipped:", error)
        return None


def _print_selected(info_data, video=None, audio=None):
    print("========== SELECTED ==========")
    print("Title        :", info_data["title"])
    print("Page         :", f"P{info_data['page']}", info_data["part"])
    print("CID          :", info_data["cid"])

    if video:
        print("Video quality:", video["quality"])
        print("Video codec  :", video["codec_name"], video["codec"])
        print("Video size   :", f"{video['width']}x{video['height']}")
        print("Video bitrate:", video["bandwidth"])
        print("Video file   :", video["size_text"])

    if audio:
        print("Audio codec  :", audio["codec_name"], audio["codec"])
        print("Audio bitrate:", audio["bandwidth"])
        print("Audio file   :", audio["size_text"])


def download(
    info_data,
    video_index,
    output_path,
    audio_index=0,
    temp_dir="downloads",
    only_audio=False,
    keep_temp=False,
):
    output_path = Path(output_path)
    temp_dir = Path(temp_dir)
    headers = info_data["headers"]
    cookies = info_data["cookies"]

    video = info_data["videos"][video_index]
    audio = info_data["audios"][audio_index]

    if not audio["urls"]:
        raise RuntimeError("Không lấy được audio URL")
    if not only_audio and not video["urls"]:
        raise RuntimeError("Không lấy được video URL")

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_tmp = temp_dir / f"{output_path.stem}.video.m4s"
    audio_tmp = temp_dir / f"{output_path.stem}.audio.m4s"

    if only_audio:
        _print_selected(info_data, audio=audio)
        download_any(audio["urls"], audio_tmp, headers, cookies, "audio")

        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-y",
                "-i", str(audio_tmp),
                "-c", "copy",
                str(output_path),
            ],
            check=True,
        )

        if not keep_temp:
            audio_tmp.unlink(missing_ok=True)

        print("Done:", output_path)
        return str(output_path)

    video_tmp.unlink(missing_ok=True)
    audio_tmp.unlink(missing_ok=True)
    _print_selected(info_data, video=video, audio=audio)

    print("\nTải video và audio song song...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_video = executor.submit(
            download_any,
            video["urls"],
            video_tmp,
            headers,
            cookies,
            "video",
        )
        future_audio = executor.submit(
            download_any,
            audio["urls"],
            audio_tmp,
            headers,
            cookies,
            "audio",
        )
        future_video.result()
        future_audio.result()

    print("\nGhép video + audio, không encode...")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-y",
            "-i", str(video_tmp),
            "-i", str(audio_tmp),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ],
        check=True,
    )

    if not keep_temp:
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