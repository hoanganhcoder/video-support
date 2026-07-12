from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests



API_TIMEOUT = 30

CDN_TEST_BYTES = 4 * 1024 * 1024
CDN_CONNECT_TIMEOUT = 5
CDN_READ_TIMEOUT = 10
CDN_TEST_WORKERS = 8

VIDEO_CONNECTIONS = 16
AUDIO_CONNECTIONS = 8
MAX_CONNECTIONS = 16

ARIA2_CONNECT_TIMEOUT = 15
ARIA2_TIMEOUT = 60
ARIA2_LOWEST_SPEED = "100K"

DEFAULT_TEMP_DIR = "downloads"



@dataclass(frozen=True)
class DownloadResult:
    path: Path
    url: str
    host: str


@dataclass(frozen=True)
class CDNTestResult:
    url: str
    host: str
    ok: bool
    received_bytes: int
    elapsed: float
    speed_bytes: float
    speed_mbps: float
    error: str | None


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Không tìm thấy chương trình '{name}'. "
            f"Hãy cài đặt trước khi tải."
        )


def require_dependencies() -> None:
    require_binary("aria2c")
    require_binary("ffmpeg")
    require_binary("ffprobe")


def extract_bvid(text: str) -> str:
    match = re.search(r"(BV[0-9A-Za-z]+)", str(text))
    if not match:
        raise ValueError("Không tìm thấy BV id trong URL")
    return match.group(1)


def safe_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]', "_", str(name))
    cleaned = cleaned.strip().rstrip(".")
    return cleaned or "video"


def fmt_bytes(value: int | float | None) -> str:
    if not value:
        return "?"

    size = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0

    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1

    return f"{size:.2f} {units[index]}"


def quality_name(qid: int | None) -> str:
    names = {
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
    }
    return names.get(qid, str(qid))


def codec_label(codec: str | None) -> str:
    value = (codec or "").lower()

    if value.startswith("avc1"):
        return "H.264/AVC"

    if value.startswith(("hev1", "hvc1")):
        return "H.265/HEVC"

    if value.startswith("av01"):
        return "AV1"

    if value.startswith("mp4a"):
        return "AAC"

    return codec or "?"


def codec_family(codec: str | None) -> str:
    value = (codec or "").lower()

    if value.startswith("avc1"):
        return "avc"

    if value.startswith(("hev1", "hvc1")):
        return "hevc"

    if value.startswith("av01"):
        return "av1"

    if value.startswith("mp4a"):
        return "aac"

    return value


def unique_urls(items: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for item in items:
        if not item:
            continue

        if item in seen:
            continue

        seen.add(item)
        result.append(item)

    return result


def stream_urls(stream: dict[str, Any]) -> list[str]:
    urls: list[str | None] = [
        stream.get("baseUrl"),
        stream.get("base_url"),
    ]

    for key in ("backupUrl", "backup_url"):
        urls.extend(stream.get(key) or [])

    return unique_urls(urls)


def url_host(url: str) -> str:
    try:
        return urlparse(url).netloc or "unknown"
    except Exception:
        return "unknown"


def load_cookies_txt(
    cookie_path: str | Path = "cookies.txt",
) -> dict[str, str]:
    path = Path(cookie_path)

    if not path.exists():
        return {}

    jar = MozillaCookieJar(str(path))

    try:
        jar.load(
            ignore_discard=True,
            ignore_expires=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Không đọc được cookie file: {path}"
        ) from exc

    cookies: dict[str, str] = {}

    allowed_domains = (
        "bilibili.com",
        "biliapi.net",
        "biliapi.com",
        "bilivideo.com",
    )

    for cookie in jar:
        domain = cookie.domain.lower()

        if any(item in domain for item in allowed_domains):
            cookies[cookie.name] = cookie.value

    return cookies


def cookies_to_header(cookies: dict[str, str]) -> str:
    return "; ".join(
        f"{key}={value}"
        for key, value in cookies.items()
    )


def build_headers(bvid: str) -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/150.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://www.bilibili.com/video/{bvid}/",
        "Origin": "https://www.bilibili.com",
        "Accept-Encoding": "identity",
    }


def create_session(
    headers: dict[str, str],
    cookies: dict[str, str],
) -> requests.Session:
    session = requests.Session()

    session.headers.update(headers)
    session.cookies.update(cookies)

    adapter = requests.adapters.HTTPAdapter(
        pool_connections=16,
        pool_maxsize=32,
        max_retries=0,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    response = session.get(
        url,
        params=params,
        timeout=API_TIMEOUT,
    )

    response.raise_for_status()

    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"API không trả về JSON hợp lệ: {url}"
        ) from exc

    if payload.get("code") != 0:
        raise RuntimeError(
            f"Bilibili API lỗi: {payload}"
        )

    return payload


def info(
    video_url: str,
    cookie_path: str | Path = "cookies.txt",
    target_qid: int = 80,
    page: int = 1,
) -> dict[str, Any]:

    cookies = load_cookies_txt(cookie_path)
    bvid = extract_bvid(video_url)
    headers = build_headers(bvid)

    with create_session(headers, cookies) as session:
        view_payload = request_json(
            session=session,
            url="https://api.bilibili.com/x/web-interface/view",
            params={"bvid": bvid},
        )

        view_data = view_payload.get("data") or {}
        pages_raw = view_data.get("pages") or []

        if not pages_raw:
            raise RuntimeError(
                "Không lấy được danh sách page của video"
            )

        if page < 1 or page > len(pages_raw):
            raise ValueError(
                f"Page {page} không hợp lệ. "
                f"Video có {len(pages_raw)} page."
            )

        selected_page = pages_raw[page - 1]

        cid = selected_page.get("cid")
        if not cid:
            raise RuntimeError(
                f"Không lấy được CID của page {page}"
            )

        play_payload = request_json(
            session=session,
            url="https://api.bilibili.com/x/player/playurl",
            params={
                "bvid": bvid,
                "cid": cid,
                "qn": target_qid,
                "fnval": 4048,
                "fourk": 1,
            },
        )

    play_data = play_payload.get("data") or {}
    dash = play_data.get("dash") or {}

    all_videos = dash.get("video") or []
    all_audios = dash.get("audio") or []

    if not all_videos:
        raise RuntimeError(
            "Bilibili không trả về video DASH stream"
        )

    if not all_audios:
        raise RuntimeError(
            "Bilibili không trả về audio DASH stream"
        )

    exact_videos = [
        stream
        for stream in all_videos
        if stream.get("id") == target_qid
    ]

    if not exact_videos:
        returned_qids = sorted({
            stream.get("id")
            for stream in all_videos
            if stream.get("id") is not None
        })

        raise RuntimeError(
            f"Không có đúng chất lượng qid={target_qid} "
            f"({quality_name(target_qid)}). "
            f"API chỉ trả về các qid: {returned_qids}. "
            f"Đã dừng, không fallback."
        )

    videos: list[dict[str, Any]] = []

    for index, stream in enumerate(exact_videos):
        urls = stream_urls(stream)

        if not urls:
            raise RuntimeError(
                f"Video stream qid={target_qid}, "
                f"codec={stream.get('codecs')} không có URL"
            )

        videos.append({
            "index": index,
            "id": stream.get("id"),
            "quality": quality_name(stream.get("id")),
            "codec": stream.get("codecs"),
            "codec_name": codec_label(stream.get("codecs")),
            "codec_family": codec_family(stream.get("codecs")),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "frame_rate": stream.get("frameRate")
                or stream.get("frame_rate"),
            "bandwidth": stream.get("bandwidth"),
            "size": stream.get("size") or 0,
            "size_text": fmt_bytes(stream.get("size") or 0),
            "urls": urls,
            "raw": stream,
        })

    audios: list[dict[str, Any]] = []

    sorted_audios = sorted(
        all_audios,
        key=lambda stream: stream.get("bandwidth", 0) or 0,
        reverse=True,
    )

    for index, stream in enumerate(sorted_audios):
        urls = stream_urls(stream)

        if not urls:
            continue

        audios.append({
            "index": index,
            "id": stream.get("id"),
            "codec": stream.get("codecs"),
            "codec_name": codec_label(stream.get("codecs")),
            "codec_family": codec_family(stream.get("codecs")),
            "bandwidth": stream.get("bandwidth"),
            "size": stream.get("size") or 0,
            "size_text": fmt_bytes(stream.get("size") or 0),
            "urls": urls,
            "raw": stream,
        })

    if not audios:
        raise RuntimeError(
            "Không có audio stream nào chứa URL hợp lệ"
        )

    return {
        "bvid": bvid,
        "cid": cid,
        "title": view_data.get("title") or bvid,
        "page": page,
        "part": selected_page.get("part") or f"P{page}",
        "target_qid": target_qid,
        "quality": quality_name(target_qid),
        "accept_quality": play_data.get("accept_quality") or [],
        "accept_description":
            play_data.get("accept_description") or [],
        "pages": [
            {
                "page": item.get("page"),
                "cid": item.get("cid"),
                "title": (
                    item.get("part")
                    or f"P{item.get('page')}"
                ),
                "duration": item.get("duration"),
            }
            for item in pages_raw
        ],
        "headers": headers,
        "cookies": cookies,
        "videos": videos,
        "audios": audios,
    }


def print_pages(info_data: dict[str, Any]) -> None:
    print("========== PAGES ==========")

    for item in info_data["pages"]:
        mark = (
            "*"
            if item["page"] == info_data["page"]
            else " "
        )

        print(
            f"{mark} P{item['page']} "
            f"| cid={item['cid']} "
            f"| {item['title']}"
        )


def print_streams(info_data: dict[str, Any]) -> None:
    print("\n========== VIDEO STREAMS ==========")

    for video in info_data["videos"]:
        print(
            f"[{video['index']}] "
            f"qid={video['id']} "
            f"{video['quality']:<8} "
            f"{video['codec_name']:<11} "
            f"{video['width']}x{video['height']} "
            f"bandwidth={video['bandwidth']} "
            f"urls={len(video['urls'])}"
        )

    print("\n========== AUDIO STREAMS ==========")

    for audio in info_data["audios"]:
        print(
            f"[{audio['index']}] "
            f"id={audio['id']} "
            f"{audio['codec_name']:<8} "
            f"bandwidth={audio['bandwidth']} "
            f"urls={len(audio['urls'])}"
        )


# ============================================================
# STRICT STREAM SELECTION
# ============================================================

def select_video(
    info_data: dict[str, Any],
    codec: str = "avc",
) -> dict[str, Any]:
    """
    Chọn đúng codec trong đúng target_qid đã lấy ở info().

    codec:
        avc
        hevc
        av1

    Không có đúng codec thì raise.
    """

    requested_codec = codec.strip().lower()

    aliases = {
        "h264": "avc",
        "h.264": "avc",
        "avc1": "avc",
        "h265": "hevc",
        "h.265": "hevc",
        "hvc1": "hevc",
        "hev1": "hevc",
    }

    requested_codec = aliases.get(
        requested_codec,
        requested_codec,
    )

    matches = [
        stream
        for stream in info_data["videos"]
        if stream["codec_family"] == requested_codec
    ]

    if not matches:
        available = [
            (
                stream["codec_family"],
                stream["codec"],
            )
            for stream in info_data["videos"]
        ]

        raise RuntimeError(
            f"Không có codec '{requested_codec}' "
            f"ở qid={info_data['target_qid']}. "
            f"Codec hiện có: {available}. "
            f"Đã dừng, không fallback."
        )

    if len(matches) > 1:
        matches.sort(
            key=lambda stream: (
                stream.get("bandwidth", 0) or 0
            ),
            reverse=True,
        )

    return matches[0]


def select_audio(
    info_data: dict[str, Any],
    audio_index: int = 0,
) -> dict[str, Any]:
    audios = info_data["audios"]

    if audio_index < 0 or audio_index >= len(audios):
        raise IndexError(
            f"audio_index={audio_index} không hợp lệ. "
            f"Chỉ có {len(audios)} audio stream."
        )

    return audios[audio_index]



def test_cdn_speed(
    url: str,
    headers: dict[str, str],
    cookies: dict[str, str],
    test_bytes: int = CDN_TEST_BYTES,
) -> CDNTestResult:
    test_headers = dict(headers)
    test_headers["Range"] = f"bytes=0-{test_bytes - 1}"
    test_headers["Accept-Encoding"] = "identity"

    started = time.perf_counter()
    received = 0

    try:
        with requests.get(
            url,
            headers=test_headers,
            cookies=cookies,
            stream=True,
            allow_redirects=True,
            timeout=(
                CDN_CONNECT_TIMEOUT,
                CDN_READ_TIMEOUT,
            ),
        ) as response:
            response.raise_for_status()

            for chunk in response.iter_content(
                chunk_size=256 * 1024
            ):
                if not chunk:
                    continue

                received += len(chunk)

                if received >= test_bytes:
                    break

        elapsed = max(
            time.perf_counter() - started,
            0.001,
        )

        speed_bytes = received / elapsed

        return CDNTestResult(
            url=url,
            host=url_host(url),
            ok=received > 0,
            received_bytes=received,
            elapsed=elapsed,
            speed_bytes=speed_bytes,
            speed_mbps=speed_bytes * 8 / 1_000_000,
            error=None,
        )

    except Exception as exc:
        elapsed = max(
            time.perf_counter() - started,
            0.001,
        )

        return CDNTestResult(
            url=url,
            host=url_host(url),
            ok=False,
            received_bytes=received,
            elapsed=elapsed,
            speed_bytes=0.0,
            speed_mbps=0.0,
            error=str(exc),
        )


def choose_fastest_cdn(
    urls: list[str],
    headers: dict[str, str],
    cookies: dict[str, str],
    max_workers: int = CDN_TEST_WORKERS,
) -> str:
    """
    Benchmark các CDN và trả về đúng một URL nhanh nhất.

    Nếu tất cả CDN lỗi thì raise.
    Sau khi đã chọn URL, bước download không fallback URL khác.
    """

    unique = unique_urls(urls)

    if not unique:
        raise RuntimeError(
            "Stream không có URL CDN"
        )

    if len(unique) == 1:
        print(
            f"[cdn] Chỉ có một CDN: {url_host(unique[0])}"
        )
        return unique[0]

    print("\n========== CDN SPEED TEST ==========")

    results: list[CDNTestResult] = []

    worker_count = min(
        max_workers,
        len(unique),
    )

    with ThreadPoolExecutor(
        max_workers=worker_count
    ) as executor:
        futures = [
            executor.submit(
                test_cdn_speed,
                url,
                headers,
                cookies,
            )
            for url in unique
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result.ok:
                print(
                    f"{result.host:<58} "
                    f"{result.speed_mbps:>9.2f} Mbps"
                )
            else:
                print(
                    f"{result.host:<58} "
                    f"FAILED | {result.error}"
                )

    successful = [
        result
        for result in results
        if result.ok and result.speed_bytes > 0
    ]

    if not successful:
        errors = [
            f"{result.host}: {result.error}"
            for result in results
        ]

        raise RuntimeError(
            "Tất cả CDN benchmark đều lỗi:\n"
            + "\n".join(errors)
        )

    successful.sort(
        key=lambda result: result.speed_bytes,
        reverse=True,
    )

    selected = successful[0]

    print(
        f"Selected CDN: {selected.host} "
        f"({selected.speed_mbps:.2f} Mbps)"
    )

    return selected.url


# ============================================================
# ARIA2 DOWNLOAD
# ============================================================

def aria2_download(
    url: str,
    output_path: str | Path,
    headers: dict[str, str],
    cookies: dict[str, str],
    connections: int,
) -> DownloadResult:
    """
    Tải đúng URL đã chọn.

    aria2 lỗi thì raise ngay.
    Không thử URL khác.
    """

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connections = int(connections)

    if connections < 1:
        raise ValueError(
            "connections phải lớn hơn hoặc bằng 1"
        )

    if connections > MAX_CONNECTIONS:
        raise ValueError(
            f"connections không được vượt quá "
            f"{MAX_CONNECTIONS}"
        )

    output.unlink(missing_ok=True)
    Path(f"{output}.aria2").unlink(missing_ok=True)

    command = [
        "aria2c",

        "-x",
        str(connections),

        "-s",
        str(connections),

        "-k",
        "1M",

        "--max-connection-per-server",
        str(connections),

        "--min-split-size",
        "1M",

        "--continue=false",
        "--file-allocation=none",

        "--max-tries=1",
        "--retry-wait=0",

        "--connect-timeout",
        str(ARIA2_CONNECT_TIMEOUT),

        "--timeout",
        str(ARIA2_TIMEOUT),

        "--lowest-speed-limit",
        ARIA2_LOWEST_SPEED,

        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--remove-control-file=true",

        "--console-log-level=notice",
        "--summary-interval=1",

        "--check-certificate=false",

        f"--user-agent={headers['User-Agent']}",
        f"--referer={headers['Referer']}",

        "--header",
        f"Origin: {headers['Origin']}",

        "--header",
        "Accept-Encoding: identity",
    ]

    cookie_header = cookies_to_header(cookies)

    if cookie_header:
        command.extend([
            "--header",
            f"Cookie: {cookie_header}",
        ])

    command.extend([
        "-d",
        str(output.parent),

        "-o",
        output.name,

        url,
    ])

    host = url_host(url)

    print(
        f"\n[download] CDN={host} "
        f"| connections={connections}"
    )

    try:
        subprocess.run(
            command,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        output.unlink(missing_ok=True)
        Path(f"{output}.aria2").unlink(
            missing_ok=True
        )

        raise RuntimeError(
            f"Tải thất bại từ CDN {host}. "
            f"aria2 exit code={exc.returncode}. "
            f"Đã dừng, không fallback CDN."
        ) from exc

    if not output.exists():
        raise RuntimeError(
            f"aria2 báo hoàn thành nhưng không tạo file: "
            f"{output}"
        )

    if output.stat().st_size <= 0:
        output.unlink(missing_ok=True)

        raise RuntimeError(
            f"File tải từ CDN {host} bị rỗng"
        )

    return DownloadResult(
        path=output,
        url=url,
        host=host,
    )


def download_stream(
    stream: dict[str, Any],
    output_path: str | Path,
    headers: dict[str, str],
    cookies: dict[str, str],
    connections: int,
    test_cdns: bool = True,
) -> DownloadResult:
    urls = stream.get("urls") or []

    if not urls:
        raise RuntimeError(
            "Stream được chọn không có URL"
        )

    if test_cdns:
        selected_url = choose_fastest_cdn(
            urls=urls,
            headers=headers,
            cookies=cookies,
        )
    else:
        selected_url = urls[0]

        print(
            f"[cdn] Không benchmark, dùng base URL: "
            f"{url_host(selected_url)}"
        )

    return aria2_download(
        url=selected_url,
        output_path=output_path,
        headers=headers,
        cookies=cookies,
        connections=connections,
    )


# ============================================================
# FFMPEG
# ============================================================

def ffprobe_info(
    file_path: str | Path,
) -> dict[str, Any]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(path)

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",

            "-show_entries",
            (
                "format=format_name,duration,size,bit_rate:"
                "stream=index,codec_type,codec_name,"
                "width,height,avg_frame_rate,bit_rate,"
                "sample_rate,channels"
            ),

            "-of",
            "json",

            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return json.loads(result.stdout)


def merge_streams(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
) -> None:
    video = Path(video_path)
    audio = Path(audio_path)
    output = Path(output_path)

    if not video.exists():
        raise FileNotFoundError(video)

    if not audio.exists():
        raise FileNotFoundError(audio)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.unlink(missing_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostats",
                "-y",

                "-i",
                str(video),

                "-i",
                str(audio),

                "-map",
                "0:v:0",

                "-map",
                "1:a:0",

                "-c",
                "copy",

                "-movflags",
                "+faststart",

                str(output),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        output.unlink(missing_ok=True)

        raise RuntimeError(
            "FFmpeg ghép video và audio thất bại"
        ) from exc

    if not output.exists() or output.stat().st_size <= 0:
        output.unlink(missing_ok=True)

        raise RuntimeError(
            "FFmpeg không tạo được file đầu ra hợp lệ"
        )


def remux_audio(
    audio_path: str | Path,
    output_path: str | Path,
) -> None:
    audio = Path(audio_path)
    output = Path(output_path)

    if not audio.exists():
        raise FileNotFoundError(audio)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.unlink(missing_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostats",
                "-y",

                "-i",
                str(audio),

                "-map",
                "0:a:0",

                "-c",
                "copy",

                str(output),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        output.unlink(missing_ok=True)

        raise RuntimeError(
            "FFmpeg remux audio thất bại"
        ) from exc

    if not output.exists() or output.stat().st_size <= 0:
        output.unlink(missing_ok=True)

        raise RuntimeError(
            "Không tạo được file audio đầu ra hợp lệ"
        )


# ============================================================
# PUBLIC DOWNLOAD API
# ============================================================

def download(
    info_data: dict[str, Any],
    output_path: str | Path,
    codec: str = "avc",
    audio_index: int = 0,
    temp_dir: str | Path = DEFAULT_TEMP_DIR,
    video_connections: int = VIDEO_CONNECTIONS,
    audio_connections: int = AUDIO_CONNECTIONS,
    test_cdns: bool = True,
    keep_temp: bool = False,
) -> str:
    """
    Tải video và audio đồng thời.

    Chất lượng đã được khóa từ info(target_qid=...).
    Codec được chọn chính xác bằng codec=...
    Không fallback chất lượng, codec hoặc CDN.
    """

    require_dependencies()

    output = Path(output_path)
    temp = Path(temp_dir)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp.mkdir(
        parents=True,
        exist_ok=True,
    )

    video = select_video(
        info_data=info_data,
        codec=codec,
    )

    audio = select_audio(
        info_data=info_data,
        audio_index=audio_index,
    )

    headers = info_data["headers"]
    cookies = info_data["cookies"]

    stem = safe_name(
        f"{info_data['bvid']}_"
        f"p{info_data['page']}_"
        f"q{info_data['target_qid']}_"
        f"{video['codec_family']}"
    )

    video_temp = temp / f"{stem}.video.m4s"
    audio_temp = temp / f"{stem}.audio.m4s"

    video_temp.unlink(missing_ok=True)
    audio_temp.unlink(missing_ok=True)
    Path(f"{video_temp}.aria2").unlink(
        missing_ok=True
    )
    Path(f"{audio_temp}.aria2").unlink(
        missing_ok=True
    )

    print("========== SELECTED ==========")
    print("Title         :", info_data["title"])
    print(
        "Page          :",
        f"P{info_data['page']}",
        info_data["part"],
    )
    print("BVID          :", info_data["bvid"])
    print("CID           :", info_data["cid"])
    print(
        "Quality       :",
        video["quality"],
        f"(qid={video['id']})",
    )
    print(
        "Video codec   :",
        video["codec_name"],
        video["codec"],
    )
    print(
        "Video size    :",
        f"{video['width']}x{video['height']}",
    )
    print(
        "Video bitrate :",
        video["bandwidth"],
    )
    print(
        "Audio codec   :",
        audio["codec_name"],
        audio["codec"],
    )
    print(
        "Audio bitrate :",
        audio["bandwidth"],
    )

    print(
        "\nDownloading video and audio concurrently..."
    )

    executor = ThreadPoolExecutor(max_workers=2)

    try:
        video_future = executor.submit(
            download_stream,
            video,
            video_temp,
            headers,
            cookies,
            video_connections,
            test_cdns,
        )

        audio_future = executor.submit(
            download_stream,
            audio,
            audio_temp,
            headers,
            cookies,
            audio_connections,
            test_cdns,
        )

        try:
            video_result = video_future.result()
            audio_result = audio_future.result()

        except Exception:
            video_future.cancel()
            audio_future.cancel()

            video_temp.unlink(missing_ok=True)
            audio_temp.unlink(missing_ok=True)

            Path(f"{video_temp}.aria2").unlink(
                missing_ok=True
            )
            Path(f"{audio_temp}.aria2").unlink(
                missing_ok=True
            )

            raise

    finally:
        executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

    print("\n========== DOWNLOADED ==========")
    print("Video CDN:", video_result.host)
    print("Audio CDN:", audio_result.host)

    print("\nMerging without re-encoding...")

    try:
        merge_streams(
            video_path=video_temp,
            audio_path=audio_temp,
            output_path=output,
        )
    except Exception:
        output.unlink(missing_ok=True)
        raise

    if not keep_temp:
        video_temp.unlink(missing_ok=True)
        audio_temp.unlink(missing_ok=True)

    probe = ffprobe_info(output)
    format_info = probe.get("format") or {}

    print("\n========== DONE ==========")
    print("Output  :", output)
    print("Duration:", format_info.get("duration"))
    print(
        "Size    :",
        fmt_bytes(
            int(format_info.get("size") or 0)
        ),
    )
    print("Bitrate :", format_info.get("bit_rate"))

    return str(output)


def download_audio(
    info_data: dict[str, Any],
    output_path: str | Path,
    audio_index: int = 0,
    temp_dir: str | Path = DEFAULT_TEMP_DIR,
    connections: int = AUDIO_CONNECTIONS,
    test_cdns: bool = True,
    keep_temp: bool = False,
) -> str:
    """
    Tải riêng audio.

    CDN đã chọn lỗi thì raise ngay, không fallback.
    """

    require_dependencies()

    output = Path(output_path)
    temp = Path(temp_dir)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio = select_audio(
        info_data=info_data,
        audio_index=audio_index,
    )

    stem = safe_name(
        f"{info_data['bvid']}_"
        f"p{info_data['page']}_"
        f"audio_{audio_index}"
    )

    audio_temp = temp / f"{stem}.audio.m4s"

    audio_temp.unlink(missing_ok=True)
    Path(f"{audio_temp}.aria2").unlink(
        missing_ok=True
    )

    result = download_stream(
        stream=audio,
        output_path=audio_temp,
        headers=info_data["headers"],
        cookies=info_data["cookies"],
        connections=connections,
        test_cdns=test_cdns,
    )

    print("Audio CDN:", result.host)

    try:
        remux_audio(
            audio_path=audio_temp,
            output_path=output,
        )
    except Exception:
        output.unlink(missing_ok=True)
        raise

    if not keep_temp:
        audio_temp.unlink(missing_ok=True)

    print("Done:", output)

    return str(output)

