
import re
import subprocess
from pathlib import Path
from http.cookiejar import MozillaCookieJar

import requests


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.bilibili.com/",
}

QUALITY_NAME = {
    127: "8K",
    126: "Dolby",
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


def safe_name(text, max_len=90):
    text = re.sub(r'[\\/:*?"<>|]+', "_", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len] or "video"


def extract_bvid(url):
    m = re.search(r"(BV[0-9A-Za-z]+)", url)
    if not m:
        raise ValueError("Không tìm thấy BV id trong URL")
    return m.group(1)


def load_cookies(cookie_path="cookies.txt"):
    path = Path(cookie_path)

    if not path.exists():
        print(f"{path}: missing, chạy không cookie")
        return {}

    jar = MozillaCookieJar(str(path))
    jar.load(ignore_discard=True, ignore_expires=True)

    cookies = {}
    for c in jar:
        domain = c.domain.lower()
        if (
            "bilibili.com" in domain
            or "biliapi.net" in domain
            or "biliapi.com" in domain
        ):
            cookies[c.name] = c.value

    return cookies


def cookies_to_header(cookies):
    return "; ".join(f"{k}={v}" for k, v in cookies.items())


def request_json(url, params, headers, cookies, timeout=30):
    r = requests.get(
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        timeout=timeout,
    )
    r.raise_for_status()

    data = r.json()

    if data.get("code") != 0:
        raise RuntimeError(data)

    return data


def quality_name(qid):
    return QUALITY_NAME.get(qid, str(qid))


def codec_name(codec):
    c = (codec or "").lower()

    if c.startswith("avc1"):
        return "H.264"
    if c.startswith(("hev1", "hvc1")):
        return "H.265"
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


def get_view(video_url, headers=None, cookies=None):
    headers = headers or DEFAULT_HEADERS
    cookies = cookies or {}

    bvid = extract_bvid(video_url)

    data = request_json(
        "https://api.bilibili.com/x/web-interface/view",
        {"bvid": bvid},
        headers,
        cookies,
    )

    return data["data"]


def get_dash(bvid, cid, headers=None, cookies=None, target_qid=64):
    headers = headers or DEFAULT_HEADERS
    cookies = cookies or {}

    data = request_json(
        "https://api.bilibili.com/x/player/playurl",
        {
            "bvid": bvid,
            "cid": cid,
            "qn": target_qid,
            "fnval": 4048,
            "fourk": 0,
        },
        headers,
        cookies,
    )["data"]

    dash = data.get("dash")
    if not dash:
        raise RuntimeError(f"No DASH stream: bvid={bvid}, cid={cid}")

    videos = dash.get("video") or []
    audios = dash.get("audio") or []

    if not videos:
        raise RuntimeError(f"No video stream: bvid={bvid}, cid={cid}")
    if not audios:
        raise RuntimeError(f"No audio stream: bvid={bvid}, cid={cid}")

    return data, videos, audios


def collect_items(view_data):
    root_bvid = view_data.get("bvid")
    root_title = view_data.get("title") or "video"

    season = view_data.get("ugc_season")

    if season and season.get("sections"):
        items = []
        idx = 1

        for section in season.get("sections") or []:
            section_title = section.get("title") or ""

            for ep in section.get("episodes") or []:
                page = ep.get("page") or {}

                bvid = ep.get("bvid") or root_bvid
                cid = ep.get("cid") or page.get("cid")
                title = (
                    ep.get("title")
                    or page.get("part")
                    or ep.get("arc", {}).get("title")
                    or f"episode_{idx}"
                )

                if bvid and cid:
                    items.append({
                        "index": idx,
                        "bvid": bvid,
                        "cid": cid,
                        "title": title,
                        "section": section_title,
                        "source": "ugc_season",
                    })
                    idx += 1

        if items:
            return items

    pages = view_data.get("pages") or []

    if len(pages) > 1:
        return [
            {
                "index": p.get("page") or i + 1,
                "bvid": root_bvid,
                "cid": p.get("cid"),
                "title": p.get("part") or f"P{i + 1}",
                "section": "",
                "source": "pages",
            }
            for i, p in enumerate(pages)
            if p.get("cid")
        ]

    cid = view_data.get("cid") or (pages[0].get("cid") if pages else None)

    return [{
        "index": 1,
        "bvid": root_bvid,
        "cid": cid,
        "title": root_title,
        "section": "",
        "source": "single",
    }]


def get_bilibili_playlist_items(video_url, cookie_path="cookies.txt"):
    """
    Chỉ lấy danh sách item, chưa download.
    Trả về list dict:
    {
        index, bvid, cid, title, section, source
    }
    """
    headers = DEFAULT_HEADERS.copy()
    cookies = load_cookies(cookie_path)

    view_data = get_view(video_url, headers=headers, cookies=cookies)
    items = collect_items(view_data)

    print("title:", view_data.get("title"))
    print("bvid :", view_data.get("bvid"))
    print("cid  :", view_data.get("cid"))
    print("source:", items[0]["source"] if items else "?")
    print("items:", len(items))

    for item in items:
        print(
            f'{item["index"]:>3}. '
            f'[{item["source"]}] '
            f'{item["title"]} '
            f'(bvid={item["bvid"]}, cid={item["cid"]})'
        )

    return items


def select_items(items, selection):
    """
    Chọn item bằng string:
    - "all"
    - "1"
    - "1,3,5"
    - "1-5"
    - "1-3,8,10-12"
    Hoặc truyền list index: [1, 2, 5]
    """
    if selection is None or selection == "all":
        return items

    wanted = set()

    if isinstance(selection, str):
        parts = selection.replace(" ", "").split(",")

        for part in parts:
            if not part:
                continue

            if "-" in part:
                a, b = part.split("-", 1)
                a, b = int(a), int(b)
                wanted.update(range(min(a, b), max(a, b) + 1))
            else:
                wanted.add(int(part))

    else:
        wanted = set(int(x) for x in selection)

    return [item for item in items if int(item["index"]) in wanted]


def pick_video(videos, target_qid=64):
    preferred = [
        (target_qid, "avc1"),
        (target_qid, "hev1"),
        (target_qid, "hvc1"),
        (80, "avc1"),
        (32, "avc1"),
        (80, "hev1"),
        (80, "hvc1"),
        (32, "hev1"),
        (32, "hvc1"),
        (16, "avc1"),
    ]

    for qid, prefix in preferred:
        found = [
            v for v in videos
            if v.get("id") == qid
            and (v.get("codecs") or "").lower().startswith(prefix)
        ]

        if found:
            return max(found, key=lambda x: x.get("bandwidth", 0) or 0)

    non_av1 = [
        v for v in videos
        if not (v.get("codecs") or "").lower().startswith("av01")
    ]

    pool = non_av1 or videos

    return max(
        pool,
        key=lambda v: (
            codec_rank(v.get("codecs")),
            v.get("id", 0) or 0,
            v.get("bandwidth", 0) or 0,
        ),
    )


def pick_audio(audios):
    return max(audios, key=lambda a: a.get("bandwidth", 0) or 0)


def aria2_download(
    url,
    output_path,
    headers=None,
    cookies=None,
    connections=16,
    quiet=False,
):
    headers = headers or DEFAULT_HEADERS
    cookies = cookies or {}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ck = cookies_to_header(cookies)

    cmd = [
        "aria2c",
        "-x", str(connections),
        "-s", str(connections),
        "--split", str(connections),
        "--max-connection-per-server", str(connections),
        "--min-split-size=1M",
        "-j", "1",
        "--continue=true",
        "--file-allocation=none",
        "--retry-wait=5",
        "--max-tries=8",
        "--timeout=30",
        "--connect-timeout=15",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--console-log-level=warn" if quiet else "--console-log-level=notice",
        "--summary-interval=0" if quiet else "--summary-interval=5",
        f"--user-agent={headers.get('User-Agent', 'Mozilla/5.0')}",
        f"--referer={headers.get('Referer', 'https://www.bilibili.com/')}",
        "-d", str(output_path.parent),
        "-o", output_path.name,
        url,
    ]

    if ck:
        cmd += ["--header", f"Cookie: {ck}"]

    subprocess.run(cmd, check=True)


def merge_av(video_file, audio_file, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-nostats",
            "-y",
            "-i", str(video_file),
            "-i", str(audio_file),
            "-c", "copy",
            str(output_file),
        ],
        check=True,
    )


def download_bilibili_items(
    items,
    download_dir="downloads",
    basename="video",
    cookie_path="cookies.txt",
    target_qid=64,
    connections=16,
    selection="all",
    skip_existing=True,
    quiet=False,
    delete_temp=True,
    single_output_path=None,
):
    """
    Download từ list items đã lấy trước.

    selection:
        "all"
        "1"
        "1,3,5"
        "1-5"
        "1-3,8,10-12"
        [1, 2, 5]

    single_output_path:
        chỉ dùng khi tải đúng 1 item và muốn output ra path cụ thể.
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    selected_items = select_items(items, selection)

    headers = DEFAULT_HEADERS.copy()
    cookies = load_cookies(cookie_path)

    print("selected items:", len(selected_items))
    print("cookies:", len(cookies), "| SESSDATA:", "OK" if "SESSDATA" in cookies else "missing")

    downloaded = []

    for item in selected_items:
        idx = item["index"]
        title = safe_name(item["title"])

        if len(selected_items) == 1 and single_output_path:
            output = Path(single_output_path)
        else:
            output = download_dir / f"{basename}_{idx:03d}_{title}.mp4"

        video_tmp = download_dir / f"{basename}_{idx:03d}.video.m4s"
        audio_tmp = download_dir / f"{basename}_{idx:03d}.audio.m4s"

        print("\n" + "=" * 72)
        print(f"[{idx}] {item['title']}")
        print("bvid:", item["bvid"])
        print("cid :", item["cid"])
        print("out :", output)

        if skip_existing and output.exists() and output.stat().st_size > 0:
            print("skip existing")
            downloaded.append(output)
            continue

        try:
            _, videos, audios = get_dash(
                item["bvid"],
                item["cid"],
                headers=headers,
                cookies=cookies,
                target_qid=target_qid,
            )

            video = pick_video(videos, target_qid=target_qid)
            audio = pick_audio(audios)

            video_url = video.get("baseUrl") or video.get("base_url")
            audio_url = audio.get("baseUrl") or audio.get("base_url")

            if not video_url:
                raise RuntimeError("Không lấy được video_url")
            if not audio_url:
                raise RuntimeError("Không lấy được audio_url")

            print(
                "selected:",
                quality_name(video.get("id")),
                codec_name(video.get("codecs")),
                f"{video.get('width')}x{video.get('height')}",
                "| audio",
                codec_name(audio.get("codecs")),
            )

            aria2_download(
                video_url,
                video_tmp,
                headers=headers,
                cookies=cookies,
                connections=connections,
                quiet=quiet,
            )

            aria2_download(
                audio_url,
                audio_tmp,
                headers=headers,
                cookies=cookies,
                connections=connections,
                quiet=quiet,
            )

            merge_av(video_tmp, audio_tmp, output)

            if delete_temp:
                video_tmp.unlink(missing_ok=True)
                audio_tmp.unlink(missing_ok=True)

            print("done:", output)
            downloaded.append(output)

        except Exception as e:
            print("FAILED:", item)
            print("ERROR :", repr(e))

    return downloaded


def download_bilibili(
    video_url,
    download_dir="downloads",
    basename="video",
    cookie_path="cookies.txt",
    target_qid=64,
    connections=16,
    selection="all",
    skip_existing=True,
    quiet=False,
    delete_temp=True,
    single_output_path=None,
):
    """
    Shortcut:
    tự get playlist items rồi download theo selection.
    """
    items = get_bilibili_playlist_items(video_url, cookie_path=cookie_path)

    return download_bilibili_items(
        items,
        download_dir=download_dir,
        basename=basename,
        cookie_path=cookie_path,
        target_qid=target_qid,
        connections=connections,
        selection=selection,
        skip_existing=skip_existing,
        quiet=quiet,
        delete_temp=delete_temp,
        single_output_path=single_output_path,
    )