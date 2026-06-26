
### tts
``` python
from tts_srt import TTSConfig, generate_tts_from_srt

config = TTSConfig(
    api_base="https://your-tts-api.com",
    voice="your_voice",
    resource_id="your_resource_id",
    rate="0",
)

results = generate_tts_from_srt(
    srt_path="input.srt",
    output_dir="tts_audio",
    config=config,
)
```

### download bilibili
``` python
items = get_bilibili_playlist_items(VIDEO_URL, cookie_path="cookies.txt")

import ipywidgets as widgets
from IPython.display import display, clear_output

item_checks = []

for item in items:
    label = f'{item["index"]:03d} | {item["title"]}'
    cb = widgets.Checkbox(
        value=True,
        description=label,
        indent=False,
        layout=widgets.Layout(width="100%")
    )
    item_checks.append((cb, item))

check_all_btn = widgets.Button(description="Check all")
uncheck_all_btn = widgets.Button(description="Uncheck all")
show_btn = widgets.Button(description="Show selected")

out = widgets.Output()

def check_all(_):
    for cb, _item in item_checks:
        cb.value = True

def uncheck_all(_):
    for cb, _item in item_checks:
        cb.value = False

def show_selected(_):
    selected_items = [item for cb, item in item_checks if cb.value]
    with out:
        clear_output()
        print("Selected:", len(selected_items))
        for item in selected_items:
            print(f'{item["index"]:03d} | {item["title"]}')

check_all_btn.on_click(check_all)
uncheck_all_btn.on_click(uncheck_all)
show_btn.on_click(show_selected)

display(widgets.HBox([check_all_btn, uncheck_all_btn, show_btn]))
display(widgets.VBox([cb for cb, _item in item_checks]))
display(out)

from bili_downloader import download_bilibili_items

selected_items = [item for cb, item in item_checks if cb.value]

files = download_bilibili_items(
    selected_items,
    download_dir=DOWNLOAD_DIR,
    basename=VIDEO_BASENAME,
    cookie_path="cookies.txt",
    target_qid=64,
    connections=16,
    selection="all",
    skip_existing=True,
    quiet=False,
    delete_temp=True,
    single_output_path=VIDEO_PATH,
)

print("Downloaded:")
for f in files:
    print(f)


```


### Trans
``` python
from pathlib import Path
from srt_translator import translate_srt

INPUT_SRT = Path("/content/output.srt")
VI_SRT = Path("/content/output_vi.srt")
WORK_DIR = Path("/content/work")
WORK_DIR.mkdir(parents=True, exist_ok=True)

out = translate_srt(
    input_srt=INPUT_SRT,
    output_srt=VI_SRT,
    system_prompt=SYSTEM_PROMPT_BOX.value,
    repair_prompt=REPAIR_PROMPT_BOX.value,
    user_prompt=USER_PROMPT_BOX.value,
    model="gpt-4.1-mini",
    api_key=None,
    base_url="https://api.shopaikey.com/v1",
    cache_path=WORK_DIR / "translation_cache_vi.json",
    resume=True,
    repair_cjk=True,
    batch_size=12,
    max_chars=2400,
    max_workers=8,
    temperature=0.1,
)

print(out)
```


### Trans gpt local

``` python
translate_srt_via_socket(
        input_srt=args.input_srt,
        output_srt=args.output_srt,
        server=args.server,
        batch_size=args.batch_size,
        poll_interval=args.poll_interval,
        job_timeout=args.job_timeout,
        retry=args.retry,
    )
```