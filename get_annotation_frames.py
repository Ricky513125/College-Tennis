import json
from pathlib import Path
from PIL import Image

# ========= 配置区 =========
ANNOTATION_FILE = "ncaa_annotations/annotations.json"
FRAMES_ROOT = Path("ncaa_frames")
OUTPUT_ROOT = Path("keyframes_compressed")

JPEG_QUALITY = 75
MAX_SIZE = 960
# =========================

OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    annotations = json.load(f)

def resize_keep_ratio(img, max_size):
    w, h = img.size
    scale = max_size / max(w, h)
    if scale >= 1:
        return img
    return img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

for ann in annotations:
    video_id = ann["video"]
    events = ann["events"]

    src_dir = FRAMES_ROOT / video_id
    dst_dir = OUTPUT_ROOT / video_id
    dst_dir.mkdir(exist_ok=True, parents=True)

    for e in events:
        frame_id = e["frame"]
        frame_name = f"{frame_id:06d}.jpg"  # 对应你的帧名
        src = src_dir / frame_name
        dst = dst_dir / frame_name

        if not src.exists():
            print(f"[WARN] Missing {src}")
            continue

        img = Image.open(src).convert("RGB")
        img = resize_keep_ratio(img, MAX_SIZE)
        img.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True)

    print(f"✅ Extracted {len(events)} keyframes for video {video_id}")
