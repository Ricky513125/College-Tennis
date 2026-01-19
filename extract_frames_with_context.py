import json
from pathlib import Path
from PIL import Image

# ========= 配置区 =========
ANNOTATION_FILE = "ncaa_annotations/annotations.json"
FRAMES_ROOT = Path("ncaa_frames")
OUTPUT_ROOT = Path("keyframes_with_context")

JPEG_QUALITY = 90   # 保持清晰度
CONTEXT = 48        # 前后各48帧
# =========================

OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    annotations = json.load(f)

for ann in annotations:
    video_id = ann["video"]
    events = ann["events"]

    src_dir = FRAMES_ROOT / video_id
    dst_video_dir = OUTPUT_ROOT / video_id
    dst_video_dir.mkdir(exist_ok=True, parents=True)

    # 获取视频总帧数
    num_frames = ann.get("num_frames", None)
    if num_frames is None:
        print(f"[WARN] Video {video_id} missing 'num_frames', will try to use max frame available.")
        all_frames = sorted(src_dir.glob("*.jpg"))
        num_frames = len(all_frames)

    for e in events:
        frame_id = e["frame"]
        # 创建以关键帧命名的子文件夹
        dst_subdir = dst_video_dir / f"{frame_id:06d}"
        dst_subdir.mkdir(exist_ok=True, parents=True)

        # 前后各CONTEXT帧
        start_frame = max(0, frame_id - CONTEXT)
        end_frame = min(num_frames - 1, frame_id + CONTEXT)

        for fid in range(start_frame, end_frame + 1):
            frame_name = f"{fid:06d}.jpg"
            src = src_dir / frame_name
            dst = dst_subdir / frame_name

            if not src.exists():
                print(f"[WARN] Missing {src}")
                continue

            img = Image.open(src).convert("RGB")
            img.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True)

        print(f"✅ Extracted frames {start_frame}-{end_frame} for keyframe {frame_id} in video {video_id}")
