# 输入数据格式说明

## 🎬 关键结论

**输入是已经提取好的帧图片，不是视频文件！**

## 📂 数据目录结构

### 正确的输入格式

```
frame_dir/
├── video_id_1/
│   ├── rally_0540_0550/
│   │   ├── 000001.jpg    ← 第1帧
│   │   ├── 000002.jpg    ← 第2帧
│   │   ├── 000003.jpg
│   │   └── ...
│   └── rally_0605_0610/
│       ├── 000001.jpg
│       ├── 000002.jpg
│       └── ...
├── video_id_2/
│   └── rally_0xxx_0xxx/
│       └── ...
└── ...

flow_dir/ (可选 - 光流数据)
├── video_id_1/
│   └── rally_0540_0550/
│       ├── 000001.npy    ← 光流文件是 .npy 格式
│       ├── 000002.npy
│       └── ...
└── ...

pose_dir/ (可选 - 骨架数据)
├── video_id_1_rally_0540_0550_skeleton.json
├── video_id_1_rally_0605_0610_skeleton.json
└── ...
```

## 🔍 代码证据

### 1. FrameReader 读取图片

从 `MD-FED/dataset/input_process.py`:

```python
class FrameReader:
    IMG_NAME = '{:06d}.jpg'  # 图片命名格式：6位数字.jpg
    
    def read_frame(self, frame_path, is_flow=False):
        if is_flow:
            # 光流文件是 .npy 格式
            flow = np.load(frame_path)
            return flow
        else:
            # RGB 帧是 .jpg 图片
            img = torchvision.io.read_image(frame_path).float() / 255
            return img
```

### 2. 图片路径构造

```python
# 从 load_frames 方法
for i in range(start, end, stride):
    # 构造图片路径
    frame_path = os.path.join(
        self._frame_dir,      # 如: /path/to/frames
        video_name,           # 如: 6VSmpCSgY7M/rally_0540_0550
        self.IMG_NAME.format(i)  # 如: 000039.jpg
    )
    # 读取图片
    img = self.read_frame(frame_path)
```

完整路径示例：
```
/mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally/6VSmpCSgY7M/rally_0540_0550/000039.jpg
```

## ⚙️ 如何准备输入数据

### 步骤 1: 从视频提取帧

如果你有视频文件，需要先提取帧：

```bash
# 使用 process_manual_rallies.py
python process_manual_rallies.py \
    --video_dir /path/to/videos \
    --annotations manual_annotations.json \
    --output_dir /path/to/frames \
    --extract_frames
```

或者使用 FFmpeg 手动提取：

```bash
# 对单个视频提取帧
mkdir -p frames/video_id/rally_name
ffmpeg -i video.mp4 \
    -vf "fps=60" \
    frames/video_id/rally_name/%06d.jpg

# 参数说明：
# -vf "fps=60": 以60fps提取（根据视频实际fps调整）
# %06d.jpg: 输出格式为6位数字，从000001.jpg开始
```

### 步骤 2: 验证数据结构

```bash
# 检查帧是否正确提取
python check_frames.py \
    --frame_dir /path/to/frames \
    --annotations manual_annotations.json
```

示例输出：
```
✓ 6VSmpCSgY7M/rally_0540_0550: 599 frames
✓ 6VSmpCSgY7M/rally_0605_0610: 299 frames
✓ Avendano__UL__Vs__Penzlin__LSU_/rally_0600_0620: 1200 frames
...
```

## 📊 数据格式要求

### RGB 帧 (必需)

- **格式**: JPG 图片
- **命名**: `{:06d}.jpg` (000001.jpg, 000002.jpg, ...)
- **尺寸**: 任意（训练时会自动resize和crop）
- **路径**: `frame_dir/video_name/frame_number.jpg`

### 光流 (可选)

- **格式**: NPY 文件 (numpy array)
- **命名**: `{:06d}.npy`
- **形状**: `(2, H, W)` 或 `(H, W, 2)`
- **内容**: 光流的 x 和 y 分量
- **路径**: `flow_dir/video_name/frame_number.npy`

生成光流：
```bash
python generate_optical_flow.py \
    --frame_dir /path/to/frames \
    --output_dir /path/to/flow \
    --model_path RAFT/models/raft-things.pth
```

### 骨架 (可选)

- **格式**: JSON 文件
- **命名**: `video_id_rally_name_skeleton.json`
- **内容**: 每帧的人体关键点
- **路径**: `pose_dir/video_id_rally_name_skeleton.json`

生成骨架：
```bash
python generate_skeleton_annotations.py \
    --frame_dir /path/to/frames \
    --output_dir /path/to/skeletons \
    --model_path deep-high-resolution-net.pytorch/models/...
```

## 🚀 训练时的数据加载流程

### 1. VTN 训练

```bash
python train_vtn_comparison.py \
    --frame_dir /path/to/frames     # 指向帧图片目录
    --manual_annotations manual_annotations.json
```

**数据加载过程**:
1. 读取 `manual_annotations.json` 获取视频列表和事件标注
2. 根据 `video_name`（如 `6VSmpCSgY7M/rally_0540_0550`）定位帧目录
3. 从指定帧范围读取 JPG 图片
4. 应用数据增强（crop, flip, normalize）
5. 组成 clip（96帧）输入模型

### 2. MD-FED Stage 3 训练

```bash
python few_shot_learning_stage3.py \
    --frame_dir /path/to/frames     # RGB 帧
    --flow_dir /path/to/flow        # 光流（可选）
    --manual_annotations manual_annotations.json
```

**数据加载过程**:
1. 同时加载 RGB 帧 + 光流
2. 如果提供 `pose_dir`，还会加载骨架数据
3. 多模态数据对齐并输入模型

## 📝 manual_annotations.json 格式

```json
[
  {
    "video": "6VSmpCSgY7M/rally_0540_0550",  ← 对应帧目录路径
    "num_frames": 599,                        ← 帧数
    "fps": 60.02,
    "events": [
      {
        "frame": 39,                          ← 第39帧 (000039.jpg)
        "label": "far_middle_serve_-_-_W_-_in"
      }
    ]
  }
]
```

## ⚠️ 常见问题

### Q1: 为什么不直接读取视频？

**A**: 
1. **性能**: 从视频解码帧非常慢，预先提取帧可以加速训练
2. **灵活性**: 可以方便地进行数据增强和预处理
3. **多模态**: 方便添加光流、骨架等额外模态
4. **一致性**: 确保训练和测试使用完全相同的帧

### Q2: 帧编号从0还是1开始？

**A**: 从 **1** 开始！
- 第1帧: `000001.jpg`
- 第2帧: `000002.jpg`
- 第39帧: `000039.jpg`

在 `manual_annotations.json` 中，`frame: 39` 对应 `000039.jpg`

### Q3: 不同视频可以有不同的帧率吗？

**A**: 可以！
- 每个视频在 annotations 中都有自己的 `fps` 字段
- 模型使用帧数而不是时间戳，所以不同帧率没问题
- 但建议保持一致以获得更好的性能

### Q4: 如果某些帧缺失怎么办？

**A**: 
- `load_frames` 方法会自动处理缺失帧
- 使用 padding 或复制相邻帧来填充
- 建议确保所有帧都存在以避免问题

### Q5: 图片可以是PNG格式吗？

**A**: 
- 代码中硬编码为 `.jpg` 格式
- 如果要用 PNG，需要修改 `IMG_NAME = '{:06d}.png'`
- 建议使用 JPG 以节省存储空间

## 🔧 数据验证脚本

创建一个简单的验证脚本：

```python
import os
import json

def verify_frames(frame_dir, annotations_file):
    """验证所有标注的视频是否有对应的帧"""
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    for item in annotations:
        video_name = item['video']
        num_frames = item['num_frames']
        
        # 检查目录是否存在
        video_path = os.path.join(frame_dir, video_name)
        if not os.path.exists(video_path):
            print(f"❌ Missing directory: {video_path}")
            continue
        
        # 检查帧文件
        missing_frames = []
        for i in range(1, num_frames + 1):
            frame_path = os.path.join(video_path, f"{i:06d}.jpg")
            if not os.path.exists(frame_path):
                missing_frames.append(i)
        
        if missing_frames:
            print(f"❌ {video_name}: Missing frames {missing_frames[:10]}...")
        else:
            print(f"✓ {video_name}: All {num_frames} frames present")

# 使用
verify_frames(
    "/mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally",
    "manual_annotations.json"
)
```

## 📈 存储空间估算

对于你的数据集（85个视频，~42,836帧）：

```
单帧大小（1920×1080 JPG）: ~200-500KB
总帧数: 42,836
估计总大小: 42,836 × 300KB ≈ 12-13 GB

光流（.npy）:
单个光流文件: ~1-2 MB
总大小: 42,836 × 1.5MB ≈ 60-65 GB

骨架（.json）:
单个skeleton文件: ~5-10 MB
总大小: 85 × 7.5MB ≈ 640 MB
```

## 🎯 总结

| 输入类型 | 格式 | 必需/可选 | 用途 |
|---------|------|-----------|------|
| **RGB 帧** | JPG 图片 | ✅ 必需 | 视觉特征提取 |
| **光流** | NPY 文件 | ⭕ 可选 | 运动信息（MD-FED使用）|
| **骨架** | JSON 文件 | ⭕ 可选 | 人体姿态（MD-FED使用）|

**VTN 只需要 RGB 帧！** 🎯

---

希望这个说明清楚了！如果还有疑问，随时问我。
