# NCAA Video Annotation using F3-Set

This directory contains scripts to process NCAA tennis videos and generate annotations using F3-set models.

## Overview

The pipeline consists of three main steps:
1. **Frame Extraction**: Extract frames from videos in `ncaa_videos` folder
2. **Metadata Creation**: Create JSON metadata files for the videos
3. **Annotation Generation**: Run inference using F3-set models to generate annotations

## Prerequisites

1. **F3-Set Model**: You need a trained F3-set model checkpoint. The model directory should contain:
   - `checkpoint_XXX.pt`: Model checkpoint file
   - `config.json`: Model configuration file
   - `loss.json` (optional): Training loss history for selecting best epoch

2. **Video Directory**: Videos should be in a directory (default: `../ncaa_videos`)

3. **Dependencies**: Install required packages from `F3Set/requirements.txt`

## Usage

### Option 1: Run Complete Pipeline (Recommended)

Run the complete pipeline in one command:

```bash
python run_full_pipeline.py \
    --video_dir ../ncaa_videos \
    --model_dir /path/to/model/directory \
    --output_dir ./ncaa_annotations \
    --frame_dir ./ncaa_frames \
    --dataset f3set-tennis
```

**Arguments:**
- `--video_dir`: Directory containing input videos (default: `../ncaa_videos`)
- `--model_dir`: **Required**. Directory containing trained F3-set model
- `--output_dir`: Directory to save outputs (default: `./ncaa_annotations`)
- `--frame_dir`: Directory to save extracted frames (default: `./ncaa_frames`)
- `--dataset`: Dataset name for loading classes (default: `f3set-tennis`)
- `--use_f3ed`: Use F3ED model instead of baseline
- `--skip_extraction`: Skip frame extraction (use existing frames)

### Option 2: Run Steps Separately

#### Step 1: Extract Frames and Create Metadata

```bash
python process_ncaa_videos.py \
    --video_dir ../ncaa_videos \
    --output_dir ./ncaa_annotations \
    --frame_dir ./ncaa_frames
```

This will:
- Extract frames from all videos in `ncaa_videos`
- Save frames to `ncaa_frames` directory
- Create `ncaa_videos_metadata.json` with video metadata

#### Step 2: Generate Annotations

```bash
python generate_annotations.py \
    --metadata ./ncaa_annotations/ncaa_videos_metadata.json \
    --frame_dir ./ncaa_frames \
    --model_dir /path/to/model/directory \
    --output_dir ./ncaa_annotations \
    --dataset f3set-tennis
```

This will:
- Load the trained model
- Run inference on all videos
- Generate `annotations.json` with detected events

## Output Format

The generated `annotations.json` file follows the F3-set format:

```json
[
  {
    "fps": 29.0,
    "height": 720,
    "width": 1280,
    "num_frames": 410,
    "video": "video_name",
    "far_name": "Unknown",
    "far_hand": "RH",
    "far_set": 0,
    "far_game": 0,
    "far_point": 0,
    "near_name": "Unknown",
    "near_hand": "RH",
    "near_set": 0,
    "near_game": 0,
    "near_point": 0,
    "events": [
      {
        "frame": 73,
        "label": "far_ad_serve_-_-_B_-_in",
        "score": 0.95
      },
      ...
    ]
  },
  ...
]
```

Each event contains:
- `frame`: Frame number where the event occurs
- `label`: Event label (from `events.txt`)
- `score`: Confidence score for the prediction

## Directory Structure

After running the pipeline, you should have:

```
College_tennis/
├── ncaa_frames/              # Extracted frames
│   ├── video1/
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   ├── ...
│   │   └── fps.txt
│   └── video2/
│       └── ...
├── ncaa_annotations/         # Output annotations
│   ├── ncaa_videos_metadata.json
│   └── annotations.json
├── process_ncaa_videos.py
├── generate_annotations.py
└── run_full_pipeline.py
```

## Notes

1. **Model Requirements**: Make sure your model checkpoint and config.json are compatible. The script will automatically select the best epoch from `loss.json` if available, otherwise it uses the last checkpoint.

2. **Frame Extraction**: Frames are extracted at 224px height (maintaining aspect ratio). This matches the default F3-set configuration.

3. **Video Formats**: Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv` (case-insensitive)

4. **GPU**: The scripts will use GPU if available (CUDA). Make sure PyTorch is installed with CUDA support if you want GPU acceleration.

5. **Memory**: Processing large videos may require significant memory. Consider processing videos in batches if you encounter memory issues.

## Troubleshooting

### Model not found
- Ensure the model directory contains `checkpoint_XXX.pt` and `config.json`
- Check that the checkpoint file name matches the epoch number

### Frame extraction fails
- Verify video files are not corrupted
- Check that OpenCV can read the video format
- Ensure sufficient disk space for frame storage

### Import errors
- Make sure you're running from the `College_tennis` directory
- Verify that `F3Set` directory exists and contains the required modules
- Install dependencies: `pip install -r F3Set/requirements.txt`

### CUDA out of memory
- Reduce batch size in `generate_annotations.py` (modify `INFERENCE_BATCH_SIZE`)
- Process videos one at a time
- Use CPU mode if GPU memory is limited

## Example

### Using F3ED Model (Recommended)

```bash
# Process all videos in ncaa_videos and generate annotations using F3ED model
python run_full_pipeline.py \
    --video_dir ../ncaa_videos \
    --model_dir ./F3Set/f3set-model/f3ed \
    --output_dir ./ncaa_annotations \
    --frame_dir ./ncaa_frames \
    --dataset f3set-tennis \
    --use_f3ed
```

### Using Baseline Model

```bash
# Process all videos using baseline model
python run_full_pipeline.py \
    --video_dir ../ncaa_videos \
    --model_dir /path/to/baseline/model \
    --output_dir ./ncaa_annotations \
    --frame_dir ./ncaa_frames \
    --dataset f3set-tennis
```

**Note**: The model directory should contain:
- `checkpoint_XXX.pt` - Model checkpoint file
- `config.json` - Model configuration
- `loss.json` (optional) - Training loss history

This will process all videos and create annotation files in `./ncaa_annotations/annotations.json`.



python generate_annotations.py \
    --metadata ./ncaa_annotations/ncaa_videos_metadata.json \
    --frame_dir ./ncaa_frames \
    --model_dir ./F3Set/f3set-model/f3ed \
    --output_dir ./ncaa_annotations \
    --dataset f3set-tennis
    --use_f3ed