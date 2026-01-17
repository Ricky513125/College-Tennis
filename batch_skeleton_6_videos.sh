#!/bin/bash
# Batch process 6 NCAA videos for skeleton annotation using extracted frames

# Configuration
FRAME_DIR="./ncaa_frames"
MODEL_PATH="./deep-high-resolution-net.pytorch/models/pose_hrnet_w48_384x288.pth"
OUTPUT_DIR="./ncaa_skeleton_annotations"
METADATA="./ncaa_annotations/ncaa_videos_metadata.json"

# Video IDs (from your previous processing)
VIDEOS=(
    "6VSmpCSgY7M"
    "Avendano__UL__Vs__Penzlin__LSU_"
    "dwPey52i1LE"
    "Hoole__SC__vs__Dong__LSU_mp4"
    "IohTeru65U4"
    "Lc9MSf6vHxU"
)

echo "============================================================"
echo "Batch Skeleton Annotation for 6 NCAA Videos"
echo "============================================================"
echo "Frame directory: $FRAME_DIR"
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each video
for video_id in "${VIDEOS[@]}"; do
    echo ""
    echo "Processing: $video_id"
    echo "----------------------------------------"
    
    python generate_skeleton_annotations.py \
        "$FRAME_DIR/$video_id" \
        --video_id "$video_id" \
        --model_path "$MODEL_PATH" \
        --config "./deep-high-resolution-net.pytorch/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml" \
        --output "$OUTPUT_DIR/${video_id}_skeleton.json" \
        --skip_frames 1 \
        --detection_threshold 0.9 \
        --use_frames
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $video_id"
    else
        echo "✗ Failed to process $video_id"
    fi
done

echo ""
echo "============================================================"
echo "Batch processing completed!"
echo "============================================================"
echo "Output files saved in: $OUTPUT_DIR"
