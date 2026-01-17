#!/bin/bash
# Batch process 6 NCAA videos for optical flow generation using RAFT

# Configuration
FRAME_DIR="./ncaa_frames"
MODEL_PATH="./RAFT/models/raft-small.pth"
OUTPUT_DIR="./ncaa_optical_flow"
METADATA="./ncaa_annotations/ncaa_videos_metadata.json"

# Video IDs
VIDEOS=(
    "6VSmpCSgY7M"
    "Avendano__UL__Vs__Penzlin__LSU_"
    "dwPey52i1LE"
    "Hoole__SC__vs__Dong__LSU_mp4"
    "IohTeru65U4"
    "Lc9MSf6vHxU"
)

echo "============================================================"
echo "Batch Optical Flow Generation for 6 NCAA Videos (RAFT)"
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
    
    python generate_optical_flow.py \
        "$FRAME_DIR/$video_id" \
        --video_id "$video_id" \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --skip_frames 1 \
        --small \
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
