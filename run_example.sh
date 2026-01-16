#!/bin/bash
# Example script to run the annotation pipeline with F3ED model

# Set paths
VIDEO_DIR="../ncaa_videos"
MODEL_DIR="./F3Set/f3set-model/f3ed"
OUTPUT_DIR="./ncaa_annotations"
FRAME_DIR="./ncaa_frames"
DATASET="f3set-tennis"

# Run the complete pipeline
python run_full_pipeline.py \
    --video_dir "$VIDEO_DIR" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --frame_dir "$FRAME_DIR" \
    --dataset "$DATASET" \
    --use_f3ed

echo "Pipeline completed! Check $OUTPUT_DIR/annotations.json for results."
