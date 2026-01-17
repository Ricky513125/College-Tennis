@echo off
REM Batch process 6 NCAA videos for skeleton annotation using extracted frames (Windows)

REM Configuration
set FRAME_DIR=.\ncaa_frames
set MODEL_PATH=.\deep-high-resolution-net.pytorch\models\pose_hrnet_w48_384x288.pth
set OUTPUT_DIR=.\ncaa_skeleton_annotations
set METADATA=.\ncaa_annotations\ncaa_videos_metadata.json

REM Video IDs
set VIDEOS=6VSmpCSgY7M Avendano__UL__Vs__Penzlin__LSU_ dwPey52i1LE Hoole__SC__vs__Dong__LSU_mp4 IohTeru65U4 Lc9MSf6vHxU

echo ============================================================
echo Batch Skeleton Annotation for 6 NCAA Videos
echo ============================================================
echo Frame directory: %FRAME_DIR%
echo Model: %MODEL_PATH%
echo Output directory: %OUTPUT_DIR%
echo ============================================================

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Process each video
for %%v in (%VIDEOS%) do (
    echo.
    echo Processing: %%v
    echo ----------------------------------------
    
    python generate_skeleton_annotations.py ^
        "%FRAME_DIR%\%%v" ^
        --video_id "%%v" ^
        --model_path "%MODEL_PATH%" ^
        --output "%OUTPUT_DIR%\%%v_skeleton.json" ^
        --skip_frames 1 ^
        --detection_threshold 0.9 ^
        --use_frames
    
    if !errorlevel! equ 0 (
        echo Successfully processed %%v
    ) else (
        echo Failed to process %%v
    )
)

echo.
echo ============================================================
echo Batch processing completed!
echo ============================================================
echo Output files saved in: %OUTPUT_DIR%
pause
