# MD-FED

Recognizing fine-grained sports events with precise temporal localization is essential for sports analytics but remains challenging due to factors such as rapid succession, motion blur, and subtle visual differences, often requiring large amounts of labeled data. Existing end-to-end visual and skeleton-based models struggle in few-shot conditions due to their reliance on pixel- or pose-based inputs alone. We propose Multimodal Distillation for Few-Shot Fine-Grained Event Detection (MD-FED), a weakly supervised approach that leverages skeleton-based features as a teacher to guide student networks in learning visual representations from RGB and optical flow. MD-FED employs a three-stage training strategy: skeleton-based pretraining, multimodal distillation, and few-shot fine-tuning. Evaluations on four sports datasets—F$^3$Set, ShuttleSet, FineGym, and Figure Skating—show that MD-FED significantly outperforms baseline models in few-shot settings, providing a scalable and effective solution for fine-grained sports event detection. Our code is publicly available. 

## Environment
The code is tested in Linux (Ubuntu 22.04) with the dependency versions in requirements.txt.

## Dataset
Refer to the READMEs in the data directory for pre-processing and setup instructions.

## Training
To train the MD-FED model, use `python3 train_MD-FED.py <dataset_name> --frame_dir=<frame_dir> --flow_dir=<flow_dir> --pose_dir=<pose_dir> --stage=<training_stage> --visual_arch=<visual_arch> --skeleton_arch=<skeleton_arch> --num_samples=<num_samples> -s <save_dir>`.

* `<dataset_name>`: name of the dataset (i.e., f3set-tennis-sub, shuttleset, finegym-BB, fs_comp)
* `<frame_dir>`: path to the extracted frames
* `<flow_dir>`: path to the extracted optical flows
* `<pose_dir>`: path to the extracted 2D poses
* `<training_stage>`: training stage (i.e., 1, 2, or 3)
* `<visual_arch>`: visual-based (RGB and optical flow) feature extractor architecture (e.g., rny002, rny002_tsm)
* `<skeleton_arch>`: skeleton-based (2D poses) feature extractor architecture (e.g., stgcn++, msg3d)
* `<num_samples>`: number of sample clips used for training in few-shot settings (e.g., 25, 100-clip); if -1, use all training samples
* `<save_dir>`: path to save logs, checkpoints, and inference results; please include "stageX" in the saved directory name (e.g., "stage1", "stage2")

Training will produce checkpoints, predictions for the `val` split, and predictions for the `test` split on the best validation epoch.

## Data format
Each dataset has plaintext files that contain the list of event types `events.txt` and elements: `elements.txt`

This is a list of the event names, one per line: `{split}.json`

This file contains entries for each video and its contained events.

```
[
    {
        "fps": 25,
        "height": 720,
        "width": 1280,
        "num_frames": 342,  // number of frames in this clip
        "video": "20210909-W-US_Open-SF-Aryna_Sabalenka-Leylah_Fernandez_170943_171285",  // "video name"_"start frame of the clip"_"end frame of the clip"
        "events": [
            {
                "frame": 100,               // Frame
                "label": EVENT_NAME,        // Event type
            },
            ...
        ],
    },
    ...
]
```

**Frame directory**

We assume pre-extracted frames, that have been resized to 224 pixels high or similar. The organization of the frames is expected to be <frame_dir>/<video_id>/<frame_number>.jpg. For example,

```
video1/
├─ 000000.jpg
├─ 000001.jpg
├─ 000002.jpg
├─ ...
video2/
├─ 000000.jpg
├─ ...
```

Similar format applies to the frames containing objects of interest.

**Flow directory**

We assume pre-extracted optical flows, that have been resized to 224 pixels high or similar. The organization of the frames is expected to be <flow_dir>/<video_id>/<flow_number>.jpg. For example,

```
video1/
├─ 000000.jpg
├─ 000001.jpg
├─ 000002.jpg
├─ ...
video2/
├─ 000000.jpg
├─ ...
```

The optical flow for all four datasets will be uploaded soon.

**Pose directory**

We assume pre-extracted 2D poses. The organization of the frames is expected to be <path_dir>/<video_id>.pkl. An example pickle file is of the format:

```
{
    "frame_dir": "20210909-W-US_Open-SF-Aryna_Sabalenka-Leylah_Fernandez_170943_171285",
    "total_frames": 342,
    "img_shape": (720, 1280),
    "keypoint": [
        // far-end player
        [
          // frame 0
          [[XX, XX],    // joint 0 (x,y)-axis
           [XX, XX],    // joint 1 (x,y)-axis
           [XX, XX],    // joint 2 (x,y)-axis
           ...]
          // frame 1
          [[XX, XX],    // joint 0 (x,y)-axis
           [XX, XX],    // joint 1 (x,y)-axis
           [XX, XX],    // joint 2 (x,y)-axis
           ...]    
          ...
        ],
        // near-end player
        [
          // frame 0
          [[XX, XX],    // joint 0 (x,y)-axis
           [XX, XX],    // joint 1 (x,y)-axis
           [XX, XX],    // joint 2 (x,y)-axis
           ...]
          // frame 1
          [[XX, XX],    // joint 0 (x,y)-axis
           [XX, XX],    // joint 1 (x,y)-axis
           [XX, XX],    // joint 2 (x,y)-axis
           ...]    
          ...
        ],
    ],
    "keypoint_score": [
        // far-end player
        [
          // frame 0
          [XX,    // joint 0 confidence score
           XX,    // joint 1 confidence score
           XX,    // joint 2 confidence score
           ...]
          // frame 1
          [XX,    // joint 0 confidence score
           XX,    // joint 1 confidence score
           XX,    // joint 2 confidence score
           ...] 
          ...
        ],
        // near-end player
        [
          // frame 0
          [XX,    // joint 0 confidence score
           XX,    // joint 1 confidence score
           XX,    // joint 2 confidence score
           ...]
          // frame 1
          [XX,    // joint 0 confidence score
           XX,    // joint 1 confidence score
           XX,    // joint 2 confidence score
           ...]   
          ...
        ],
    ],
    
}
```

The 2D human poses for all four datasets will be uploaded soon.
