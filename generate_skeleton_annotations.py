#!/usr/bin/env python3
"""
Generate skeleton annotations for videos using HRNet pose estimation model.
Processes videos frame by frame and extracts pose keypoints.
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add HRNet to path
hrnet_path = os.path.join(os.path.dirname(__file__), 'deep-high-resolution-net.pytorch')
sys.path.insert(0, os.path.join(hrnet_path, 'lib'))

# Import HRNet modules
from config import cfg, update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform
import models

# COCO keypoint names (17 keypoints)
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

COCO_KEYPOINT_INDEXES = {i: name for i, name in enumerate(COCO_KEYPOINT_NAMES)}


def get_person_detection_boxes(model, img, threshold=0.5):
    """Detect person bounding boxes using Faster R-CNN"""
    from PIL import Image
    pil_image = Image.fromarray(img)
    transform = transforms.Compose([transforms.ToTensor()])
    transformed_img = transform(pil_image)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pred = model([transformed_img.to(device)])
    
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
    pred_scores = list(pred[0]['scores'].cpu().detach().numpy())
    
    person_boxes = []
    for pred_class, pred_box, pred_score in zip(pred_classes, pred_boxes, pred_scores):
        if (pred_score > threshold) and (pred_class == 'person'):
            person_boxes.append(pred_box)
    
    return person_boxes


def box_to_center_scale(box, model_image_width, model_image_height):
    """Convert bounding box to center and scale"""
    center = np.zeros((2), dtype=np.float32)
    
    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5
    
    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200
    
    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    
    return center, scale


def get_pose_estimation_prediction(pose_model, image, centers, scales, transform, device):
    """Get pose estimation predictions"""
    rotation = 0
    
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)
        
        model_input = transform(model_input)
        model_inputs.append(model_input)
    
    model_inputs = torch.stack(model_inputs)
    
    output = pose_model(model_inputs.to(device))
    coords, _ = get_final_preds(
        cfg,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))
    
    return coords


def process_frames_from_directory(frame_dir, video_id, pose_model, box_model, output_file,
                                   detection_threshold=0.9, skip_frames=1):
    """
    Process frames from a directory and extract skeleton annotations.
    
    Args:
        frame_dir: Directory containing frame images (e.g., ./ncaa_frames/video_id/)
        video_id: Video identifier
        pose_model: HRNet pose estimation model
        box_model: Person detection model (Faster R-CNN)
        output_file: Path to save JSON output
        detection_threshold: Threshold for person detection
        skip_frames: Process every N frames (1 = all frames)
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Transformation for pose estimation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Get frame files
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if not frame_files:
        raise ValueError(f"No frame files found in {frame_dir}")
    
    # Read FPS if available
    fps_file = os.path.join(frame_dir, 'fps.txt')
    if os.path.exists(fps_file):
        with open(fps_file, 'r') as f:
            fps = float(f.read().strip())
    else:
        fps = 30.0  # Default
        print(f"Warning: fps.txt not found, using default {fps} fps")
    
    # Get image dimensions from first frame
    first_frame_path = os.path.join(frame_dir, frame_files[0])
    first_img = cv2.imread(first_frame_path)
    if first_img is None:
        raise ValueError(f"Could not read first frame: {first_frame_path}")
    height, width = first_img.shape[:2]
    
    total_frames = len(frame_files)
    print(f"Frame directory: {frame_dir}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}, Size: {width}x{height}")
    print(f"Processing every {skip_frames} frame(s)")
    
    # Process frames
    annotations = []
    processed_count = 0
    
    pbar = tqdm(total=total_frames, desc=f"Processing {video_id}")
    
    for idx, frame_file in enumerate(frame_files):
        # Skip frames if needed
        if idx % skip_frames != 0:
            pbar.update(1)
            continue
        
        frame_path = os.path.join(frame_dir, frame_file)
        image_bgr = cv2.imread(frame_path)
        
        if image_bgr is None:
            print(f"Warning: Could not read frame {frame_file}")
            pbar.update(1)
            continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Person detection
        pred_boxes = get_person_detection_boxes(box_model, image_rgb, threshold=detection_threshold)
        
        # Initialize frame annotation
        # Extract frame number from filename (e.g., "000000.jpg" -> 0)
        try:
            frame_num = int(frame_file.replace('.jpg', ''))
        except ValueError:
            # If filename doesn't match expected format, use index
            frame_num = idx
        frame_annotation = {
            'frame': frame_num,
            'persons': []
        }
        
        if pred_boxes:
            # Get centers and scales for pose estimation
            centers = []
            scales = []
            for box in pred_boxes:
                center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                centers.append(center)
                scales.append(scale)
            
            # Pose estimation
            pose_preds = get_pose_estimation_prediction(
                pose_model, image_rgb, centers, scales, pose_transform, device
            )
            
            # Store keypoints for each person
            for i, coords in enumerate(pose_preds):
                person_keypoints = {}
                for j, (x, y) in enumerate(coords):
                    keypoint_name = COCO_KEYPOINT_INDEXES[j]
                    person_keypoints[keypoint_name] = {
                        'x': float(x),
                        'y': float(y),
                        'visible': 1.0 if x > 0 and y > 0 else 0.0
                    }
                
                frame_annotation['persons'].append({
                    'person_id': i,
                    'bbox': [
                        [float(pred_boxes[i][0][0]), float(pred_boxes[i][0][1])],
                        [float(pred_boxes[i][1][0]), float(pred_boxes[i][1][1])]
                    ],
                    'keypoints': person_keypoints
                })
        
        annotations.append(frame_annotation)
        processed_count += 1
        pbar.update(1)
    
    pbar.close()
    
    # Create output structure
    output_data = {
        'video_id': video_id,
        'frame_dir': str(frame_dir),
        'fps': float(fps),
        'total_frames': total_frames,
        'processed_frames': processed_count,
        'width': width,
        'height': height,
        'keypoint_names': COCO_KEYPOINT_NAMES,
        'annotations': annotations
    }
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSkeleton annotations saved to: {output_file}")
    print(f"Processed {processed_count} frames")
    print(f"Total persons detected: {sum(len(a['persons']) for a in annotations)}")
    
    return output_file


def process_video(video_path, pose_model, box_model, output_file, 
                  inference_fps=None, detection_threshold=0.9):
    """
    Process a video and extract skeleton annotations.
    
    Args:
        video_path: Path to input video
        pose_model: HRNet pose estimation model
        box_model: Person detection model (Faster R-CNN)
        output_file: Path to save JSON output
        inference_fps: FPS for inference (None = process all frames)
        detection_threshold: Threshold for person detection
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Transformation for pose estimation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Open video
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {total_frames} frames, {fps:.2f} fps, {width}x{height}")
    
    # Determine skip frame count
    if inference_fps and inference_fps < fps:
        skip_frame_cnt = round(fps / inference_fps)
        print(f"Processing every {skip_frame_cnt} frames (target FPS: {inference_fps})")
    else:
        skip_frame_cnt = 1
        print("Processing all frames")
    
    # Process frames
    annotations = []
    frame_count = 0
    processed_count = 0
    
    pbar = tqdm(total=total_frames, desc="Processing video")
    
    while vidcap.isOpened():
        ret, image_bgr = vidcap.read()
        frame_count += 1
        
        if not ret:
            break
        
        # Skip frames if needed
        if frame_count % skip_frame_cnt != 0:
            pbar.update(1)
            continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Person detection
        pred_boxes = get_person_detection_boxes(box_model, image_rgb, threshold=detection_threshold)
        
        # Initialize frame annotation
        frame_annotation = {
            'frame': frame_count - 1,  # 0-indexed
            'persons': []
        }
        
        if pred_boxes:
            # Get centers and scales for pose estimation
            centers = []
            scales = []
            for box in pred_boxes:
                center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                centers.append(center)
                scales.append(scale)
            
            # Pose estimation
            pose_preds = get_pose_estimation_prediction(
                pose_model, image_rgb, centers, scales, pose_transform, device
            )
            
            # Store keypoints for each person
            for i, coords in enumerate(pose_preds):
                person_keypoints = {}
                for j, (x, y) in enumerate(coords):
                    keypoint_name = COCO_KEYPOINT_INDEXES[j]
                    person_keypoints[keypoint_name] = {
                        'x': float(x),
                        'y': float(y),
                        'visible': 1.0 if x > 0 and y > 0 else 0.0
                    }
                
                frame_annotation['persons'].append({
                    'person_id': i,
                    'bbox': [
                        [float(pred_boxes[i][0][0]), float(pred_boxes[i][0][1])],
                        [float(pred_boxes[i][1][0]), float(pred_boxes[i][1][1])]
                    ],
                    'keypoints': person_keypoints
                })
        
        annotations.append(frame_annotation)
        processed_count += 1
        pbar.update(skip_frame_cnt)
    
    vidcap.release()
    pbar.close()
    
    # Create output structure
    output_data = {
        'video_path': str(video_path),
        'fps': float(fps),
        'total_frames': total_frames,
        'processed_frames': processed_count,
        'width': width,
        'height': height,
        'keypoint_names': COCO_KEYPOINT_NAMES,
        'annotations': annotations
    }
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSkeleton annotations saved to: {output_file}")
    print(f"Processed {processed_count} frames")
    print(f"Total persons detected: {sum(len(a['persons']) for a in annotations)}")
    
    return output_file


def load_models(model_path, config_path=None):
    """Load HRNet pose estimation model and person detection model"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load config - must use a proper config file
    if config_path is None:
        # Try to find a default config based on model name
        base_path = os.path.join(
            os.path.dirname(__file__),
            'deep-high-resolution-net.pytorch',
            'experiments',
            'coco',
            'hrnet'
        )
        
        if 'w48' in model_path and '384x288' in model_path:
            config_path = os.path.join(base_path, 'w48_384x288_adam_lr1e-3.yaml')
        elif 'w32' in model_path and '256x192' in model_path:
            config_path = os.path.join(base_path, 'w32_256x192_adam_lr1e-3.yaml')
        else:
            # Default to w48_384x288
            config_path = os.path.join(base_path, 'w48_384x288_adam_lr1e-3.yaml')
    
    if config_path and os.path.exists(config_path):
        print(f"Using config file: {config_path}")
        # Create a mock args object for update_config
        class Args:
            def __init__(self):
                self.cfg = config_path
                self.opts = []
                self.modelDir = ''
                self.logDir = ''
                self.dataDir = ''
                self.prevModelDir = ''
        args = Args()
        update_config(cfg, args)
    else:
        # Try alternative paths
        alt_paths = [
            os.path.join(os.path.dirname(__file__), 'deep-high-resolution-net.pytorch', 'experiments', 'coco', 'hrnet', 'w48_384x288_adam_lr1e-3.yaml'),
            os.path.join(os.path.dirname(__file__), 'deep-high-resolution-net.pytorch', 'experiments', 'coco', 'w48_384x288_adam_lr1e-3.yaml'),
        ]
        
        found = False
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                config_path = alt_path
                print(f"Using config file: {config_path}")
                class Args:
                    def __init__(self):
                        self.cfg = config_path
                        self.opts = []
                        self.modelDir = ''
                        self.logDir = ''
                        self.dataDir = ''
                        self.prevModelDir = ''
                args = Args()
                update_config(cfg, args)
                found = True
                break
        
        if not found:
            raise FileNotFoundError(
                f"Config file not found. Tried:\n" +
                "\n".join([f"  - {p}" for p in [config_path] + alt_paths]) +
                f"\nPlease specify a valid config file using --config option"
            )
    
    print(f"Loading pose model from: {model_path}")
    print(f"Model image size: {cfg.MODEL.IMAGE_SIZE}")
    
    # Load pose estimation model
    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    pose_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    pose_model.to(device)
    pose_model.eval()
    
    # Load person detection model (Faster R-CNN)
    # This is separate from HRNet - it's used to detect people in the frame first
    print("Loading person detection model (Faster R-CNN)...")
    print("Note: This model is downloaded automatically from torchvision (one-time download)")
    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(device)
    box_model.eval()
    
    return pose_model, box_model, device


def main():
    parser = argparse.ArgumentParser(
        description='Generate skeleton annotations for videos using HRNet'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to input video file OR frame directory'
    )
    parser.add_argument(
        '--video_id',
        type=str,
        default=None,
        help='Video ID (required when using frame directory)'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        default=None,
        help='Base frame directory (default: ./ncaa_frames)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./deep-high-resolution-net.pytorch/models/pose_hrnet_w48_384x288.pth',
        help='Path to HRNet model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--inference_fps',
        type=int,
        default=None,
        help='Target FPS for inference (only for video input)'
    )
    parser.add_argument(
        '--skip_frames',
        type=int,
        default=1,
        help='Process every N frames (for frame directory input, default: 1)'
    )
    parser.add_argument(
        '--detection_threshold',
        type=float,
        default=0.9,
        help='Person detection threshold (default: 0.9)'
    )
    parser.add_argument(
        '--use_frames',
        action='store_true',
        help='Force using frame directory mode'
    )
    
    args = parser.parse_args()
    
    # Determine input type
    if args.use_frames or os.path.isdir(args.input_path):
        # Frame directory mode
        if args.frame_dir:
            frame_dir = os.path.join(args.frame_dir, args.video_id or os.path.basename(args.input_path))
        else:
            frame_dir = args.input_path
        
        if not os.path.exists(frame_dir):
            print(f"Error: Frame directory not found: {frame_dir}")
            return
        
        video_id = args.video_id or os.path.basename(frame_dir)
        
        if args.output is None:
            args.output = f'./{video_id}_skeleton.json'
        
        print("=" * 60)
        print("HRNet Skeleton Annotation Generator (Frame Directory Mode)")
        print("=" * 60)
        print(f"Frame directory: {frame_dir}")
        print(f"Video ID: {video_id}")
        print(f"Model: {args.model_path}")
        print(f"Output: {args.output}")
        print("=" * 60)
        
        input_type = 'frames'
    else:
        # Video file mode
        if not os.path.exists(args.input_path):
            print(f"Error: Video file not found: {args.input_path}")
            return
        
        if args.output is None:
            video_name = os.path.splitext(os.path.basename(args.input_path))[0]
            args.output = f'./{video_name}_skeleton.json'
        
        print("=" * 60)
        print("HRNet Skeleton Annotation Generator (Video Mode)")
        print("=" * 60)
        print(f"Video: {args.input_path}")
        print(f"Model: {args.model_path}")
        print(f"Output: {args.output}")
        print("=" * 60)
        
        input_type = 'video'
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Load models
    print("\nLoading models...")
    pose_model, box_model, device = load_models(args.model_path, args.config)
    
    # Process input
    print("\nProcessing...")
    if input_type == 'frames':
        process_frames_from_directory(
            frame_dir,
            video_id,
            pose_model,
            box_model,
            args.output,
            args.detection_threshold,
            args.skip_frames
        )
    else:
        process_video(
            args.input_path,
            pose_model,
            box_model,
            args.output,
            args.inference_fps,
            args.detection_threshold
        )
    
    print("\n" + "=" * 60)
    print("Skeleton annotation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
