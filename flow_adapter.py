#!/usr/bin/env python3
"""
Adapter module to handle .npy format optical flow files for MD-FED.
This module patches MD-FED's FrameReader to support .npy flow files.
"""

import os
import numpy as np
import torch
import torchvision


def patch_frame_reader_for_npy_flow():
    """
    Patch MD-FED's FrameReader to support .npy flow files.
    This function modifies the FrameReader class to handle .npy format flow files.
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MD-FED'))
    
    from dataset.input_process import FrameReader
    
    # Store original methods
    original_read_frame = FrameReader.read_frame
    original_load_frames = FrameReader.load_frames
    
    def read_frame_with_npy_support(self, frame_path, is_flow=False):
        """Enhanced read_frame that supports .npy flow files"""
        if is_flow and frame_path.endswith('.npy'):
            # Handle .npy flow files
            if not os.path.exists(frame_path):
                return torch.zeros(2, 224, 224)
            
            # Load .npy file: shape is [H, W, 2]
            flow_np = np.load(frame_path)
            
            # Convert to [2, H, W] format
            if len(flow_np.shape) == 3 and flow_np.shape[2] == 2:
                flow_tensor = torch.from_numpy(flow_np).permute(2, 0, 1).float()
            else:
                # Fallback: create zero flow
                flow_tensor = torch.zeros(2, 224, 224)
            
            return flow_tensor
        else:
            # Use original implementation for JPG files
            return original_read_frame(self, frame_path, is_flow)
    
    def load_frames_with_npy_support(self, video_name, start, end, pad=False, stride=1, randomize=False):
        """Enhanced load_frames that supports .npy flow files"""
        rand_crop_state = None
        rand_state_backup = None
        ret_rgb = [] if self._frame_dir is not None else torch.zeros(3, 224, 224)
        ret_flow = [] if self._flow_dir is not None else torch.zeros(2, 224, 224)
        ret_sk = [] if self._pose_dir is not None else torch.zeros(2, 17, 2)
        n_pad_start = 0
        n_pad_end = 0
        
        import pandas as pd
        skeletons = None
        if self._pose_dir is not None:
            pickle_path = os.path.join(self._pose_dir, '%s.pkl' % video_name)
            if os.path.exists(pickle_path):
                skeletons = pd.read_pickle(pickle_path)
        
        for frame_num in range(start, end, stride):
            if frame_num < 0:
                n_pad_start += 1
                continue
            
            img_num = FrameReader.IMG_NAME.format(frame_num)
            keypoints = torch.zeros(2, 17, 2)
            flow = torch.zeros(2, 224, 224)
            
            frame_path = None
            flow_path = None
            
            if self._frame_dir is not None:
                frame_path = os.path.join(self._frame_dir, video_name, img_num)
            
            if self._flow_dir is not None:
                # For .npy flow files, we need frame_num-1 to frame_num
                # Flow files are named: frame1_frame2.npy (e.g., 000000_000001.npy)
                if frame_num > 0:
                    prev_frame_num = frame_num - 1
                    flow_filename = f'{prev_frame_num:06d}_{frame_num:06d}.npy'
                    flow_path = os.path.join(self._flow_dir, video_name, flow_filename)
                else:
                    # For first frame, use zero flow
                    flow_path = None
            
            try:
                # RGB image
                if self._frame_dir is not None and os.path.exists(frame_path):
                    img = self.read_frame(frame_path)
                elif self._frame_dir is not None:
                    img = torch.zeros(3, 224, 224)
                
                # Optical flow (handle .npy format)
                if self._flow_dir is not None:
                    if flow_path and os.path.exists(flow_path) and flow_path.endswith('.npy'):
                        # Load .npy flow file
                        flow = self.read_frame(flow_path, is_flow=True)
                    elif flow_path and os.path.exists(flow_path):
                        # Try JPG format as fallback
                        flow = self.read_frame(flow_path, is_flow=True)
                    else:
                        # Use zero flow if file doesn't exist
                        flow = torch.zeros(2, 224, 224)
                
                # 2D skeleton
                if self._pose_dir is not None and skeletons is not None:
                    if frame_num < skeletons['keypoint'].shape[1]:
                        keypoints = skeletons['keypoint'][:, frame_num, :, :]
                        keypoints = self.normalize_keypoints(keypoints)
                        keypoints = torch.from_numpy(keypoints).float()
                
                # Apply transforms
                import random
                if self._crop_transform:
                    if self._same_transform:
                        if rand_crop_state is None:
                            rand_crop_state = random.getstate()
                        else:
                            rand_state_backup = random.getstate()
                            random.setstate(rand_crop_state)
                    
                    if self._crop_transform is not None:
                        if self._frame_dir is not None and len(ret_rgb) == 0 or (ret_rgb and not isinstance(ret_rgb[0], int)):
                            if not isinstance(img, int):
                                img = self._crop_transform(img)
                        if self._flow_dir is not None:
                            flow = self._crop_transform(flow)
                    
                    if rand_state_backup is not None:
                        random.setstate(rand_state_backup)
                        rand_state_backup = None
                
                if not self._same_transform:
                    if self._frame_dir is not None and not isinstance(img, int):
                        img = self._img_transform(img)
                    if self._flow_dir is not None:
                        flow = self._flow_transform(flow)
                
                if self._frame_dir is not None and not isinstance(img, int):
                    ret_rgb.append(img)
                
                if self._flow_dir is not None:
                    ret_flow.append(flow)
                    
                if self._pose_dir is not None:
                    ret_sk.append(keypoints)
            
            except (RuntimeError, IndexError, FileNotFoundError, KeyError) as e:
                n_pad_end += 1
                # Continue with zero tensors
                if self._frame_dir is not None:
                    ret_rgb.append(torch.zeros(3, 224, 224))
                if self._flow_dir is not None:
                    ret_flow.append(torch.zeros(2, 224, 224))
                if self._pose_dir is not None:
                    ret_sk.append(torch.zeros(2, 17, 2))
        
        # Stack tensors
        if self._frame_dir is not None:
            if len(ret_rgb) == 0:
                ret_rgb = [torch.zeros(3, 224, 224)]
            ret_rgb = torch.stack(ret_rgb, dim=int(len(ret_rgb[0].shape) == 4)) if ret_rgb else torch.zeros(3, 224, 224)
            if self._same_transform:
                ret_rgb = self._img_transform(ret_rgb)
        
        if self._flow_dir is not None:
            if len(ret_flow) == 0:
                ret_flow = [torch.zeros(2, 224, 224)]
            ret_flow = torch.stack(ret_flow, dim=int(len(ret_flow[0].shape) == 4)) if ret_flow else torch.zeros(2, 224, 224)
            if self._same_transform:
                ret_flow = self._flow_transform(ret_flow)
        
        if self._pose_dir is not None:
            if len(ret_sk) == 0:
                ret_sk = [torch.zeros(2, 17, 2)]
            ret_sk = torch.stack(ret_sk, dim=int(len(ret_sk[0].shape) == 4)) if ret_sk else torch.zeros(2, 17, 2)
        
        # Padding
        import torch.nn as nn
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            if self._frame_dir is not None:
                ret_rgb = nn.functional.pad(
                    ret_rgb, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0))
            if self._flow_dir is not None:
                ret_flow = nn.functional.pad(
                    ret_flow, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0))
            if self._pose_dir is not None:
                ret_sk = nn.functional.pad(
                    ret_sk, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0))
        
        return ret_rgb, ret_flow, ret_sk
    
    # Patch the methods
    FrameReader.read_frame = read_frame_with_npy_support
    FrameReader.load_frames = load_frames_with_npy_support
    
    print("FrameReader patched to support .npy flow files")


if __name__ == '__main__':
    # Test the patching
    patch_frame_reader_for_npy_flow()
    print("Flow adapter module loaded successfully")
