#!/usr/bin/env python3

import os
import cv2
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from util.io import load_json
from .transform import RandomGaussianNoise, RandomHorizontalFlipFLow, \
    RandomOffsetFlow, SeedableRandomSquareCrop, ThreeCrop


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class FrameReader:

    IMG_NAME = '{:06d}.jpg'

    def __init__(self, frame_dir, crop_transform, img_transform, flow_transform, same_transform,
                 pose_dir=None, flow_dir=None):
        self._frame_dir = frame_dir
        self._flow_dir = flow_dir
        self._pose_dir = pose_dir
        self._crop_transform = crop_transform
        self._img_transform = img_transform
        self._flow_transform = flow_transform
        self._same_transform = same_transform

    def read_frame(self, frame_path, is_flow=False):
        img = torchvision.io.read_image(frame_path).float() / 255
        if is_flow:
            img = img[1:, :, :]  # GB channels contain data
        return img

    def normalize_keypoints(self, keypoints, image_width=1280, image_height=720, reference_keypoints=(5, 6)):
        num_people, num_joints, _ = keypoints.shape

        # Step 1: Normalize to range [-1, 1] based on image dimensions
        keypoints[:, :, 0] = (keypoints[:, :, 0] / image_width) * 2 - 1  # Normalize x-coordinates
        keypoints[:, :, 1] = (keypoints[:, :, 1] / image_height) * 2 - 1  # Normalize y-coordinates

        # Step 2: Center the keypoints around a reference keypoint (e.g., pelvis or body center)
        reference_point = np.mean(keypoints[:, reference_keypoints, :], axis=1, keepdims=True)  # Compute center
        keypoints -= reference_point  # Shift to center

        return keypoints

    def load_frames(self, video_name, start, end, pad=False, stride=1, randomize=False):
        rand_crop_state = None
        rand_state_backup = None
        ret_rgb = [] if self._frame_dir is not None else torch.zeros(3, 224, 224)
        ret_flow = [] if self._flow_dir is not None else torch.zeros(3, 224, 224)
        ret_sk = [] if self._pose_dir is not None else torch.zeros(2, 17, 2)
        n_pad_start = 0
        n_pad_end = 0

        skeletons, d = None, None
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
            if self._frame_dir is not None:
                frame_path = os.path.join(self._frame_dir, video_name, img_num)
            if self._flow_dir is not None:
                flow_path = os.path.join(self._flow_dir, video_name, img_num)
            try:
                # rgb image
                if self._frame_dir is not None:
                    img = self.read_frame(frame_path)

                # optical flow
                if self._flow_dir is not None:
                    flow = self.read_frame(flow_path, is_flow=True)

                # 2D skeleton
                if self._pose_dir is not None and skeletons is not None:
                    keypoints = skeletons['keypoint'][:, frame_num, :, :]
                    keypoints = self.normalize_keypoints(keypoints)
                    keypoints = torch.from_numpy(keypoints).float()

                # rgb image crop
                if self._crop_transform:
                    if self._same_transform:
                        if rand_crop_state is None:
                            rand_crop_state = random.getstate()
                        else:
                            rand_state_backup = random.getstate()
                            random.setstate(rand_crop_state)

                    if self._crop_transform is not None:
                        if self._frame_dir is not None:
                            img = self._crop_transform(img)
                        if self._flow_dir is not None:
                            flow = self._crop_transform(flow)

                    if rand_state_backup is not None:
                        # Make sure that rand state still advances
                        random.setstate(rand_state_backup)
                        rand_state_backup = None

                if not self._same_transform:
                    if self._frame_dir is not None:
                        img = self._img_transform(img)
                    if self._flow_dir is not None:
                        flow = self._flow_transform(flow)

                if self._frame_dir is not None:
                    ret_rgb.append(img)

                if self._flow_dir is not None:
                    ret_flow.append(flow)
                    
                if self._pose_dir is not None:
                    ret_sk.append(keypoints)

            except (RuntimeError, IndexError):
                n_pad_end += 1
                
        # In the multicrop case, the shape is (B, T, C, H, W)
        if self._frame_dir is not None:
            if len(ret_rgb) == 0:
                print(self._frame_dir)
                print(video_name, start, end)
            ret_rgb = torch.stack(ret_rgb, dim=int(len(ret_rgb[0].shape) == 4))
            if self._same_transform:
                ret_rgb = self._img_transform(ret_rgb)

        if self._flow_dir is not None:
            ret_flow = torch.stack(ret_flow, dim=int(len(ret_flow[0].shape) == 4))
            if self._same_transform:
                ret_flow = self._flow_transform(ret_flow)

        if self._pose_dir is not None:
            ret_sk = torch.stack(ret_sk, dim=int(len(ret_sk[0].shape) == 4))

        # Always pad start, but only pad end if requested
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


# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5

def _get_deferred_rgb_transform():
    img_transforms = [
        # Jittering separately is faster (low variance)
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(saturation=(0.7, 1.2))
            ]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(brightness=(0.7, 1.2))
            ]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(contrast=(0.7, 1.2))
            ]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    return torch.jit.script(nn.Sequential(*img_transforms))


def _get_deferred_bw_transform():
    img_transforms = [
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(brightness=0.3)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(contrast=0.3)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        RandomGaussianNoise()
    ]
    return torch.jit.script(nn.Sequential(*img_transforms))


def _load_frame_deferred(gpu_transform, batch, device):
    frame = batch['frame'].to(device)
    with torch.no_grad():
        for i in range(frame.shape[0]):
            frame[i] = gpu_transform(frame[i])
    return frame


def _get_img_transforms(
        is_eval,
        crop_dim,
        same_transform,
        modality='rgb',
        defer_transform=False,
        multi_crop=False
):
    crop_transform = None
    if crop_dim is not None:
        if multi_crop:
            assert is_eval
            crop_transform = ThreeCrop(crop_dim)
        elif is_eval:
            crop_transform = transforms.CenterCrop(crop_dim)
        elif same_transform:
            print('=> Using seeded crops!')
            crop_transform = SeedableRandomSquareCrop(crop_dim)
        else:
            crop_transform = transforms.RandomCrop(crop_dim)

    img_transforms = []
    if modality == 'rgb':
        if not is_eval:
            # img_transforms.append(
            #     transforms.RandomHorizontalFlip())

            if not defer_transform:
                img_transforms.extend([
                    # Jittering separately is faster (low variance)
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(hue=0.2)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([
                            transforms.ColorJitter(saturation=(0.7, 1.2))
                        ]), p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([
                            transforms.ColorJitter(brightness=(0.7, 1.2))
                        ]), p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([
                            transforms.ColorJitter(contrast=(0.7, 1.2))
                        ]), p=0.25),

                    # Jittering together is slower (high variance)
                    # transforms.RandomApply(
                    #     nn.ModuleList([
                    #         transforms.ColorJitter(
                    #             brightness=(0.7, 1.2), contrast=(0.7, 1.2),
                    #             saturation=(0.7, 1.2), hue=0.2)
                    #     ]), p=0.8),

                    transforms.RandomApply(
                        nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25)
                ])

        if not defer_transform:
            img_transforms.append(transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD))
    elif modality == 'bw':
        if not is_eval:
            img_transforms.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25)])
        img_transforms.append(transforms.Grayscale())

        if not defer_transform:
            if not is_eval:
                img_transforms.extend([
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(brightness=0.3)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(contrast=0.3)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
                ])

            img_transforms.append(transforms.Normalize(
                mean=[0.5], std=[0.5]))

            if not is_eval:
                img_transforms.append(RandomGaussianNoise())
    elif modality == 'flow':
        assert not defer_transform

        img_transforms.append(transforms.Normalize(
            mean=[0.5, 0.5], std=[0.5, 0.5]))

        if not is_eval:
            img_transforms.extend([
                # RandomHorizontalFlipFLow(),
                RandomOffsetFlow(),
                RandomGaussianNoise()
            ])
    else:
        raise NotImplementedError(modality)

    img_transform = torch.jit.script(nn.Sequential(*img_transforms))
    return crop_transform, img_transform


def _print_info_helper(src_file, labels):
        num_frames = sum([x['num_frames'] for x in labels])
        num_events = sum([len(x['events']) for x in labels])
        print('{} : {} videos, {} frames, {:0.5f}% non-bg'.format(
            src_file, len(labels), num_frames,
            num_events / num_frames * 100))


IGNORED_NOT_SHOWN_FLAG = False
class ActionSeqDataset(Dataset):

    def __init__(
            self,
            classes,                    # dict of class names to idx
            label_file,                 # path to label json
            frame_dir,                  # path to frames
            clip_len,                   # number of frames per clip
            dataset_len,                # Number of clips
            flow_dir=None,              # path to flows
            pose_dir=None,              # path to 2D poses
            is_eval=True,               # Disable random augmentation
            crop_dim=None,              # crop dimension
            stride=1,                   # Downsample frame rate
            same_transform=True,        # Apply the same random augmentation to
                                        # each frame in a clip
            dilate_len=0,               # Dilate ground truth labels
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start
                                        # and end of videos
            stage=1,                    # training stage
            num_samples=-1,             # number of training samples
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        if num_samples < 0:
            num_samples = len(self._labels)
        for i in range(num_samples):
            self._labels[i]['keep'] = True
        if stage != 2:
            self._labels = self._labels[:num_samples]
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}

        # Sample videos weighted by their length
        num_frames = [v['num_frames'] for v in self._labels]
        self._weights_by_length = np.array(num_frames) / np.sum(num_frames)

        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        self._dataset_len = dataset_len
        assert dataset_len > 0
        self._pad_len = pad_len  # SET PAD LEN = 0
        assert pad_len >= 0
        self._is_eval = is_eval

        # Label modifications
        self._dilate_len = dilate_len

        # Try to do defer the latter half of the transforms to the GPU
        self._gpu_transform = None
        if not is_eval and same_transform:
            print('=> Deferring some RGB transforms to the GPU!')
            self._gpu_transform = _get_deferred_rgb_transform()

        crop_transform, img_transform = _get_img_transforms(
            is_eval, crop_dim, same_transform, defer_transform=self._gpu_transform is not None)
        _, flow_transform = _get_img_transforms(
            is_eval, crop_dim, same_transform, modality='flow')

        self._frame_reader = FrameReader(frame_dir, crop_transform, img_transform, flow_transform, same_transform,
                                         pose_dir=pose_dir, flow_dir=flow_dir)

    def load_frame_gpu(self, batch, device):
        if self._gpu_transform is None:
            frame = batch['frame'].to(device)
        else:
            frame = _load_frame_deferred(self._gpu_transform, batch, device)
        return frame

    def load_flow_gpu(self, batch, device):
        flow = batch['flow'].to(device)
        return flow

    def load_skeleton_gpu(self, batch, device):
        skeleton = batch['skeleton'].to(device)
        return skeleton

    def _sample_uniform(self):
        video_meta = random.choices(
            self._labels, weights=self._weights_by_length)[0]
        video_len = video_meta['num_frames']
        fps = video_meta['fps']
        stride = self._stride
        base_idx = -self._pad_len * stride + random.randint(
            0, max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * stride))

        return video_meta, base_idx, stride

    def _get_one(self):
        video_meta, base_idx, stride = self._sample_uniform()

        frames, flows, skeletons = self._frame_reader.load_frames(
            video_meta['video'], base_idx,
            base_idx + self._clip_len * stride, pad=True,
            stride=stride, randomize=not self._is_eval)

        start_time = base_idx / video_meta['fps']
        end_time = (base_idx + self._clip_len * self._stride) / video_meta['fps']

        # coarse-grained per-frame label
        coarse_labels = np.zeros(self._clip_len, np.int64)
        for event in video_meta['events']:
            event_frame = event['frame']
            # Index of event in label array
            label_idx = (event_frame - base_idx) // stride
            if (label_idx >= 0 and label_idx < self._clip_len):
                for i in range(max(0, label_idx), min(self._clip_len, label_idx + 1)):
                    coarse_labels[i] = 1

        # fine-grained per-frame label
        fine_labels = np.zeros((self._clip_len, len(self._class_dict)), np.int64)
        for event in video_meta['events']:
            event_frame = event['frame']
            # Index of event in label array
            label_idx = (event_frame - base_idx) // stride
            if (label_idx >= 0 and label_idx < self._clip_len):
                for i in range(max(0, label_idx), min(self._clip_len, label_idx + 1)):
                    for sub_label in event['label'].split('_'):
                        if sub_label in self._class_dict:
                            fine_labels[i, self._class_dict[sub_label] - 1] = 1
        if 'keep' in video_meta:
            # print(video_meta)
            coarse_mask = np.ones(self._clip_len, np.int64)
        else:
            coarse_mask = np.zeros(self._clip_len, np.int64)

        return {'frame': frames,
                'flow': flows,
                'skeleton': skeletons,
                'contains_event': int(np.sum(coarse_labels) > 0),
                'coarse_label': coarse_labels,
                'fine_label': fine_labels,
                'coarse_mask': coarse_mask}

    def __getitem__(self, unused):
        ret = self._get_one()
        return ret

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)


class ActionSeqVideoDataset(Dataset):

    def __init__(
            self,
            classes,
            label_file,
            frame_dir,
            clip_len,
            flow_dir=None,
            pose_dir=None,
            overlap_len=0,
            crop_dim=None,
            stride=1,
            pad_len=DEFAULT_PAD_LEN,
            flip=False,
            multi_crop=False,
            skip_partial_end=True,
            is_test=False,
            num_samples=-1,
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        if not is_test and num_samples > 0:
            self._labels = self._labels[:num_samples]
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        self._stride = stride

        crop_transform, img_transform = _get_img_transforms(
            is_eval=True, crop_dim=crop_dim, same_transform=True, multi_crop=multi_crop)
        _, flow_transform = _get_img_transforms(
            is_eval=True, crop_dim=crop_dim, same_transform=True, modality='flow', multi_crop=multi_crop)

        # No need to enforce same_transform since the transforms are deterministic
        self._frame_reader = FrameReader(frame_dir, crop_transform, img_transform, flow_transform, True,
                                         pose_dir=pose_dir, flow_dir=flow_dir)

        self._flip = flip
        self._multi_crop = multi_crop

        self._clips = []
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(0, l['num_frames'] - (overlap_len * stride)
                        * int(skip_partial_end)), \
                # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride
            ):
                has_clip = True
                self._clips.append((l['video'], l['fps'], i))
            assert has_clip, l

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        video_name, fps, start = self._clips[idx]

        stride = self._stride
        frames, flows, skeletons = self._frame_reader.load_frames(video_name, start, start + self._clip_len * stride,
                                                                  pad=True, stride=stride)

        if self._flip:
            frames = torch.stack((frames, frames.flip(-1)), dim=0)

        return {'video': video_name, 'fps': fps, 'start': start // stride, 'frame': frames, 'flow': flows,
                'skeleton': skeletons}

    def get_labels(self, video, index=0):
        meta = self._labels[self._video_idxs[video]]
        num_frames = meta['num_frames']
        num_labels = num_frames // self._stride
        coarse_labels = np.zeros(num_labels, int)
        fine_labels = np.zeros((num_labels, len(self._class_dict)), int)
        
        for event in meta['events']:
            frame = event['frame']
            label = event['label']
            if frame < num_frames:
                coarse_labels[frame // self._stride] = 1
                fine_label = np.zeros(len(self._class_dict), int)
                for sub_label in label.split('_'):
                    if sub_label in self._class_dict:
                        fine_label[self._class_dict[sub_label] - 1] = 1
                fine_labels[frame // self._stride, :] = fine_label
            else:
                print('Warning: {} >= {} is past the end {}'.format(
                    frame, num_frames, meta['video']))
        return coarse_labels, fine_labels

    @property
    def augment(self):
        return self._flip or self._multi_crop

    @property
    def videos(self):
        return sorted([
            (v['video'], v['num_frames'] // self._stride,
             v['fps'] / self._stride) for v in self._labels])

    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                x_copy['fps'] /= self._stride
                x_copy['num_frames'] //= self._stride
                for e in x_copy['events']:
                    e['frame'] //= self._stride
                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x['num_frames'] for x in self._labels])
        num_events = sum([len(x['events']) for x in self._labels])
        print('{} : {} videos, {} frames ({} stride), {:0.5f}% non-bg'.format(
            self._src_file, len(self._labels), num_frames, self._stride,
            num_events / num_frames * 100))