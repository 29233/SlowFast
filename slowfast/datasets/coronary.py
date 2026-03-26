# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Coronary CT Angiography Dataset for Video Classification.

This module provides a SlowFast-compatible dataset loader for coronary CT images.
It reads DICOM sequences from ID-named folders and uses metadata CSV for
train/val/test splits and frame range selection.
"""

import os
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from slowfast.utils.env import pathmgr

from .build import DATASET_REGISTRY

# Import DICOM reading utilities from project root
# Try multiple paths to find the dcm module
_DCM_MODULE_PATHS = [
    str(Path(__file__).parent.parent.parent.parent),  # SlowFast/slowfast/datasets/../../../../../ -> project root
    r"/18018998051/CTA",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."),
]

for _path in _DCM_MODULE_PATHS:
    if os.path.isdir(_path) and os.path.exists(os.path.join(_path, "dcm.py")):
        if _path not in sys.path:
            sys.path.insert(0, _path)
        break

from dcm import read_series, apply_window_preset


@DATASET_REGISTRY.register()
class Coronary(torch.utils.data.Dataset):
    """
    Coronary CT Angiography video loader.

    Construct the Coronary video loader, then sample clips from the DICOM sequences.
    For training and validation, a single clip is randomly sampled from every video
    with random cropping, scaling, and flipping. For testing, multiple clips are
    uniformly sampled from every video with uniform cropping.

    The dataset expects:
    - A metadata CSV file with columns: ID, Start Frame, End Frame, Plaque, Branch, SplitA, SplitB
    - DICOM files organized in ID-named subfolders under DATA.PATH_PREFIX
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Coronary video loader with a given CSV file.

        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
            num_retries (int): number of retries for failed loading.
        """
        assert mode in ["train", "val", "test"], \
            "Split '{}' not supported for Coronary".format(mode)
        self.mode = mode
        self.cfg = cfg
        self._video_meta = {}
        self._num_retries = num_retries
        self._num_epoch = 0.0
        self._num_yielded = 0

        # Determine which split column to use (SplitA or SplitB)
        # Use default value if not specified in config
        self.split_column = cfg.DATA.SPLIT_COLUMN if hasattr(cfg.DATA, "SPLIT_COLUMN") else "Split"

        # For training or validation mode, one single clip is sampled from every video.
        # For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every video.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        self._construct_loader()
        self.aug = False
        self.rand_erase = False

        if self.mode == "train" and cfg.AUG.ENABLE:
            self.aug = True
            if cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader from metadata CSV.

        The CSV format is:
        ID,Start Frame,End Frame,Plaque,Branch,SplitA,SplitB
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            "{}.csv".format(self.mode)
        )

        # Try to load from split-specific file first, fall back to full metadata
        if not pathmgr.exists(path_to_file):
            # Load full metadata and filter by split
            metadata_path = os.path.join(
                self.cfg.DATA.PATH_TO_DATA_DIR,
                "metadataV0.csv"
            )
            assert pathmgr.exists(metadata_path), \
                "{} not found".format(metadata_path)

            with pathmgr.open(metadata_path, "r") as f:
                df = pd.read_csv(f)

            # Filter by split column
            split_df = df[df[self.split_column] == self.mode].reset_index(drop=True)

            # Save temporary CSV for faster subsequent loads
            temp_csv_path = os.path.join(
                self.cfg.DATA.PATH_TO_DATA_DIR,
                "{}_{}.csv".format(self.mode, self.split_column.lower())
            )
            split_df.to_csv(temp_csv_path, index=False)
            path_to_file = temp_csv_path
        else:
            with pathmgr.open(path_to_file, "r") as f:
                split_df = pd.read_csv(f)

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._frame_ranges = []  # Store (start, end) for each video
        self._video_meta = {}

        for clip_idx, row in split_df.iterrows():
            video_id = str(int(row['ID']))
            # Start Frame and End Frame columns may not exist in all metadata files
            # Use 0 as default if columns don't exist or values are missing
            if 'Start Frame' in row:
                start_frame = int(row['Start Frame']) if not pd.isna(row['Start Frame']) else 0
            else:
                start_frame = 0
            if 'End Frame' in row:
                end_frame = int(row['End Frame']) if not pd.isna(row['End Frame']) else 0
            else:
                end_frame = 0

            # Create label from Plaque value (convert to classification bins if needed)
            # For regression, we can use Plaque directly
            plaque = row['Plaque']
            if pd.isna(plaque):
                label = 0  # Default label for missing values
            else:
                # Convert plaque to classification label (e.g., 0-100 -> 0-9 bins)
                # Or use as-is for regression
                label = int(plaque)

            for idx in range(self._num_clips):
                video_path = os.path.join(self.cfg.DATA.PATH_PREFIX, video_id)
                self._path_to_videos.append(video_path)
                self._labels.append(label)
                self._spatial_temporal_idx.append(idx)
                self._frame_ranges.append((start_frame, end_frame))
                self._video_meta[clip_idx * self._num_clips + idx] = {
                    'video_id': video_id,
                    'plaque': plaque,
                    'branch': row.get('Branch', 'Unknown'),
                }

        assert len(self._path_to_videos) > 0, \
            "Failed to load Coronary split {} from {}".format(self.mode, path_to_file)

    def _decord_decode(self, video_path: str,
                       start_frame: int, end_frame: int) -> Optional[Tuple[np.ndarray, List[int]]]:
        """
        Load DICOM sequence and extract frames.

        Temporal sampling is now done in __getitem__ for unified control.
        This method simply loads frames based on the specified range.

        Args:
            video_path: Path to folder containing DICOM files
            start_frame: Start frame index (inclusive)
            end_frame: End frame index (inclusive)

        Returns:
            Tuple of (frames, frame_indices) or None if loading fails
            - frames: numpy array of shape (T, H, W, C)
            - frame_indices: list of original frame indices from the video
        """
        import gc

        try:
            # Load DICOM series with timeout check
            # Check if path exists before attempting to load
            if not os.path.exists(video_path):
                print(f"Warning: Video path does not exist: {video_path}")
                return None

            # Load DICOM series
            series = read_series(video_path)

            if len(series) == 0:
                return None

            # Get total frame count
            num_frames = len(series.images)
            if num_frames == 0:
                return None

            # Load frames without temporal sampling
            # (Sampling is done in __getitem__ for unified control)
            if start_frame == 0 and end_frame == 0:
                # Load entire sequence
                frame_indices = list(range(num_frames))
            else:
                # Load specified range
                start_idx = max(0, min(start_frame, num_frames - 1))
                end_idx = max(start_idx, min(end_frame, num_frames - 1))
                frame_indices = list(range(start_idx, end_idx + 1))

            # Load only required frames (lazy loading)
            loaded_frames = []
            for idx in frame_indices:
                try:
                    # Load single frame directly without creating full volume
                    img = series.images[idx]
                    frame = img.to_ndarray(apply_rescale=True)
                    loaded_frames.append(frame)
                except (IndexError, Exception) as e:
                    # If frame loading fails, use last successful frame
                    if loaded_frames:
                        loaded_frames.append(loaded_frames[-1])
                    else:
                        return None

            # Stack frames: (T, H, W)
            frames = np.stack(loaded_frames, axis=0)

            # Clear memory
            del series
            del loaded_frames
            gc.collect()

            # Apply window/level for cardiac CT
            frames = apply_window_preset(frames, preset='cardiac')

            # Convert to 0-255 range (use float32 intermediate to avoid overflow)
            frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)

            # Expand to 3 channels (RGB)
            frames = np.stack([frames] * 3, axis=-1)  # Shape: (T, H, W, 3)

            return frames, frame_indices

        except Exception as e:
            print(f"Error loading DICOM series from {video_path}: {e}")
            return None

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video index.

        Args:
            index (int): the video index provided by the pytorch sampler.

        Returns:
            frames (tensor): the frames sampled from the video.
            label (int): the label of the current video.
            index (int): the index of the video.
            time_idx (array): temporal indices of sampled frames.
            metadata (dict): additional metadata.
        """
        if self.mode in ["train", "val"]:
            temporal_sample_index = -1  # Random sampling
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index] //
                self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] %
                 self.cfg.TEST.NUM_SPATIAL_CROPS)
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale = max_scale = crop_size = self.cfg.DATA.TEST_CROP_SIZE
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        # Try to load the video - Coronary base class
        for i_try in range(self._num_retries):
            try:
                import gc
                import time

                video_path = self._path_to_videos[index]
                start_frame, end_frame = self._frame_ranges[index]

                # Print debug info for slow loading
                start_time = time.time()
                result = self._decord_decode(video_path, start_frame, end_frame)
                load_time = time.time() - start_time

                if result is None:
                    raise ValueError(f"Failed to decode video at {video_path}")

                frames, frame_indices = result

                if load_time > 10:
                    print(f"Warning: Slow video loading detected: {video_path} took {load_time:.2f}s")

                if len(frames) == 0:
                    raise ValueError(f"Failed to decode video at {video_path}")

                # Use all loaded frames (sampling done in __getitem__ for Coronary_multitask)
                frames_tensor = torch.from_numpy(frames).float()  # (T, H, W, 3)
                frames_tensor = frames_tensor / 255.0  # Normalize to [0, 1]

                # Delete original frames to free memory
                del frames
                gc.collect()

                # Apply data augmentation for training
                if self.aug and self.mode == "train":
                    # Apply random horizontal flip (NHWC format)
                    if random.random() > 0.5:
                        frames_tensor = torch.flip(frames_tensor, dims=[2])

                # Permute to (C, T, H, W) format for tensor_normalize and pack_pathway_output
                # This follows the standard SlowFast pipeline (see kinetics.py)
                frames_tensor = frames_tensor.permute(3, 0, 1, 2)

                # Normalize with dataset mean/std (expects C in first dimension)
                frames_tensor = self._tensor_normalize(
                    frames_tensor,
                    self.cfg.DATA.MEAN,
                    self.cfg.DATA.STD
                )

                # Apply spatial sampling (crop/resize)
                # spatial_sampling expects (N, C, H, W) where N=num_frames
                # But works with (C, T, H, W) as the underlying ops use shape[2] and shape[3] for H, W
                frames_tensor = self._spatial_sampling(
                    frames_tensor,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                )

                # Pack pathway output (for SlowFast models with multiple pathways)
                frames_out = self._pack_pathway_output(frames_tensor)

                label = self._labels[index]
                time_idx = np.array(frame_indices)

                return frames_out, label, index, time_idx, self._video_meta[index]

            except Exception as e:
                print(f"Failed to load video idx {index} (trial {i_try}): {e}")
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

        # If all retries failed
        raise RuntimeError(
            f"Failed to fetch video after {self._num_retries} retries."
        )

    def _spatial_sampling(self, frames, spatial_idx, min_scale, max_scale, crop_size):
        """
        Perform spatial sampling on video frames.

        Args:
            frames: Tensor of shape (C, T, H, W)
            spatial_idx: Spatial crop index (-1 for random)
            min_scale: Minimum scale
            max_scale: Maximum scale
            crop_size: Target crop size

        Returns:
            Cropped and resized frames
        """
        from . import utils as utils
        return utils.spatial_sampling(
            frames,
            spatial_idx=spatial_idx,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=False,  # Already applied above
            inverse_uniform_sampling=False,
            aspect_ratio=None,
            scale=None,
            motion_shift=False,
        )

    def _tensor_normalize(self, tensor, mean, std):
        """
        Normalize tensor with given mean and std.
        """
        from . import utils as utils
        return utils.tensor_normalize(tensor, mean, std)

    def _pack_pathway_output(self, frames):
        """
        Pack output frames into pathway format.
        For single pathway models, returns as-is.
        For SlowFast, returns [slow_pathway, fast_pathway].
        """
        from . import utils as utils
        return utils.pack_pathway_output(self.cfg, frames)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def _set_epoch_num(self, epoch):
        """Set the current epoch number."""
        self._num_epoch = epoch


@DATASET_REGISTRY.register()
class Coronary_multitask(Coronary):
    """
    Coronary CT Angiography video loader for multi-task learning.

    Extends the base Coronary dataset to support:
    - Classification target: confidence score (0-1) for plaque presence
    - Regression target: normalized plaque percentage (0-1)

    The dataset expects:
    - A metadata CSV file with columns: ID, Start Frame, End Frame, Plaque, Branch, Split
    - DICOM files organized in ID-named subfolders under DATA.PATH_PREFIX
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Coronary multi-task dataset loader.

        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
            num_retries (int): number of retries for failed loading.
        """
        # Initialize multi-task specific attributes BEFORE calling parent __init__
        # because _construct_loader() (called by parent) needs these attributes
        self.plaque_norm_factor = cfg.CORONARY.PLAQUE_NORM_FACTOR
        self.num_proposals = cfg.CORONARY.NUM_PROPOSALS
        self.use_multi_token = cfg.CORONARY.USE_MULTI_TOKEN

        super().__init__(cfg, mode, num_retries)

    def _construct_loader(self):
        """
        Construct the video loader from metadata CSV.

        The CSV format is:
        ID,Plaque,Branch,Split

        相同 ID 的行会被聚合为一个样本，每个样本包含多个目标（类似检测任务）。
        每个目标有 plaque 值和 branch 类别。
        """
        # Load metadata and filter by split
        metadata_path = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            "metadataV1.csv"  # Use metadataV1.csv with Split column
        )
        assert pathmgr.exists(metadata_path), \
            "{} not found".format(metadata_path)

        with pathmgr.open(metadata_path, "r") as f:
            df = pd.read_csv(f)

        # Filter by split column (train/val/test)
        split_df = df[df['Split'] == self.mode.capitalize()].reset_index(drop=True)

        # 按 ID 聚合相同样本
        grouped = split_df.groupby('ID')

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._frame_ranges = []  # Store (start, end) for each video
        self._video_meta = {}

        # Multi-task targets - 每个样本包含多个目标
        self._cls_targets = []  # Classification targets (类别索引：0-3 为分支，4 为背景，-1 为 padding)
        self._reg_targets = []  # Regression targets (每个目标的 normalized plaque)

        # Branch 类别映射（用于将字符串转换为类别索引）
        # 类别 0-3: LAD, RCA, LCX, LM; 类别 4: 背景 (无斑块)
        self.branch2idx = {'LAD': 0, 'RCA': 1, 'LCX': 2, 'LM': 3}
        self.idx2branch = {v: k for k, v in self.branch2idx.items()}
        self.num_branches = len(self.branch2idx)
        self.bg_class = self.num_branches  # 背景类索引 = 4

        for clip_idx, (video_id, group_df) in enumerate(grouped):
            video_id = str(int(video_id))
            # Start Frame and End Frame columns may not exist in metadataV1.csv
            # Use 0 as default if columns don't exist or values are missing
            if 'Start Frame' in group_df.columns:
                start_frame = int(group_df['Start Frame'].iloc[0]) if not pd.isna(group_df['Start Frame'].iloc[0]) else 0
            else:
                start_frame = 0
            if 'End Frame' in group_df.columns:
                end_frame = int(group_df['End Frame'].iloc[0]) if not pd.isna(group_df['End Frame'].iloc[0]) else 0
            else:
                end_frame = 0

            # 获取所有 plaque 和 branch（同一个 ID 可能有多个斑块目标）
            plaques = group_df['Plaque'].fillna(0).tolist()
            branches = group_df['Branch'].fillna('Unknown').tolist()
            num_targets = len(plaques)

            # Normalize plaque to 0-1 range (每个目标独立归一化)
            plaques_normalized = [min(max(p / self.plaque_norm_factor, 0.0), 1.0) for p in plaques]

            # Classification target: 类别索引 (0-3 为分支，4 为背景/无斑块)
            # 对于 plaque > 0 的目标，使用 branch 类别；否则为背景类
            cls_targets = []
            for plaque, branch in zip(plaques, branches):
                if plaque > 0 and branch in self.branch2idx:
                    cls_targets.append(self.branch2idx[branch])  # 0-3
                else:
                    cls_targets.append(self.bg_class)  # 4 (背景)

            for idx in range(self._num_clips):
                video_path = os.path.join(self.cfg.DATA.PATH_PREFIX, video_id)
                self._path_to_videos.append(video_path)

                # For multi-task, store targets as lists (one per target)
                self._cls_targets.append(cls_targets)  # List[int] (class indices: 0-4)
                self._reg_targets.append(plaques_normalized)  # List[float]

                # Legacy label field (使用 plaque 列表，会在 __getitem__ 中处理)
                self._labels.append(plaques_normalized)

                self._spatial_temporal_idx.append(idx)
                self._frame_ranges.append((start_frame, end_frame))
                # 只保留必要的 metadata 字段（去掉会导致错误的字符串列表）
                self._video_meta[clip_idx * self._num_clips + idx] = {
                    'video_id': video_id,
                    'num_targets': num_targets,  # 目标数量
                }

        assert len(self._path_to_videos) > 0, \
            "Failed to load Coronary split {} from {}".format(self.mode, metadata_path)

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, labels, and video index.

        For multi-task learning, returns:
        - frames: video frames tensor
        - label: legacy label (regression target)
        - index: video index
        - time_idx: temporal indices
        - metadata: dict with cls_target, reg_target, etc.
        """
        if self.mode in ["train", "val"]:
            temporal_sample_index = -1  # Random sampling
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index] //
                self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] %
                 self.cfg.TEST.NUM_SPATIAL_CROPS)
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale = max_scale = crop_size = self.cfg.DATA.TEST_CROP_SIZE
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        # Try to load the video
        for i_try in range(self._num_retries):
            try:
                import gc
                import time

                video_path = self._path_to_videos[index]
                start_frame, end_frame = self._frame_ranges[index]

                # Print debug info for slow loading
                start_time = time.time()
                result = self._decord_decode(video_path, start_frame, end_frame)
                load_time = time.time() - start_time

                if result is None:
                    raise ValueError(f"Failed to decode video at {video_path}")

                frames, loaded_frame_indices = result

                if load_time > 10:
                    print(f"Warning: Slow video loading detected: {video_path} took {load_time:.2f}s")

                if len(frames) == 0:
                    raise ValueError(f"Failed to decode video at {video_path}")

                # Sample frames according to cfg.DATA.NUM_FRAMES and SAMPLING_RATE
                num_frames_needed = self.cfg.DATA.NUM_FRAMES
                sampling_rate = self.cfg.DATA.SAMPLING_RATE

                # Calculate total frames needed
                span = (num_frames_needed - 1) * sampling_rate + 1

                # If video is shorter than needed, just use all frames
                if len(frames) <= num_frames_needed:
                    sampled_frame_indices = list(range(len(frames)))
                    # Pad if necessary
                    while len(sampled_frame_indices) < num_frames_needed:
                        sampled_frame_indices.append(sampled_frame_indices[-1])
                else:
                    # Random temporal sampling for training
                    if self.mode == "train" and len(frames) > span:
                        start_idx = random.randint(0, len(frames) - span)
                    else:
                        # Center crop for validation/testing
                        start_idx = max(0, (len(frames) - span) // 2)

                    # Sample frames with specified rate
                    sampled_frame_indices = [
                        start_idx + i * sampling_rate
                        for i in range(num_frames_needed)
                    ]
                    sampled_frame_indices = [min(idx, len(frames) - 1) for idx in sampled_frame_indices]

                # Map sampled indices back to original frame indices
                time_idx = np.array([loaded_frame_indices[i] for i in sampled_frame_indices])

                sampled_frames = frames[sampled_frame_indices]  # (T, H, W, 3)

                # Delete original frames to free memory
                del frames
                gc.collect()

                # Convert to tensor and normalize
                frames_tensor = torch.from_numpy(sampled_frames).float()  # (T, H, W, 3)
                frames_tensor = frames_tensor / 255.0  # Normalize to [0, 1]

                # Delete sampled frames to free memory
                del sampled_frames
                gc.collect()

                # Apply data augmentation for training (NHWC format)
                if self.aug and self.mode == "train":
                    # Apply random horizontal flip
                    if random.random() > 0.5:
                        frames_tensor = torch.flip(frames_tensor, dims=[2])

                # Permute to (C, T, H, W) format for tensor_normalize and pack_pathway_output
                # This follows the standard SlowFast pipeline (see kinetics.py)
                # frames_tensor = frames_tensor.permute(3, 0, 1, 2)

                # Normalize with dataset mean/std (expects C in first dimension)
                frames_tensor = self._tensor_normalize(
                    frames_tensor,
                    self.cfg.DATA.MEAN,
                    self.cfg.DATA.STD
                )

                # Apply spatial sampling (crop/resize)
                # spatial_sampling works with (T, C, H, W) as underlying ops use shape[2:] for H, W
                frames_tensor = frames_tensor.permute(0, 3, 1, 2)
                frames_tensor = self._spatial_sampling(
                    frames_tensor,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                )

                # Pack pathway output (for SlowFast models with multiple pathways)
                frames_out = self._pack_pathway_output(frames_tensor.permute(1, 0, 2, 3))

                # Prepare multi-task targets (每个样本包含多个目标)
                # cls_targets: List[int] (class indices: 0-3 for branches, 4 for background) -> Tensor
                # reg_targets: List[float] -> Tensor
                cls_targets = torch.tensor(self._cls_targets[index], dtype=torch.long)
                reg_targets = torch.tensor(self._reg_targets[index], dtype=torch.float32)

                # 统一 padding 到 num_proposals（无论 use_multi_token 是否为 True）
                # 这样确保 batch 中所有样本的维度一致
                num_targets = len(cls_targets)
                if num_targets < self.num_proposals:
                    # Pad with -1 (padding class, ignored in loss)
                    pad_size = self.num_proposals - num_targets
                    cls_targets = torch.cat([cls_targets, torch.full((pad_size,), 4, dtype=cls_targets.dtype)])
                    reg_targets = torch.cat([reg_targets, torch.zeros(pad_size, dtype=reg_targets.dtype)])
                elif num_targets > self.num_proposals:
                    # 如果目标数量多于 proposal 数量，截断
                    cls_targets = cls_targets[:self.num_proposals]
                    reg_targets = reg_targets[:self.num_proposals]

                # 生成有效掩码（1 表示真实目标，0 表示 padding）
                # padding 目标的 cls_target 为 4
                valid_mask = (cls_targets < 4).float()

                # 将 padding 的类别索引替换为 0（避免 CrossEntropyLoss 报错）
                # 实际损失计算会被 valid_mask 过滤
                cls_targets_padded = cls_targets.clone()
                # cls_targets_padded[cls_targets < 0] = 0

                # Pack metadata with targets
                # 包含 video_id 用于调试和结果记录（不会转移到 GPU）
                metadata = {
                    'video_id': self._video_meta[index]['video_id'],
                    'cls_target': cls_targets_padded,
                    'reg_target': reg_targets,
                    'valid_mask': valid_mask,  # (num_proposals,)
                }

                # Return frames, legacy label (已弃用，使用 reg_target), index, time_idx, metadata
                # 为了保持接口一致，label 也返回 padding 后的 reg_target
                return frames_out, reg_targets, index, time_idx, metadata

            except Exception as e:
                print(f"Failed to load video idx {index} (trial {i_try}): {e}")
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

        # If all retries failed
        raise RuntimeError(
            f"Failed to fetch video after {self._num_retries} retries."
        )
