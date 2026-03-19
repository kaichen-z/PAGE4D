from __future__ import annotations
import sys
import os
import glob
import random
import re
import math
from typing import Any, Dict, List, Optional
import pdb
import cv2
import numpy as np
import torch
from ..base_dataset import BaseDataset
from ..dataset_util import (
    get_stride_distribution,
    read_image_cv2,
    threshold_depth_map,
    create_tracking_grid,
    apply_flow_transforms_simple,
    apply_mask_transforms_simple,
    track_points_with_flow_and_occlusion,
)

test_seq = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]

T_m = np.array([
 [ 1.0157,  0.1828, -0.2389,  0.0113],
 [ 0.0009, -0.8431, -0.6413, -0.0098],
 [-0.3009,  0.6147, -0.8085,  0.0111],
 [ 0.0,     0.0,     0.0,     1.0   ]], dtype=np.float32)

T_ROS = np.array([
 [-1, 0, 0, 0],
 [ 0, 0, 1, 0],
 [ 0, 1, 0, 0],
 [ 0, 0, 0, 1]], dtype=np.float32)

def _quat_to_rotmat(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    # Normalize quaternion
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]])
    return R

def to_homogeneous(proc_extri):
    if proc_extri.shape != (3,4):
        raise ValueError(f"Expected (3,4), got {proc_extri.shape}")
    bottom = np.array([[0, 0, 0, 1]], dtype=proc_extri.dtype)
    extri44 = np.vstack([proc_extri, bottom])
    return extri44

def check_coord_system(points_3d_world, extrinsics):
    N = points_3d_world.shape[0]
    ones = np.ones((N,1))
    pts_w_h = np.hstack([points_3d_world, ones])  # (N,4)
    pts_c_h = (extrinsics @ pts_w_h.T).T
    z = pts_c_h[:,2]
    mean_z = np.mean(z)
    pos_ratio = np.mean(z > 0)
    if pos_ratio > 0.9:
        return "opencv (+Z forward)", mean_z, pos_ratio
    elif pos_ratio < 0.1:
        return "opengl (-Z forward)", mean_z, pos_ratio
    else:
        return "uncertain (mixed)", mean_z, pos_ratio

def matrix_check(image, depth, K, T_cw, eps=1e-6):
    """
    image: h, w, 3 ||  depth: h, w || K: 3, 3 || T_cw: 4, 4 world to camera
    """
    def backproject_to_world(uvz, K, T_cw):
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        u,v,z = uvz[:,0], uvz[:,1], uvz[:,2]
        Xc = np.stack([(u-cx)/fx * z, (v-cy)/fy * z, z, np.ones_like(z)], axis=1)  # (N,4)
        C2W = np.linalg.inv(T_cw)
        Xw = (C2W @ Xc.T).T[:, :3]
        return Xw
    def project_to_pixels(Xw, K, T_cw):
        Xw_h = np.hstack([Xw, np.ones((Xw.shape[0],1))])
        Xc = (T_cw @ Xw_h.T).T[:, :3]
        x, y, z = Xc[:,0], Xc[:,1], Xc[:,2]
        u = K[0,0]*x/z + K[0,2]
        v = K[1,1]*y/z + K[1,2]
        return np.stack([u,v],1)
    # 保证 T_cw 是 4x4
    T_cw = np.asarray(T_cw, dtype=np.float32)
    if T_cw.shape == (3,4):
        T_cw = np.vstack([T_cw, np.array([0,0,0,1], dtype=np.float32)])
    elif T_cw.shape == (3,3):
        T_cw = np.vstack([np.hstack([T_cw, np.zeros((3,1), dtype=np.float32)]),
                          np.array([0,0,0,1], dtype=np.float32)])
    elif T_cw.shape != (4,4):
        raise ValueError(f"T_cw must be 3x4, 3x3 or 4x4, got {T_cw.shape}")
    H,W = image.shape[:2]
    ys = np.linspace(16, H-16, 50).astype(int)
    xs = np.linspace(16, W-16, 50).astype(int)
    uu, vv = np.meshgrid(xs, ys)
    zz = depth[vv, uu]
    # 调试信息
    valid_mask = np.isfinite(zz) & (zz > eps)
    uu = uu[valid_mask]
    vv = vv[valid_mask]
    zz = zz[valid_mask]
    # 检查是否有有效深度值
    # if np.sum(valid_mask) == 0:
    #     return False
    uvz = np.stack([uu.ravel(), vv.ravel(), zz.ravel()], axis=1).astype(np.float32)
    Xw = backproject_to_world(uvz, K, T_cw)
    uv_hat = project_to_pixels(Xw, K, T_cw)
    err = np.linalg.norm(uv_hat - uvz[:,:2], axis=1)
    # 检查err数组是否为空
    # if len(err) == 0:
    #     print("Warning: No valid reprojection errors calculated ==========================")
    #     return False
    print("reproj err px  mean/med/max:", err.mean(), np.median(err), err.max())
    return err.max()

def check_nan_values(proc_img, proc_depth, proc_intri, proc_extri):
    """
    Check for NaN values in processed image, depth, intrinsics, and extrinsics.
    Args:
        proc_img: torch.Tensor or np.ndarray
        proc_depth: torch.Tensor or np.ndarray
        proc_intri: torch.Tensor or np.ndarray
        proc_extri: torch.Tensor or np.ndarray
    Returns:
        bool: True if any NaN values found, False otherwise
    """
    has_nan = False
    def _check(name, arr):
        nonlocal has_nan
        if arr is None:
            return
        if isinstance(arr, torch.Tensor):
            if torch.isnan(arr).any():
                print(f"Warning: NaN found in {name}")
                has_nan = True
        elif isinstance(arr, np.ndarray):
            if np.isnan(arr).any():
                print(f"Warning: NaN found in {name}")
                has_nan = True
        else:
            # fallback: try numpy conversion
            try:
                if np.isnan(np.array(arr)).any():
                    print(f"Warning: NaN found in {name}")
                    has_nan = True
            except Exception:
                pass
    _check("proc_img", proc_img)
    _check("proc_depth", proc_depth)
    _check("proc_intri", proc_intri)
    _check("proc_extri", proc_extri)
    return has_nan

class BonnDataset(BaseDataset):
    """Bonn RGBD dataset loader for VGGT.
    Each *sequence* is treated as an independent video. During training /
    evaluation we sample *clips* consisting of two or more frames with a
    configurable temporal stride.
    Dataset structure:
    - RGB images: rgbd_bonn_dataset/*/rgb/*.png
    - Depth maps: rgbd_bonn_dataset/*/depth/*.png  
    - Camera poses: rgbd_bonn_dataset/*/groundtruth.txt
    """
    def __init__(
        self,
        common_conf,
        dataset_location: str = "/workspace/data/kaichen/data/test/bonn/rgbd_bonn_dataset",
        split: str = "val",
        sequence_names: Optional[List[str]] = None,
        strides: Optional[List[int]] = [1],
        clip_step: int = 1,
        min_num_images: int = 8,
        len_train: int = 1000,
        len_test: int = 240,
        verbose: bool = True,
        dist_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(common_conf=common_conf)
        self.dataset_location = dataset_location
        self.split = split
        # If `strides` is None or empty, we will return *full sequences* instead of short clips.
        if strides is None or len(strides) == 0:
            self.strides: List[int] = []
            self._clip_mode = False  # full‐sequence mode
        else:
            self.strides = list(strides)
            self._clip_mode = True
        self.clip_step = clip_step
        self.min_num_images = min_num_images
        self.verbose = verbose
        # Discover sequences
        if sequence_names is None:
            self.sequence_names = self._discover_sequences()
        else:
            self.sequence_names = sequence_names
        if self.split == "test" or self.split == "val":
            self.sequence_names = ["rgbd_bonn_" + i for i in test_seq]
        else: # split = "else"
            raise ValueError(f"Invalid split: {self.split}")
        # Build per-sequence metadata and clips
        self._build_metadata()
        self.len_train = len_train if split == "train" else len_test
        # Apply stride distribution if specified
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)
        print(
            f"Loaded {len(self.sequence_names)} Bonn sequences – "
            f"{len(self.clip_data)} clips in total.")
        self.ENABLE_TRACK = False

    # Sequence discovery & metadata
    def _discover_sequences(self) -> List[str]:
        """Discover available Bonn sequences."""
        if not os.path.exists(self.dataset_location):
            raise FileNotFoundError(f"Dataset location not found: {self.dataset_location}")
        seqs = [d for d in os.listdir(self.dataset_location)
            if os.path.isdir(os.path.join(self.dataset_location, d))]
        seqs.sort()
        if self.verbose:
            print("[Bonn] Discovered sequences:", seqs)
        return seqs

    def _build_metadata(self):
        """Populate self.sequence_metadata + self.clip_data."""
        # Load sequences and build metadata
        self._load_sequences()

    def _load_sequences(self): 
        # Loading only based on initial 110.
        self.sequence_list = []
        self.sequence_metadata = {}
        self.clip_data = []
        for seq_name in self.sequence_names:
            seq_dir   = os.path.join(self.dataset_location, seq_name)
            rgb_dir   = os.path.join(seq_dir, "rgb")
            depth_dir = os.path.join(seq_dir, "depth")
            gt_path   = os.path.join(seq_dir, "groundtruth.txt")
            # 1) read *.png and sort (name-sorted == time order in Bonn)
            rgb_files   = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
            depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
            # 2) enforce 1:1 pairing and cap to first 110
            rgb_files   = rgb_files[30:140]
            depth_files = depth_files[30:140]
            # 3) read GT (keep rows as-is; take the first n rows)
            pose_data =  np.loadtxt(gt_path)
            pose_data = pose_data[30:140]
            num_frames = 110
            frame_indices = list(range(110))
            self.sequence_list.append(seq_name)
            self.sequence_metadata[seq_name] = {
                "num_frames": num_frames,
                "frame_indices": frame_indices,
                "rgb_files": rgb_files,
                "depth_files": depth_files,
                "gt_rows": pose_data,}
            self._generate_clips_for_sequence(seq_name, num_frames, frame_indices)

    def _generate_clips_for_sequence(self, seq_name: str, num_frames: int, frame_indices: List[int]):
        """Generate training clips for a sequence with different strides."""
        if not self._clip_mode:
            # Full-sequence mode: one entry per sequence containing *all* frames
            self.clip_data.append({
                    "seq_name": seq_name,
                    "frame_indices": frame_indices,
                    "stride": None,})
        else:
            # Generate clip metadata with different strides
            for stride in self.strides:
                # 确保stride不超过可用帧数
                if stride >= num_frames:
                    continue
                max_start = num_frames - 1 - stride  # need at least two frames
                if max_start < 0:
                    continue
                for start in range(0, max_start + 1, self.clip_step):
                    fid0 = frame_indices[start]
                    fid1 = frame_indices[start + stride]
                    self.clip_data.append(
                        {
                            "seq_name": seq_name,
                            "frame_indices": [fid0, fid1],
                            "stride": stride,})

    def _resample_clips(self, strides: List[int], dist_type: str):
        """Resample clips according to stride distribution."""
        # Group clips by stride
        stride_clips = {stride: [] for stride in strides}
        for i, clip in enumerate(self.clip_data):
            stride_clips[clip['stride']].append(i)
        # Get stride distribution
        dist = get_stride_distribution(strides, dist_type=dist_type)
        dist = dist / np.max(dist)
        # Calculate number of clips per stride
        max_clips = max(len(clips) for clips in stride_clips.values())
        clips_per_stride = [min(len(stride_clips[stride]), int(dist[i] * max_clips)) 
                           for i, stride in enumerate(strides)]
        if self.verbose:
            print(f'Resampled clips per stride: {dict(zip(strides, clips_per_stride))}')
        # Resample clips
        resampled_indices = []
        for i, stride in enumerate(strides):
            available_clips = stride_clips[stride]
            if len(available_clips) > 0:
                selected = np.random.choice(
                    available_clips, 
                    size=min(clips_per_stride[i], len(available_clips)), 
                    replace=False)
                resampled_indices.extend(selected)
        # Update clip data
        self.clip_data = [self.clip_data[i] for i in resampled_indices]

    def _contiguous_window_impr(self, available_frames, base_frames, want):
        avail = sorted(set(available_frames))
        if want <= 0 or not avail:
            return []
        # 允许 base 为空：直接从头取 want 个（如果不够就重复最后一个）
        if not base_frames:
            if len(avail) >= want:
                return avail[:want]
            # 不够则重复最后一个
            return avail + [avail[-1]] * (want - len(avail))
        base = sorted(set(base_frames))
        # ---- 推断 stride（与原来一致）----
        if len(base) >= 2:
            diffs = [abs(b - base[0]) for b in base[1:] if b != base[0]]
            stride = diffs[0] if len(diffs) == 1 else (math.gcd(*diffs) if diffs else 1)
            if stride <= 0:
                stride = 1
        else:
            stride = 1
        # ---- 过滤到同余子序列 ----
        anchor = base[0]
        lane = [f for f in avail if (f - anchor) % stride == 0]
        if not lane:
            # 回退：不用同余过滤，直接从 avail 取固定长度，保证不报错
            if len(avail) >= want:
                return avail[:want]
            return avail + [avail[-1]] * (want - len(avail))
        # ---- 在 lane 上找覆盖 base 的最小窗口 ----
        pos = {f: i for i, f in enumerate(lane)}
        idxs = [pos[f] for f in base if f in pos]
        if not idxs:
            # base 不在 lane（极端），直接固定长度
            if len(lane) >= want:
                return lane[:want]
            return lane + [lane[-1]] * (want - len(lane))
        l, r = min(idxs), max(idxs)
        # 如果覆盖范围超过 want：截尾
        if (r - l + 1) > want:
            l = r - want + 1
            return lane[l:r+1]
        # 否则先右后左补
        need = want - (r - l + 1)
        add_r = min(need, len(lane) - 1 - r)
        r += add_r
        need -= add_r
        add_l = min(need, l)
        l -= add_l
        window = lane[l:r+1]
        Flag = True
        # 仍不足的情况下，用"重复边界"填满（保证固定长度）
        if len(window) < want:
            window = window + [window[-1]] * (want - len(window))
            Flag = False
        return window, Flag

    def _load_camera_poses(self, seq_meta, frame_indices):
        poses_T_c_w = []
        for fid in frame_indices:
            row = seq_meta["gt_rows"][fid].astype(np.float64)   # [ts tx ty tz qx qy qz qw]
            tx, ty, tz = row[1:4]; qx, qy, qz, qw = row[4:8]
            # quat → R
            R = _quat_to_rotmat(qw, qx, qy, qz)
            T_w_m = np.eye(4, dtype=np.float64)
            T_w_m[:3,:3] = R
            T_w_m[:3, 3] = [tx, ty, tz]
            # ROS bug & marker→sensor
            T_w_mocap = np.linalg.inv(T_ROS) @ T_w_m @ T_ROS
            T_w_s     = T_w_mocap @ T_m
            T_c_w     = np.linalg.inv(T_w_s).astype(np.float32)
            poses_T_c_w.append(T_c_w)
        return poses_T_c_w

    def _estimate_intrinsics(self, image_shape):
        fx, fy = 542.822841, 542.576870
        cx, cy = 315.593520, 237.756098
        K = np.array([[fx, 0, cx],
                    [0,  fy, cy],
                    [0,  0,  1]], dtype=np.float32)
        dist = np.array([0.039903, -0.099343, -0.000730, -0.000144, 0.0], dtype=np.float32)
        return K, dist

    # Public helpers (Dataset API)
    def get_data(
        self, seq_index: int | None = None, img_per_seq: int | None = None,
        seq_name: str | None = None, ids: List[int] | None = None,
        aspect_ratio: float = 1.0,) -> Dict[str, Any]:
        """Return one clip worth of data (images, depth, camera, points, …)."""
        # -------------------- Select clip -------------------------------
        if seq_index is None and seq_name is None:
            seq_index = random.randint(0, len(self.clip_data) - 1)
        if seq_name is None:
            clip = self.clip_data[seq_index]
        else:
            # If the caller explicitly requested a sequence + ids
            if ids is None:
                raise ValueError("When seq_name is provided, ids must also be set.")
            clip = {"seq_name": seq_name, "frame_indices": ids, "stride": None}
        seq_name = clip["seq_name"]
        base_frame_indices = clip["frame_indices"]
        seq_metadata = self.sequence_metadata[seq_name]
        available_frames = seq_metadata["frame_indices"]
        # Handle img_per_seq parameter - extend frame indices if needed
        frame_indices, _ = self._contiguous_window_impr(available_frames, base_frame_indices, img_per_seq)
        target_shape = self.get_target_shape(aspect_ratio)
        # Load camera poses
        poses = self._load_camera_poses(seq_metadata, frame_indices)

        original_images = []
        for i, fid in enumerate(frame_indices):
            # Find the corresponding file path for this frame index
            rgb_path = seq_metadata["rgb_files"][fid]
            image = read_image_cv2(rgb_path)
            if image is None:
                raise FileNotFoundError(rgb_path)
            original_images.append(image)
        # -------------------- Load depth maps -------------------------
        depths = []
        for i, fid in enumerate(frame_indices):
            # Find the corresponding file path for this frame index
            depth_path = seq_metadata["depth_files"][fid]
            if os.path.exists(depth_path):
                depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_map is not None:
                    # Convert to float and scale if needed
                    if depth_map.dtype == np.uint16:
                        depth_map = depth_map.astype(np.float32) / 5000.0  # Convert mm to meters
                    else:
                        depth_map = depth_map.astype(np.float32)
                    depth_map = threshold_depth_map(depth_map, max_percentile=98)
                else:
                    depth_map = np.zeros(original_images[0].shape[:2], dtype=np.float32)
            else:
                depth_map = np.zeros(original_images[0].shape[:2], dtype=np.float32)
            depths.append(depth_map)
        # -------------------- Process Images -------------------------
        images, processed_depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points, point_masks = [], [], []
        processed_tracks = []; processed_track_masks = []
        for i, fid in enumerate(frame_indices):
            # Get pose for this frame
            pose = poses[i] if i < len(poses) else np.eye(4)
            # Get original image and depth for this frame
            original_image = original_images[i]
            depth_map = depths[i]
            original_size = np.array(original_image.shape[:2])  # (H, W)
            # Estimate intrinsics
            K, dist = self._estimate_intrinsics(original_image.shape)
            H, W = original_image.shape[:2]
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), alpha=0)
            map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (W, H), cv2.CV_32FC1)
            image_undist = cv2.remap(original_image, map1, map2, cv2.INTER_LINEAR)
            depth_undist = cv2.remap(depth_map, map1, map2, cv2.INTER_NEAREST)
            original_size = np.array(image_undist.shape[:2])  # (H, W)
            # Process one image
            (   proc_img,
                proc_depth,
                proc_extri,
                proc_intri,
                world_pts,
                cam_pts,
                pt_mask,
                processed_track,
                processed_track_mask,
            ) = self.process_one_image(
                image_undist,
                depth_undist,
                pose[:3, :],  # 3x4 extrinsic matrix
                newK,
                original_size,
                target_shape,
                track=None,  # No tracking for now
                filepath=f"frame_{fid:04d}",)
            # print(check_nan_values(world_pts, pt_mask, proc_extri, proc_extri), '===========check_nan_values===========')
            # check = check_coord_system(world_pts.reshape(-1, 3)[pt_mask.reshape(-1)], proc_extri)
            # print(check, '===========check===========')
            # try:
            #     err_max = matrix_check(proc_img, proc_depth, proc_intri, proc_extri)
            #     if err_max > 1e-4:
            #         print(seq_name, '===========matrix_check===========')
            #         with open('/workspace/code/12_4d/VGGT-4D_T/training/fail.txt', 'a') as f:
            #             f.write(f"{seq_name}\n")
            # except:
            #     print(seq_name, '===========matrix_check===========')
            #     with open('/workspace/code/12_4d/VGGT-4D_T/training/fail.txt', 'a') as f:
            #         f.write(f"{seq_name}\n")

            images.append(proc_img)
            processed_depths.append(proc_depth)
            extrinsics.append(to_homogeneous(proc_extri))
            intrinsics.append(proc_intri)
            world_points.append(world_pts)
            cam_points.append(cam_pts)
            point_masks.append(pt_mask)
            processed_tracks.append(processed_track)
            processed_track_masks.append(processed_track_mask)
        return {
            "seq_name": f"bonn_{seq_name}",
            "ids": np.array(frame_indices, dtype=np.int32),
            "frame_num": len(extrinsics),
            "images": images,
            "depths": processed_depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "abandon_pose": False,
            "cam_points": cam_points,
            "world_points": world_points,
            "abandon_geometry": False,
            "point_masks": point_masks,
            "tracks": np.zeros((len(images), self.track_num, 2)),
            "track_masks": np.zeros((len(images), self.track_num)),
            "temporal_features": self._compute_temporal_features(frame_indices),
        }

    def _compute_temporal_features(self, frame_indices: List[int]) -> np.ndarray:
        """
        Compute temporal features for the given frame indices.
        Args: frame_indices: List of frame indices (e.g., [0, 100, 200, 300])
        Returns: np.ndarray: Normalized temporal features for each frame in [-1, 1] range
        """
        if not frame_indices:
            return np.array([])
        frame_indices_array = np.array(frame_indices)
        if len(frame_indices_array) > 1:
            min_frame = frame_indices_array.min()
            max_frame = frame_indices_array.max()
            frame_range = max_frame - min_frame
            if frame_range > 0:
                # Normalize to [-1, 1] range
                temporal_features = 2.0 * (frame_indices_array - min_frame) / frame_range - 1.0
            else:
                # All frames are the same, set to 0.0 (center of [-1, 1])
                temporal_features = np.full_like(frame_indices_array, 0.0, dtype=np.float32)
        else:
            # Only one frame available, set to 0.0 (center of [-1, 1])
            temporal_features = np.full_like(frame_indices_array, 0.0, dtype=np.float32)
        return temporal_features.astype(np.float32)

    def __len__(self):
        # In full-sequence mode each entry in clip_data corresponds 1-to-1 with a sequence,
        # otherwise it is number-of-clips.
        return len(self.clip_data)