from __future__ import annotations
import sys

# Directory layout assumed (default `dataset_location=/shared/ssd_30T/gaspard/sintel-data`):
sdk_path = "/workspace/data/kaichen/data/test/Sintel/MPI-Sintel-depth-training-20150305/sdk/python"
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)
import os
import glob
import random
import re
import sys
from typing import Any, Dict, List, Optional
import pdb
import cv2
import numpy as np

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


try:
    # The helper lives at <dataset_root>/depth-training/sdk/python/sintel_io.py
    # We locate it dynamically so that the dataset is portable.
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except Exception:
    _THIS_DIR = ""

test_list = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2", "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]

def to_homogeneous(proc_extri):
    if proc_extri.shape != (3,4):
        raise ValueError(f"Expected (3,4), got {proc_extri.shape}")
    bottom = np.array([[0, 0, 0, 1]], dtype=proc_extri.dtype)
    extri44 = np.vstack([proc_extri, bottom])
    return extri44

def _import_sintel_io(dataset_location: str):
    sdk_path = os.path.join(dataset_location, "MPI-Sintel-depth-training-20150305", "sdk", "python")
    if sdk_path not in sys.path:
        sys.path.append(sdk_path)
    try:
        import sintel_io  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"Could not import sintel_io from '{sdk_path}'. Make sure the MPI-Sintel "
            "SDK is present inside the dataset directory."
        ) from exc
    return sintel_io

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
    # 新增：只保留有效深度
    valid_mask = np.isfinite(zz) & (zz > eps)
    uu = uu[valid_mask]
    vv = vv[valid_mask]
    zz = zz[valid_mask]
    uvz = np.stack([uu.ravel(), vv.ravel(), zz.ravel()], axis=1).astype(np.float32)
    Xw = backproject_to_world(uvz, K, T_cw)
    uv_hat = project_to_pixels(Xw, K, T_cw)
    err = np.linalg.norm(uv_hat - uvz[:,:2], axis=1)
    print("reproj err px  mean/med/max:", err.mean(), np.median(err), err.max())

class SintelDataset(BaseDataset):
    """MPI-Sintel training split loader for VGGT.

    Each *sequence* is treated as an independent video.  During training /
    evaluation we sample *clips* consisting of two or more frames with a
    configurable temporal stride.  This mirrors the behaviour of the existing
    Kubric / Odyssey loaders so that Sintel can be mixed seamlessly.
    """

    def __init__(
        self,
        common_conf,
        dataset_location: str = "/shared/ssd_30T/gaspard/sintel-data",
        quality: str = "final",  # one of {"final", "clean", "albedo"}
        split: str = "val",
        sequence_names: Optional[List[str]] = None,
        strides: Optional[List[int]] = [1, 2],
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
        self.quality = quality
        self.split = split  # train / val (only train exists for Sintel)
        # If `strides` is None or empty, we will return *full sequences* instead of short clips.
        if strides is None or len(strides) == 0:
            self.strides: List[int] = []
            self._clip_mode = False  # full‐sequence mode
        else:
            self.strides = list(strides)
            self._clip_mode = True
        self.clip_step = clip_step
        self.min_num_images = min_num_images
        self.verbose = False

        # Import sintel_io helper
        self._sintel_io = _import_sintel_io(dataset_location)

        # Discover sequences
        if sequence_names is None:
            self.sequence_names = self._discover_sequences()
        else:
            self.sequence_names = sequence_names
        
        if self.split == "test" or self.split == "val":
            self.sequence_names = test_list
        else:
            raise ValueError(f"Invalid split: {self.split}")
        # Build per-sequence metadata and clips
        self._build_metadata()

        self.len_train = len_train if split == "train" else len_test

        # Apply stride distribution if specified
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)

        if self.verbose:
            print(
                f"Loaded {len(self.sequence_names)} Sintel sequences – "
                f"{len(self.clip_data)} clips in total (quality={self.quality})."
            )

        self.ENABLE_TRACK = False

    # Sequence discovery & metadata
    def _discover_sequences(self) -> List[str]:
        images_root = os.path.join(
            self.dataset_location, "training", self.quality
        )
        seqs = [
            d
            for d in os.listdir(images_root)
            if os.path.isdir(os.path.join(images_root, d))
        ]
        seqs.sort()
        if self.verbose:
            print("[Sintel] Discovered sequences:", seqs)
        return seqs

    def _build_metadata(self):
        """Populate self.sequence_metadata + self.clip_data."""
        # Load sequences and build metadata
        self._load_sequences()

    def _load_sequences(self):
        """Load and organize Sintel sequences."""
        
        # Initialize storage
        self.sequence_list = []
        self.sequence_metadata = {}
        self.clip_data = []
        
        # Process each sequence
        for seq_name in self.sequence_names:
            img_dir = os.path.join(
                self.dataset_location, "MPI-Sintel-training_images", "training", self.quality, seq_name
            )
            
            if not os.path.exists(img_dir):
                if self.verbose:
                    print(f'[Sintel] Sequence directory does not exist: {img_dir}')
                continue
                
            if self.verbose:
                print(f'[Sintel] Processing sequence: {seq_name}')
                
            # Get available image files
            img_files = sorted(glob.glob(os.path.join(img_dir, "frame_*.png")))
            
            if len(img_files) < self.min_num_images:
                if self.verbose:
                    print(
                        f"[Sintel] Skipping {seq_name}: not enough frames ({len(img_files)})"
                    )
                continue

            # Extract frame indices from image files
            frame_indices = []
            for img_file in img_files:
                try:
                    # Use regex to extract frame number from filename (more robust than magic numbers)
                    match = re.search(r'frame_(\d+)\.png$', os.path.basename(img_file))
                    if match:
                        frame_idx = int(match.group(1))
                        frame_indices.append(frame_idx)
                    else:
                        if self.verbose:
                            print(f'[Sintel] Warning: Filename does not match expected pattern: {img_file}')
                        continue
                except (ValueError, AttributeError) as e:
                    if self.verbose:
                        print(f'[Sintel] Warning: Could not parse frame index from {img_file}: {e}')
                    continue
            
            frame_indices = sorted(frame_indices)
            num_frames = len(frame_indices)
            
            if num_frames < self.min_num_images:
                if self.verbose:
                    print(f'[Sintel] Skipping {seq_name}: insufficient valid frames ({num_frames})')
                continue
                
            # Store sequence metadata
            self.sequence_list.append(seq_name)
            self.sequence_metadata[seq_name] = {
                "img_dir": img_dir,
                "num_frames": num_frames,
                "frame_indices": frame_indices,
                "img_files": img_files
            }
            
            # Generate clips for this sequence
            self._generate_clips_for_sequence(seq_name, num_frames, frame_indices)
            
        if self.verbose:
            print(f'[Sintel] Successfully loaded {len(self.sequence_list)} sequences')

    def _generate_clips_for_sequence(self, seq_name: str, num_frames: int, frame_indices: List[int]):
        """Generate training clips for a sequence with different strides."""
        
        if not self._clip_mode:
            # Full-sequence mode: one entry per sequence containing *all* frames
            self.clip_data.append(
                {
                    "seq_name": seq_name,
                    "frame_indices": frame_indices,
                    "stride": None,
                }
            )
        else:
            # Generate clip metadata with different strides
            for stride in self.strides:
                max_start = num_frames - 1 - stride  # need at least two frames
                for start in range(0, max_start + 1, self.clip_step):
                    fid0 = frame_indices[start]
                    fid1 = frame_indices[start + stride]
                    self.clip_data.append(
                        {
                            "seq_name": seq_name,
                            "frame_indices": [fid0, fid1],
                            "stride": stride,
                        }
                    )

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
                    replace=False
                )
                resampled_indices.extend(selected)
                
        # Update clip data
        self.clip_data = [self.clip_data[i] for i in resampled_indices]

    def _contiguous_window(self, available_frames, base_frames, want):
        """
        在 available_frames 里找一个长度为 want 的连续窗口，
        覆盖 base_frames 的最小-最大范围；先尽量向右扩，不够再向左补。
        若整体帧数不足，就返回能覆盖的最长连续段。
        """
        avail = list(available_frames)
        pos = {f: i for i, f in enumerate(avail)}
        idxs = [pos[f] for f in base_frames if f in pos]
        if not idxs:
            # 兜底：返回最前面的连续 want 帧
            return avail[:min(want, len(avail))]
        l = min(idxs)
        r = max(idxs)
        # 如果 base 覆盖范围本身就比 want 长，截取其末尾的连续 want 帧（不打乱顺序）
        if (r - l + 1) > want:
            l = r - want + 1
            return avail[l:r + 1]
        need = want - (r - l + 1)
        # 先尽量向右扩
        add_r = min(need, len(avail) - 1 - r)
        r += add_r
        need -= add_r
        # 还不够再向左补
        add_l = min(need, l)
        l -= add_l
        # need 可能仍 > 0，表示总帧数不足；直接返回 [l, r]
        return avail[l:r + 1]

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
        # 仍不足的情况下，用“重复边界”填满（保证固定长度）
        if len(window) < want:
            window = window + [window[-1]] * (want - len(window))
            Flag = False
        return window, Flag

    # Public helpers (Dataset API)
    def get_data(
        self,
        seq_index: int | None = None,
        img_per_seq: int | None = None,
        seq_name: str | None = None,
        ids: List[int] | None = None,
        aspect_ratio: float = 1.0,
    ) -> Dict[str, Any]:
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
        if img_per_seq:
            frame_indices, Flag = self._contiguous_window_impr(available_frames, base_frame_indices, img_per_seq)
        else:
            frame_indices = base_frame_indices
        stride = frame_indices[1] - frame_indices[0]

        target_shape = self.get_target_shape(aspect_ratio)

        # Paths common to various modalities
        depth_root = os.path.join(
            self.dataset_location, "MPI-Sintel-depth-training-20150305", "training", "depth", seq_name
        )
        cam_root = os.path.join(
            self.dataset_location,
            "MPI-Sintel-depth-training-20150305",
            "training",
            "camdata_left",
            seq_name,
        )
        flow_root = os.path.join(
            self.dataset_location, "MPI-Sintel-training_extras", "training", "flow", seq_name
        )
        occlusion_root = os.path.join(
            self.dataset_location, "MPI-Sintel-training_extras", "training", "occlusions", seq_name
        )
        invalid_root = os.path.join(
            self.dataset_location, "MPI-Sintel-training_extras", "training", "invalid", seq_name
        )

        # -------------------- Load original images first -------------------------
        original_images = []
        for fid in frame_indices:
            img_path = os.path.join(
                self.sequence_metadata[seq_name]["img_dir"], f"frame_{fid:04d}.png"
            )
            image = read_image_cv2(img_path)
            if image is None:
                raise FileNotFoundError(img_path)
            original_images.append(image)

        # -------------------- Load flows in original coordinates -------------------------
        flows = []
        for fid in frame_indices[:-1]:
            flow_path = os.path.join(flow_root, f"frame_{fid:04d}.flo")
            if os.path.exists(flow_path):
                u, v = self._sintel_io.flow_read(flow_path)
                flow = np.stack([u, v], axis=-1).astype(np.float32)
            else:
                flow = None
                print(f"Flow not found for frame {fid}")
            flows.append(flow)
        # Pad to same length as images so consumers can index [i] safely.
        flows.append(None)

        # -------------------- Load occlusion/invalid masks in original coordinates -------------------------
        occlusion_masks = []
        invalid_masks = []
        
        for i, fid in enumerate(frame_indices[:-1]):
            # Load occlusion mask in original resolution
            occlusion_path = os.path.join(occlusion_root, f"frame_{fid:04d}.png")
            original_occlusion_mask = None
            if os.path.exists(occlusion_path):
                original_occlusion_mask = cv2.imread(occlusion_path, cv2.IMREAD_GRAYSCALE)
                original_occlusion_mask = (original_occlusion_mask > 0).astype(bool)

            # Load invalid mask in original resolution
            invalid_path = os.path.join(invalid_root, f"frame_{fid:04d}.png")
            original_invalid_mask = None
            if os.path.exists(invalid_path):
                original_invalid_mask = cv2.imread(invalid_path, cv2.IMREAD_GRAYSCALE)
                original_invalid_mask = (original_invalid_mask > 0).astype(bool)

            occlusion_masks.append(original_occlusion_mask)
            invalid_masks.append(original_invalid_mask)
        
        # Pad to same length as images
        occlusion_masks.append(None)
        invalid_masks.append(None)

        # -------------------- 2D Point Tracking on Original Images -------------------------
        original_tracks = []
        visibility_masks_all = []

        if len(frame_indices) > 0 and self.ENABLE_TRACK:
            # Create initial tracking grid on the ORIGINAL first frame
            first_original_shape = original_images[0].shape[:2]  # (H, W) original size
            initial_points = create_tracking_grid(
                first_original_shape, num_points=self.track_num
            )

            # Initialize tracking for first frame
            original_tracks.append(initial_points.copy())
            visibility_masks_all.append(np.ones(len(initial_points), dtype=bool))

            # Track points through subsequent frames using original flow
            current_points = initial_points.copy()
            current_visibility = np.ones(len(initial_points), dtype=bool)

            for i in range(len(frame_indices) - 1):
                flow = flows[i]  # Original flow from frame i to frame i+1
                occlusion_mask = occlusion_masks[i]  # Original occlusion mask
                invalid_mask = invalid_masks[i]  # Original invalid mask

                if flow is not None:
                    # Track points using original flow and masks - no transformation needed
                    tracked_points, visibility = track_points_with_flow_and_occlusion(
                        current_points,
                        flow,  # Original flow, no transformation
                        original_images[i + 1].shape[:2],  # Original target image shape
                        depth_map=None,  # Don't use depth for visibility
                        occlusion_mask=occlusion_mask,  # Original occlusion mask
                        invalid_mask=invalid_mask,  # Original invalid mask
                    )

                    # Check bounds on original image dimensions
                    h, w = original_images[i + 1].shape[:2]
                    margin = 20
                    permanently_out_of_bounds = (
                        (tracked_points[:, 0] < -margin)
                        | (tracked_points[:, 0] > w + margin)
                        | (tracked_points[:, 1] < -margin)
                        | (tracked_points[:, 1] > h + margin)
                    )

                    # Update visibility
                    current_visibility = current_visibility & ~permanently_out_of_bounds
                    visibility = visibility & current_visibility

                else:
                    # No flow available, assume points stay in same position
                    tracked_points = current_points.copy()
                    
                    # Check if points are still within reasonable bounds
                    h, w = original_images[i + 1].shape[:2]
                    margin = 20
                    in_reasonable_bounds = (
                        (tracked_points[:, 0] >= -margin)
                        & (tracked_points[:, 0] <= w + margin)
                        & (tracked_points[:, 1] >= -margin)
                        & (tracked_points[:, 1] <= h + margin)
                    )
                    visibility = in_reasonable_bounds & current_visibility

                # Update current state
                current_points = tracked_points.copy()
                current_visibility = visibility.copy()

                # Store results
                original_tracks.append(tracked_points)
                visibility_masks_all.append(visibility)

        if stride != 1 or not Flag or not self.ENABLE_TRACK:
            original_tracks = []
            visibility_masks_all = []
            for i in range(len(frame_indices)):
                original_tracks.append(np.zeros((self.track_num, 2), dtype=np.float32))
                visibility_masks_all.append(np.zeros((self.track_num), dtype=bool))

        # -------------------- Process Images with Original Tracks -------------------------
        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points, point_masks = [], [], []
        processed_tracks = []; processed_track_masks = []

        for i, fid in enumerate(frame_indices):
            # -------------------------------------------------- Depth ----
            depth_map = None
            depth_path = os.path.join(depth_root, f"frame_{fid:04d}.dpt")
            if os.path.exists(depth_path):
                depth_map = self._sintel_io.depth_read(depth_path)
                depth_map = threshold_depth_map(depth_map, max_percentile=98)

            # -------------------------------------------------- Cam ------
            cam_path = os.path.join(cam_root, f"frame_{fid:04d}.cam")
            if not os.path.exists(cam_path):
                raise FileNotFoundError(cam_path)
            intri_raw, extri_raw = self._sintel_io.cam_read(cam_path)
            intri_opencv = intri_raw.astype(np.float32)
            extri_opencv = extri_raw.astype(np.float32)  # world → cam

            # Get original image and track for this frame
            original_image = original_images[i]
            original_track = original_tracks[i] if i < len(original_tracks) else None
            original_size = np.array(original_image.shape[:2])  # (H, W)

            # Process one image - this will transform the original track to processed coordinates
            (
                proc_img,
                proc_depth,
                proc_extri,
                proc_intri,
                world_pts,
                cam_pts,
                pt_mask,
                processed_track,
                processed_track_mask,
            ) = self.process_one_image(
                original_image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_shape,
                track=original_track,  # Pass original track coordinates
                filepath=f"frame_{fid:04d}",
            )
            # print(world_pts.shape, '===========world_coords_points===========', proc_extri.shape, '===========processed_extri===========')
            # check = check_coord_system(world_pts.reshape(-1, 3)[pt_mask.reshape(-1)], proc_extri)
            # print(check, '===========check===========')
            # pdb.set_trace()
            # matrix_check(proc_img, proc_depth, proc_extri, proc_intri)
            # pdb.set_trace()
            # print(np.mean(proc_depth), '===========processed_depth===========', np.min(proc_depth), '===========np.min(processed_depth)===========', np.max(proc_depth), '===========np.max(processed_depth)===========')

            images.append(proc_img)
            depths.append(proc_depth)
            extrinsics.append(to_homogeneous(proc_extri))
            intrinsics.append(proc_intri)
            world_points.append(world_pts)
            cam_points.append(cam_pts)
            point_masks.append(pt_mask)
            processed_tracks.append(processed_track)
            processed_track_masks.append(processed_track_mask*visibility_masks_all[i])

        if processed_track[0].shape[0] < self.track_num:
            cur_len = processed_tracks[0].shape[0]
            idx = np.random.choice(cur_len, self.track_num, replace=True)
            for num in range(len(processed_tracks)):
                processed_tracks[num] = processed_tracks[num][idx]
                processed_track_masks[num] = processed_track_masks[num][idx]
        elif processed_track[0].shape[0] > self.track_num:
            for num in range(len(processed_tracks)):
                processed_tracks[num] = processed_tracks[num][:self.track_num]
                processed_track_masks[num] = processed_track_masks[num][:self.track_num]
        
        return {
            "seq_name": f"sintel_{seq_name}",
            "ids": np.array(frame_indices, dtype=np.int32),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "abandon_pose": False,
            "abandon_geometry": False,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "flows": flows,
            "occlusion_masks": occlusion_masks,
            "invalid_masks": invalid_masks,
            "tracks": processed_tracks,  # These are now properly transformed
            "track_masks": processed_track_masks,
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

    # Simple standalone test utility (only runs when the file is executed


if __name__ == "__main__":
    # Quick smoke-test
    class _DummyConf:
        def __init__(self):
            # Match defaults in BaseDataset.SimpleConfig used in other loaders
            self.img_size = 518
            self.patch_size = 14
            self.augs = type("augs", (), {"scales": [1.0, 1.0], "aspects": [1.0, 1.0]})
            self.rescale = False
            self.rescale_aug = False
            self.landscape_check = False
            self.training = False
            self.img_nums = [2, 2]
            self.track_num = 1024

    ds = SintelDataset(common_conf=_DummyConf(), verbose=True, strides=None)
    sample = ds.get_data(seq_index=0, aspect_ratio=1.0)
    print(
        sample["seq_name"],
        sample["images"][0].shape,
        sample["depths"][0].shape if sample["depths"][0] is not None else None,
        sample["cam_points"][0].shape if sample["cam_points"][0] is not None else None,
        (
            sample["world_points"][0].shape
            if sample["world_points"][0] is not None
            else None
        ),
        (
            sample["point_masks"][0].shape
            if sample["point_masks"][0] is not None
            else None
        ),
        sample["flows"][0].shape if sample["flows"][0] is not None else None,
        f"tracked_points: {len(sample['tracked_points'])} frames",
        f"first frame points: {sample['tracked_points'][0].shape if len(sample['tracked_points']) > 0 else None}",
        f"visibility: {sample['visibility_masks'][0].shape if len(sample['visibility_masks']) > 0 else None}",
    )
