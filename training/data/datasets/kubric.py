import sys
import os
import glob
import random
import numpy as np
import cv2
import json
import logging
from typing import Optional, List, Dict, Any
import os.path as osp
import einops
import pdb

# Add paths for imports
sys.path.append('../../..')

# Local imports
from ..base_dataset import BaseDataset
from ..dataset_util import get_stride_distribution, read_image_cv2, threshold_depth_map


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
    print(T_cw.shape, '===========T_cw===========')
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

class KubricDataset(BaseDataset):
    """
    Kubric MOVI-E dataset loader for the VGGT training pipeline.

    The loader expects each scene directory (e.g. ``scene_0001``) to contain at minimum:
      • RGBA frames  ``rgba_XXXXX.png``
      • Per-pixel depth maps ``depth_XXXXX.tiff`` (floating or 16-bit depth)
      • A ``metadata.json`` file holding camera positions & quaternions (+ other Kubric metadata)
      • Optionally, precomputed tracks ``tracks.npy`` and track masks ``track_masks.npy`` files.
    Optional files (segmentation, flows, etc.) are ignored for now.
    The class converts the Kubric camera description to OpenCV-style intrinsics & world→camera
    extrinsics, then feeds them through the shared ``BaseDataset.process_one_image`` pipeline.

    Tracks and track masks are also loaded if present in the scene directory.
    """
    
    def __init__(
        self,
        common_conf,
        dataset_location: str = '/data/kubric',
        split: str = 'train',
        sequence_names: Optional[List[str]] = None,
        strides: List[int] = [1],
        clip_step: int = 2,
        min_num_images: int = 8,
        len_train: int = 10000,
        len_test: int = 1000,
        quick: bool = False,
        verbose: bool = False,
        dist_type: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Kubric dataset.
        
        Args:
            common_conf: Common configuration object
            dataset_location: Path to Kubric dataset root
            split: Dataset split ('train' or 'val')
            sequence_names: List of sequence names to load. If None, auto-discovers all sequences.
            strides: List of temporal strides to use
            clip_step: Step size for sampling clips
            min_num_images: Minimum number of images per sequence
            len_train: Training dataset length
            len_test: Test dataset length  
            quick: Quick mode for testing (uses subset)
            verbose: Verbose logging
            dist_type: Distribution type for stride sampling
        """
        print(f'Loading Kubric MOVI-E dataset from {dataset_location}...')
        super().__init__(common_conf=common_conf)
        
        # Dataset configuration
        self.dataset_location = dataset_location
        self.split = split
        self.strides = strides
        self.clip_step = clip_step
        self.min_num_images = min_num_images
        self.quick = quick
        self.verbose = verbose
        self.dist_type = dist_type
        
        # Auto-discover sequences if not specified
        if sequence_names is None:
            self.sequence_names = self._discover_sequences()
            if self.verbose:
                print(f'Auto-discovered sequences: {self.sequence_names}')
        else:
            self.sequence_names = sequence_names
        
        # Set dataset length
        if split == 'train':
            self.len_train = len_train
        else:
            self.len_train = len_test
            
        # Configuration from common_conf
        self.debug = getattr(common_conf, 'debug', False)
        self.get_nearby = getattr(common_conf, 'get_nearby', True)
        self.load_depth = getattr(common_conf, 'load_depth', True)
        self.inside_random = getattr(common_conf, 'inside_random', False)
        
        # Load sequences and organize data
        self._load_sequences()
        
        # Apply stride distribution if specified
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)
            
        print(f'Loaded {len(self.sequence_list)} sequences with {len(self.clip_data)} clips')
        self.ENABLE_TRACK = False

    def _discover_sequences(self) -> List[str]:
        """
        Auto-discover all valid Kubric sequence directories.
        
        Returns:
            List of sequence names (directory names) that contain required files
        """
        return self._discover_sequences_generic(
            dataset_location=self.dataset_location,
            required_subdirs=[],  # No specific subdirs required for Kubric
            required_files=['metadata.json'],  # Kubric requires metadata.json
            image_subdir='',  # Images are in the root of each scene directory  
            image_pattern='rgba_*.png',  # Kubric uses rgba_XXXXX.png pattern
            min_num_images=self.min_num_images,
            verbose=self.verbose
        )

    # Helper functions 
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion (x, y, z, w) to a 3×3 rotation matrix."""
        q = q.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        x, y, z, w = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        rot = np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ], dtype=np.float32)
        return rot

    def _create_extrinsic(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Return 4×4 world→camera extrinsic matrix."""
        # Kubric uses a different convention (right-handed OpenGL). We flip the Y/Z axes
        # to match OpenCV (right-handed camera with +Z forward).
        flip_mat = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

        R_c2w = self._quaternion_to_rotation_matrix(quaternion) @ flip_mat
        T_c2w = np.asarray(position, dtype=np.float32).reshape(3)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = T_c2w

        w2c = np.linalg.inv(c2w).astype(np.float32)
        return w2c

    def _compute_intrinsic(self, width: int, height: int, fov_rad: float) -> np.ndarray:
        """Compute pin-hole intrinsics from horizontal field-of-view (radians)."""
        fx = fy = (width / 2.0) / np.tan(fov_rad / 2.0)
        cx = width / 2.0
        cy = height / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    def _load_sequences(self):
        """Scan scene folders and pre-compute frame lists & camera parameters."""

        self.sequence_list = []
        self.sequence_metadata = {}
        self.clip_data = []

        for seq_name in self.sequence_names:
            seq_path = os.path.join(self.dataset_location, seq_name)

            if not os.path.isdir(seq_path):
                if self.verbose:
                    print(f"Sequence path does not exist: {seq_path}")
                continue

            if self.verbose:
                print(f"Processing sequence: {seq_name}")

            # Gather frame file names
            rgba_files = sorted([
                f for f in os.listdir(seq_path) if f.startswith("rgba_") and f.endswith(".png")
            ])

            if len(rgba_files) < self.min_num_images:
                if self.verbose:
                    print(f"  Skipping {seq_name}: insufficient frames ({len(rgba_files)})")
                continue

            frame_indices = []
            for fname in rgba_files:
                try:
                    fid = int(fname.split("_")[-1].split(".")[0])
                    frame_indices.append(fid)
                except Exception:
                    if self.verbose:
                        print(f"  Warning: could not parse frame index from {fname}")
            frame_indices = sorted(frame_indices)

            num_frames = len(frame_indices)
            # Load metadata.json
            metadata_path = os.path.join(seq_path, "metadata.json")
            if not os.path.isfile(metadata_path):
                if self.verbose:
                    print(f"  Skipping {seq_name}: missing metadata.json")
                continue

            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                camera_meta = metadata.get("camera", {})
            except Exception as e:
                if self.verbose:
                    print(f"  Error loading metadata for {seq_name}: {e}")
                continue

            # Store per-sequence info
            self.sequence_list.append(seq_name)
            self.sequence_metadata[seq_name] = {
                "seq_path": seq_path,
                "num_frames": num_frames,
                "frame_indices": frame_indices,
                "camera": metadata.get("camera", {}),
                "instances": metadata.get("instances", []),
                "metadata": metadata.get("metadata", {}),
            }

            # Generate training clips
            self._generate_clips_for_sequence(seq_name, num_frames, frame_indices)

        print(f"Successfully loaded {len(self.sequence_list)} Kubric sequences")

    def _generate_clips_for_sequence(self, seq_name: str, num_frames: int, frame_indices: List[int]):
        """Generate training clips for a sequence with different strides."""
        
        for stride in self.strides:
            max_start_idx = num_frames - 2 * stride
            
            for start_idx in range(0, max_start_idx + 1, self.clip_step):
                # Generate frame indices for this clip (2 frames minimum)
                clip_frame_indices = [
                    frame_indices[start_idx],
                    frame_indices[min(start_idx + stride, num_frames - 1)]
                ]
                
                # Store clip information
                self.clip_data.append({
                    'seq_name': seq_name,
                    'frame_indices': clip_frame_indices,
                    'start_idx': start_idx,
                    'stride': stride,
                    'available_frames': frame_indices
                })

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

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: List[int] = None,
        aspect_ratio: float = 1.0,
    ) -> Dict[str, Any]:
        """Load a clip and return data in VGGT format."""

        # ------------------------ select clip -----------------------------
        if self.inside_random or seq_index is None:
            seq_index = random.randint(0, len(self.clip_data) - 1)

        clip_info = self.clip_data[seq_index]
        seq_name = clip_info["seq_name"]
        base_frame_indices = clip_info["frame_indices"]
        available_frames = clip_info["available_frames"]
        # Extend indices if img_per_seq > len(base_frame_indices)
        if img_per_seq:
            frame_indices, FLAG = self._contiguous_window_impr(available_frames, base_frame_indices, img_per_seq)
        else:
            frame_indices = base_frame_indices
        seq_meta = self.sequence_metadata[seq_name]
        cam_meta = seq_meta["camera"]
        # --------------------- load & process frames -----------------------
        target_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []

        # Load precomputed tracks
        if self.ENABLE_TRACK:
            # Prefer preprocessed tracks if they exist
            tracks_path = os.path.join(seq_meta["seq_path"], 'tracks.npy')
            track_masks_path = os.path.join(seq_meta["seq_path"], 'track_masks.npy')
            all_tracks = np.load(tracks_path)
            all_track_masks = np.load(track_masks_path)
            all_scene_frames = seq_meta['frame_indices']
            frame_to_idx_map = {frame_num: i for i, frame_num in enumerate(all_scene_frames)}
            slice_indices = [frame_to_idx_map[f] for f in frame_indices if f in frame_to_idx_map]
            if slice_indices:
                tracks = all_tracks[:, slice_indices, :]
                track_masks = all_track_masks[:, slice_indices]
                # Normalize track count to match configuration
                current_track_num = tracks.shape[0]
                target_track_num = self.track_num
                # Handle NaN and inf values in tracks before resampling: set corresponding track_masks to False
                nan_mask = np.isnan(tracks).any(axis=2)  # (N, T) - True where any coordinate is NaN
                inf_mask = np.isinf(tracks).any(axis=2)  # (N, T) - True where any coordinate is inf
                invalid_mask = nan_mask | inf_mask  # Combined mask for NaN or inf
                track_masks = track_masks & (~invalid_mask)  # Set to False where tracks have NaN or inf
                # Also replace NaN and inf values in tracks with zeros for safety
                tracks = np.where(np.isnan(tracks) | np.isinf(tracks), 0.0, tracks)
                if current_track_num != target_track_num:
                    if current_track_num > target_track_num:
                        # Too many tracks, subsample randomly
                        indices = np.random.choice(current_track_num, target_track_num, replace=False)
                        indices = np.sort(indices)  # Keep order consistent
                        tracks = tracks[indices]
                        track_masks = track_masks[indices]
                    else:
                        # Too few tracks, oversample with replacement
                        indices = np.random.choice(current_track_num, target_track_num, replace=True)
                        tracks = tracks[indices]
                        track_masks = track_masks[indices]

            
            tracks = tracks.transpose(1, 0, 2)
            track_masks = track_masks.transpose(1, 0)
        else:
            tracks, track_masks = None, None
        processed_tracks = []; processed_track_masks = []
        for num, frame_idx in enumerate(frame_indices):
            # Image
            img_path = os.path.join(seq_meta["seq_path"], f"rgba_{frame_idx:05d}.png")
            image = read_image_cv2(img_path)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
            # Depth
            depth_map = None
            if self.load_depth:
                depth_path = os.path.join(seq_meta["seq_path"], f"depth_{frame_idx:05d}.tiff")
                if os.path.exists(depth_path):
                    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    if depth_raw is not None:
                        depth_map = depth_raw.astype(np.float32)
                        depth_map = threshold_depth_map(depth_map, max_percentile=98)

            # Camera params
            # Safety check for index overflow
            fid = min(frame_idx, len(cam_meta.get("positions", [])) - 1)
            position = np.array(cam_meta["positions"][fid], dtype=np.float32)
            quaternion = np.array(cam_meta["quaternions"][fid], dtype=np.float32)
            #extri_opencv = self._create_extrinsic(position, quaternion)
            extri_opencv = create_extrinsic_w2c_from_kubric(position, quaternion, order='wxyz')

            # Intrinsics (constant across frames)
            h, w = image.shape[:2]
            fov = float(cam_meta.get("field_of_view", np.deg2rad(50)))  # radian already in json
            #intri_opencv = self._compute_intrinsic(w, h, fov)
            intri_opencv = compute_intrinsic_from_fov(w, h, fov, fov_type='horizontal', use_pixel_center=True)
            
            depth_map = euclidean_to_Z(depth_map, intri_opencv)

            original_size = np.array([h, w])
            original_track = tracks[num] if tracks is not None else None
            
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
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_shape,
                track=original_track,
                filepath=img_path,
            )

            # check = check_coord_system(world_pts.reshape(-1, 3)[pt_mask.reshape(-1)], proc_extri)
            # print(check, '===========check===========')
            # matrix_check(proc_img, proc_depth, proc_extri, proc_intri)

            images.append(proc_img)
            depths.append(proc_depth)
            extrinsics.append(proc_extri)
            intrinsics.append(proc_intri)
            cam_points.append(cam_pts)
            world_points.append(world_pts)
            point_masks.append(pt_mask)
            if self.ENABLE_TRACK:
                processed_tracks.append(processed_track)
                processed_track_masks.append(processed_track_mask&track_masks[num])
            else:
                processed_tracks.append(None)
                processed_track_masks.append(None)
        if not self.ENABLE_TRACK:
            processed_tracks = np.zeros((len(images), self.track_num, 2))
            processed_track_masks = np.zeros((len(images), self.track_num))        # Return in VGGT format
        return {
            "seq_name": f"kubric_{seq_name}",
            "ids": np.array(frame_indices),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "abandon_pose": False,
            "abandon_geometry": False,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "tracks": processed_tracks,
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

def map_tracks_to_target(
    tracks,            # (N, T, 2) in source pixel coords, or normalized
    track_masks,       # (N, T)
    src_size,          # (Hs, Ws) 轨迹对应的原始渲染尺寸（从 metadata.json 读）
    tgt_size,          # (Ht, Wt) 当前图像（喂给模型的那一帧）尺寸
    assume_normalized=False,     # 若确定是 [0,1]，设 True
    use_letterbox=True,          # 你的图像预处理如保持长宽比+居中黑边，就 True
    add_pixel_center=False,      # 若怀疑需要 +0.5 像素中心修正，设 True 试一下
    swap_xy=False                # 若怀疑 (y,x)，设 True 试一下
):
    tracks = tracks.copy().astype(np.float32)
    masks  = track_masks.copy()
    Hs, Ws = src_size
    Ht, Wt = tgt_size
    # 1) 归一化->像素（如需要）
    if assume_normalized:
        tracks[..., 0] *= (Ws - 1)
        tracks[..., 1] *= (Hs - 1)
    # 2) 可选像素中心修正
    if add_pixel_center:
        tracks[..., 0] += 0.5
        tracks[..., 1] += 0.5
    # 3) 可选轴交换
    if swap_xy:
        tracks = tracks[..., ::-1]  # (x,y) <-> (y,x)
    # 4) 应用和图像一致的 resize / letterbox
    if use_letterbox:
        scale = min(Wt / float(Ws), Ht / float(Hs))
        new_w, new_h = int(round(Ws * scale)), int(round(Hs * scale))
        x0 = (Wt - new_w) // 2
        y0 = (Ht - new_h) // 2
        tracks[..., 0] = tracks[..., 0] * scale + x0
        tracks[..., 1] = tracks[..., 1] * scale + y0
    else:
        sx = Wt / float(Ws)
        sy = Ht / float(Hs)
        tracks[..., 0] *= sx
        tracks[..., 1] *= sy
    # 5) 屏蔽图外 & 清理 NaN/Inf
    bad = np.isnan(tracks).any(-1) | np.isinf(tracks).any(-1)
    out = (tracks[..., 0] < 0) | (tracks[..., 0] >= Wt) | \
          (tracks[..., 1] < 0) | (tracks[..., 1] >= Ht)
    masks = masks & (~bad) & (~out)
    tracks = np.where(np.isnan(tracks) | np.isinf(tracks), 0.0, tracks)
    tracks[..., 0] = np.clip(tracks[..., 0], 0, Wt - 1)
    tracks[..., 1] = np.clip(tracks[..., 1], 0, Ht - 1)
    return tracks, masks

S_cv2gl = np.diag([1.0, -1.0, -1.0]).astype(np.float32)  # OpenCV->OpenGL camera basis

def quat_to_R_wxyz(q):
    """q=(w,x,y,z) -> 3x3."""
    q = np.asarray(q, np.float32)
    q /= np.linalg.norm(q) + 1e-8
    w, x, y, z = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float32)

def quat_to_R_xyzw(q):
    """q=(x,y,z,w) -> 3x3."""
    x,y,z,w = np.asarray(q, np.float32)
    return quat_to_R_wxyz([w,x,y,z])

def create_extrinsic_w2c_from_kubric(position, quaternion, order='wxyz'):
    """
    position: camera center in WORLD (3,)
    quaternion: OpenGL camera orientation (c2w) as quaternion
    order: 'wxyz' or 'xyzw'
    Returns: w2c (4x4) in OpenCV convention.
    """
    R_c2w_gl = quat_to_R_wxyz(quaternion) if order=='wxyz' else quat_to_R_xyzw(quaternion)
    t_c2w = np.asarray(position, np.float32).reshape(3)

    # OpenGL -> OpenCV on the CAMERA side (post-multiply)
    R_c2w_cv = R_c2w_gl @ S_cv2gl

    C2W_cv = np.eye(4, dtype=np.float32)
    C2W_cv[:3,:3] = R_c2w_cv
    C2W_cv[:3, 3] = t_c2w

    W2C_cv = np.linalg.inv(C2W_cv).astype(np.float32)
    return W2C_cv

def compute_intrinsic_from_fov(W, H, fov_rad, fov_type='horizontal', use_pixel_center=True):
    """
    fov_type: 'horizontal' or 'vertical'
    """
    if fov_type == 'horizontal':
        fx = (W/2.0) / np.tan(fov_rad/2.0)
        fy = fx * (W/float(H))
    else:  # 'vertical'
        fy = (H/2.0) / np.tan(fov_rad/2.0)
        fx = fy * (H/float(W))
    cx = (W-1)/2.0 if use_pixel_center else W/2.0
    cy = (H-1)/2.0 if use_pixel_center else H/2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return K

def euclidean_to_Z(depth_d, K):
    """depth_d: (H,W) 欧氏距离; K: 3x3 像素内参 -> 返回相机Z深度 (H,W)"""
    H, W = depth_d.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    u = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    v = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
    x = (u - cx) / fx
    y = (v - cy) / fy
    s = np.sqrt(x*x + y*y + 1.0)          # ||v||
    Z = depth_d / s                        # Z = d / ||v||
    return Z

if __name__ == "__main__":
    # Simple test
    print("Testing Kubric dataset...")
    
    # Create a simple config object
    class SimpleConfig:
        def __init__(self):
            self.img_size = 224
            self.patch_size = 14
            self.training = True
            self.rescale = True
            self.rescale_aug = False
            self.landscape_check = False
            self.debug = False
            self.get_nearby = True
            self.load_depth = True
            self.inside_random = False
            self.augs = {'scales': [0.8, 1.2]}
    
    config = SimpleConfig()
    
    # Create dataset
    dataset = KubricDataset(
        common_conf=config,
        dataset_location='/shared/ssd_14T/gaspard/output', # test with your local path
        sequence_names=['scene_0001'],
        quick=True,
        verbose=True,
        len_train=100
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test getting a sample
        sample = dataset.get_data(seq_index=0, img_per_seq=4, aspect_ratio=1.0)
        print(f"Sample keys: {sample.keys()}")
        print(f"Images shape: {[img.shape for img in sample['images']]}")
        print(f"IDs: {sample['ids']}")
        print("Test completed successfully!")
        if sample['tracks'] is not None:
            print(f"Tracks shape: {sample['tracks'].shape}")
            print(f"Track masks shape: {sample['track_masks'].shape}")
            track_masks = sample['track_masks']
            visible_points_per_frame = np.sum(track_masks, axis=0)
            print(f"Visible points per frame in track_masks: {visible_points_per_frame}")
        else:
            print("Tracks not found in sample.")
    else:
        print("No data found - check dataset path") 