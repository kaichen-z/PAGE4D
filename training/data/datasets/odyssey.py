import sys
import os
import glob
import random
import pdb
import numpy as np
import cv2
import json
import logging
from typing import Optional, List, Dict, Any
import os.path as osp

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

class OdysseyDataset(BaseDataset):
    """
    Odyssey dataset loader for VGGT training pipeline.
    
    This dataset loads Odyssey sequences which contain:
    - RGB images (rgbs/rgb_XXXXX.jpg)
    - Depth images (depths/depth_XXXXX.png) 
    - Mask images (masks/mask_XXXXX.png)
    - Camera parameters and trajectories in anno.npz
    - Scene metadata in scene_info.json
    """
    
    def __init__(
        self,
        common_conf,
        dataset_location: str = '/data/odyssey',
        split: str = 'train',
        sequence_names: Optional[List[str]] = None,  
        strides: List[int] = [1, 2, 3, 4, 5, 6, 7, 8],
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
        Initialize Odyssey dataset.
        
        Args:
            common_conf: Common configuration object
            dataset_location: Path to Odyssey dataset root
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
        print(f'Loading Odyssey dataset from {dataset_location}...')
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

        self.ENABLE_TRACK = False

    def _discover_sequences(self) -> List[str]:
        """
        Auto-discover all valid Odyssey sequence directories.
        
        Returns:
            List of sequence names (directory names) that contain required files
        """
        return self._discover_sequences_generic(
            dataset_location=self.dataset_location,
            required_subdirs=['rgbs'],
            required_files=['anno.npz'],
            image_subdir='rgbs',
            image_pattern='rgb_*.jpg',
            min_num_images=self.min_num_images,
            verbose=self.verbose
        )

    def _load_sequences(self):
        """Load and organize Odyssey sequences."""
        
        # Initialize storage
        self.sequence_list = []
        self.sequence_metadata = {}
        self.clip_data = []
        
        # Process each sequence
        for seq_name in self.sequence_names:
            seq_path = os.path.join(self.dataset_location, seq_name)
            
            if not os.path.exists(seq_path):
                if self.verbose:
                    print(f'Sequence path does not exist: {seq_path}')
                continue
                
            if self.verbose:
                print(f'Processing sequence: {seq_name}')
                
            # Check for required files
            rgb_path = os.path.join(seq_path, 'rgbs')
            anno_path = os.path.join(seq_path, 'anno.npz')
            scene_info_path = os.path.join(seq_path, 'scene_info.json')
            
            if not (os.path.isdir(rgb_path) and os.path.isfile(anno_path)):
                if self.verbose:
                    print(f'  Skipping {seq_name}: missing rgbs directory or anno.npz')
                continue
                
            # Load sequence annotations
            try:
                annotations = np.load(anno_path, allow_pickle=True)
                scene_info = {}
                if os.path.exists(scene_info_path):
                    with open(scene_info_path, 'r') as f:
                        scene_info = json.load(f)
            except Exception as e:
                if self.verbose:
                    print(f'  Error loading {seq_name}: {e}')
                continue
                
            # Get available RGB files
            rgb_files = sorted([
                f for f in os.listdir(rgb_path) 
                if f.startswith('rgb_') and f.endswith('.jpg')
            ])
            
            if len(rgb_files) < self.min_num_images:
                if self.verbose:
                    print(f'  Skipping {seq_name}: insufficient frames ({len(rgb_files)})')
                continue
            
            # Extract frame indices from RGB files
            frame_indices = []
            for rgb_file in rgb_files:
                try:
                    frame_idx = int(rgb_file.split('_')[1].split('.')[0])
                    frame_indices.append(frame_idx)
                except (ValueError, IndexError):
                    if self.verbose:
                        print(f'  Warning: Could not parse frame index from {rgb_file}')
                    continue
            
            frame_indices = sorted(frame_indices)
            num_frames = len(frame_indices)
            
            if num_frames < self.min_num_images:
                if self.verbose:
                    print(f'  Skipping {seq_name}: insufficient valid frames ({num_frames})')
                continue
                
            # Store sequence metadata
            self.sequence_list.append(seq_name)
            self.sequence_metadata[seq_name] = {
                'seq_path': seq_path,
                'num_frames': num_frames,
                'frame_indices': frame_indices,
                'annotations': annotations,
                'scene_info': scene_info,
                'rgb_files': rgb_files
            }
            
            # Generate clips for this sequence
            self._generate_clips_for_sequence(seq_name, num_frames, frame_indices)
            
        print(f'Successfully loaded {len(self.sequence_list)} sequences')

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
        """
        Get sequence data in VGGT format.
        
        Args:
            seq_index: Index of the clip to retrieve
            img_per_seq: Number of images per sequence
            seq_name: Name of the sequence (not used)
            ids: Specific frame IDs (not used)
            aspect_ratio: Target aspect ratio for image processing
            
        Returns:
            Dictionary containing sequence data in VGGT format
        """
        
        if self.inside_random:
            seq_index = random.randint(0, len(self.clip_data) - 1)
            
        # Get clip information
        clip_info = self.clip_data[seq_index]
        seq_name = clip_info['seq_name']
        base_frame_indices = clip_info['frame_indices']
        available_frames = clip_info['available_frames']
        
        # Get sequence metadata
        seq_meta = self.sequence_metadata[seq_name]
        
        # Extend indices if img_per_seq > len(base_frame_indices)
        if img_per_seq:
            frame_indices, FLAG = self._contiguous_window_impr(available_frames, base_frame_indices, img_per_seq)
        else:
            frame_indices = base_frame_indices
            
        # Get target image shape
        target_image_shape = self.get_target_shape(aspect_ratio)
        
        # Load and process images
        images = []
        depths = []
        extrinsics = []
        intrinsics = []
        cam_points = []
        world_points = []
        point_masks = []
        tracks = None
        track_masks = None

        # Extract track information if available
        if self.ENABLE_TRACK:
            if 'trajs_2d' in seq_meta['annotations']:
                tracks, track_masks = self._extract_tracks(
                    seq_meta['annotations'], frame_indices
                )
            tracks = tracks.transpose(1, 0, 2)
            track_masks = track_masks.transpose(1, 0)
        else:
            tracks = None
            track_masks = None

        processed_tracks = []; processed_track_masks = []

        for num, frame_idx in enumerate(frame_indices):
            # Load image
            rgb_path = os.path.join(
                seq_meta['seq_path'], 'rgbs', f'rgb_{frame_idx:05d}.jpg'
            )
            image = read_image_cv2(rgb_path)
            
            if image is None:
                raise ValueError(f"Could not load image: {rgb_path}")
            
            # Load depth
            depth_map = None
            if self.load_depth:
                depth_path = os.path.join(
                    seq_meta['seq_path'], 'depths', f'depth_{frame_idx:05d}.png'
                )
                #if os.path.exists(depth_path):
                depth16 = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                #if depth16 is not None:
                # Convert 16-bit depth to meters (assuming depth is in mm or similar)
                depth_map = depth16.astype(np.float32) / 65535.0 * 1000.0 # Convert to meters
                #depth_map = threshold_depth_map(depth_map, max_percentile=98)
            
            # Get camera parameters
            annotations = seq_meta['annotations']
            
            # Find the annotation index for this frame
            anno_frame_idx = frame_idx  # Assuming 1:1 mapping
            if anno_frame_idx >= annotations['intrinsics'].shape[0]:
                # Use the last available frame if out of bounds
                anno_frame_idx = annotations['intrinsics'].shape[0] - 1
                
            intri_opencv = annotations['intrinsics'][anno_frame_idx].astype(np.float32)
            extri_opencv = annotations['extrinsics'][anno_frame_idx].astype(np.float32)
            #extri_opencv = to_w2c_opencv(extri_opencv, fmt="w2c_opencv")
            # print(extri_opencv[:3, :3], '===========extri_opencv===========')
            # print(np.min(depth_map), '===========extri_opencv===========', np.max(depth_map))
            original_size = np.array(image.shape[:2])
            original_track = tracks[num] if tracks is not None else None
            # Process image using VGGT pipeline
            (
                processed_image,
                processed_depth,
                processed_extri,
                processed_intri,
                world_coords_points,
                cam_coords_points,
                point_mask,
                processed_track,
                processed_track_mask,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                track=original_track,
                filepath=rgb_path,
            )

            # print(world_coords_points.shape, '===========world_coords_points===========', point_mask.shape, '===========point_mask===========', processed_extri.shape, '===========processed_extri===========')
            # check = check_coord_system(world_coords_points.reshape(-1, 3)[point_mask.reshape(-1)], processed_extri)
            # print(check, '===========check===========')
            # pdb.set_trace()
            # matrix_check(processed_image, processed_depth, processed_intri, processed_extri)
            # print(np.mean(processed_depth), '===========processed_depth===========', np.min(processed_depth), '===========np.min(processed_depth)===========', np.max(processed_depth), '===========np.max(processed_depth)===========')

            images.append(processed_image)
            depths.append(processed_depth)
            extrinsics.append(processed_extri)
            intrinsics.append(processed_intri)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)

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
            "seq_name": f"odyssey_{seq_name}",
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
        
    def _extract_tracks(self, annotations: Dict, frame_indices: List[int]):
        """Extract track information from Odyssey annotations."""
        
        try:
            trajs_2d = annotations['trajs_2d']  # Shape: (T, N, 2)
            valids = annotations['valids']      # Shape: (T, N)
            visibs = annotations['visibs']      # Shape: (T, N)
            
            # Select trajectories for the specified frames
            selected_trajs = trajs_2d[frame_indices]  # Shape: (len(frame_indices), N, 2)
            selected_valids = valids[frame_indices]   # Shape: (len(frame_indices), N)
            selected_visibs = visibs[frame_indices]   # Shape: (len(frame_indices), N)
            
            # Transpose to get (N, T, 2) format expected by VGGT
            tracks = selected_trajs.transpose(1, 0, 2)  # (N, T, 2)
            
            # Create combined visibility mask (valid and visible)
            track_masks = (selected_valids & selected_visibs).transpose(1, 0)  # Shape: (N, T)
            
            # Get the configured number of tracks from common config
            target_track_num = self.track_num
            current_track_num = tracks.shape[0]
            
            # Handle NaN and inf values in tracks before resampling: set corresponding track_masks to False
            nan_mask = np.isnan(tracks).any(axis=2)  # (N, T) - True where any coordinate is NaN
            inf_mask = np.isinf(tracks).any(axis=2)  # (N, T) - True where any coordinate is inf
            invalid_mask = nan_mask | inf_mask  # Combined mask for NaN or inf
            track_masks = track_masks & (~invalid_mask)  # Set to False where tracks have NaN or inf
            
            # Also replace NaN and inf values in tracks with zeros for safety
            tracks = np.where(np.isnan(tracks) | np.isinf(tracks), 0.0, tracks)
            
            if current_track_num == target_track_num:
                # Perfect match, return as is
                return tracks, track_masks
            elif current_track_num > target_track_num:
                # Too many tracks, subsample randomly
                indices = np.random.choice(current_track_num, target_track_num, replace=False)
                indices = np.sort(indices)  # Keep order consistent
                tracks = tracks[indices]
                track_masks = track_masks[indices]
                return tracks, track_masks
            else:
                # Too few tracks, oversample with replacement
                indices = np.random.choice(current_track_num, target_track_num, replace=True)
                tracks = tracks[indices]
                track_masks = track_masks[indices]
                return tracks, track_masks
            
        except (KeyError, IndexError) as e:
            if self.verbose:
                print(f"Could not extract tracks: {e}")
            return None, None


def to_w2c_opencv(E: np.ndarray, fmt: str = "w2c_opencv") -> np.ndarray:
    """
    Convert an Odyssey extrinsic matrix to OpenCV world->camera (w2c).
    Args:
        E: (4,4) extrinsic from the annotations.
        fmt:
          - "c2w_opengl": camera-to-world in OpenGL convention (−Z forward)
          - "c2w_opencv": camera-to-world in OpenCV convention (+Z forward)
          - "w2c_opencv": already world-to-camera in OpenCV
          - "w2c_opengl": world-to-camera in OpenGL convention (rare)
    Returns:
        w2c_cv: (4,4) world->camera in OpenCV convention (+Z forward)
    """
    E = np.asarray(E, dtype=np.float32)
    assert E.shape == (4, 4), f"Expected 4x4, got {E.shape}"
    # sanity: homogeneous bottom row
    if not np.allclose(E[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Matrix bottom row is not [0,0,0,1]; check row/column-major or file format.")
    A = np.diag([1.0, -1.0, -1.0]).astype(np.float32)  # OpenGL->OpenCV camera axes
    if fmt == "c2w_opengl":
        # X_w = R_gl * X_c_gl + t_gl
        R_gl = E[:3, :3]
        t_gl = E[:3, 3]
        # Convert to OpenCV camera coords: X_c_cv = A * X_c_gl
        # So X_w = R_gl * A^{-1} * X_c_cv + t_gl = (R_gl * A) * X_c_cv + t_gl
        R_cv = R_gl @ A
        t_cv = t_gl.copy()
        # Now invert to w2c (OpenCV): [R^T | -R^T t]
        w2c_cv = np.eye(4, dtype=np.float32)
        w2c_cv[:3, :3] = R_cv.T
        w2c_cv[:3, 3]  = -R_cv.T @ t_cv
        return w2c_cv
    elif fmt == "c2w_opencv":
        # Just invert
        return np.linalg.inv(E).astype(np.float32)
    elif fmt == "w2c_opencv":
        # Already in target form
        return E.astype(np.float32)
    elif fmt == "w2c_opengl":
        # First turn w2c (OpenGL) into w2c (OpenCV).
        # We have X_c_gl = R_gl_w2c * X_w + t_gl_w2c
        # And X_c_cv = A * X_c_gl = A*(R_gl_w2c * X_w + t_gl_w2c)
        R_gl_w2c = E[:3, :3]
        t_gl_w2c = E[:3, 3]
        w2c_cv = np.eye(4, dtype=np.float32)
        w2c_cv[:3, :3] = A @ R_gl_w2c
        w2c_cv[:3, 3]  = A @ t_gl_w2c
        return w2c_cv
    else:
        raise ValueError(f"Unknown fmt '{fmt}'.")
    
if __name__ == "__main__":
    # Simple test
    print("Testing Odyssey dataset...")
    
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
    
    # Test 1: Auto-discovery (new pattern)
    print("\n=== Testing auto-discovery (sequence_names=None) ===")
    dataset_auto = OdysseyDataset(
        common_conf=config,
        dataset_location='', # test with your local path
        sequence_names=None,  # Auto-discover all sequences
        quick=True,
        verbose=True,
        len_train=100
    )
    print(f"Auto-discovered dataset length: {len(dataset_auto)}")
    
    # Test 2: Explicit sequence specification (original pattern)
    print("\n=== Testing explicit sequences (original pattern) ===")
    dataset_explicit = OdysseyDataset(
        common_conf=config,
        dataset_location='', # test with your local path
        sequence_names=['dancing'],  # Explicit sequence list
        quick=True,
        verbose=True,
        len_train=100
    )
    print(f"Explicit dataset length: {len(dataset_explicit)}")
    
    if len(dataset_auto) > 0:
        # Test getting a sample
        sample = dataset_auto.get_data(seq_index=0, img_per_seq=4, aspect_ratio=1.0)
        print(f"Sample keys: {sample.keys()}")
        print(f"Images shape: {[img.shape for img in sample['images']]}")
        print(f"IDs: {sample['ids']}")
        print("Test completed successfully!")
    else:
        print("No data found - check dataset path") 
