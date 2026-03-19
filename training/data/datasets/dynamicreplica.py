import sys
import os
import glob
import random
import numpy as np
import cv2
import logging
import gzip
import json
from typing import Optional, List, Dict, Any
import os.path as osp
import pdb
import torch
# Add paths for imports
sys.path.append('../../..')
from PIL import Image
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

def matrix_check(image, depth, K, T_cw):
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
    # 采样若干像素
    H,W = image.shape[:2]
    ys = np.linspace(16, H-16, 50).astype(int)
    xs = np.linspace(16, W-16, 50).astype(int)
    uu, vv = np.meshgrid(xs, ys)
    zz = depth[vv, uu]
    uvz = np.stack([uu.ravel(), vv.ravel(), zz.ravel()], axis=1).astype(np.float32)
    Xw = backproject_to_world(uvz, K, T_cw)
    uv_hat = project_to_pixels(Xw, K, T_cw)
    err = np.linalg.norm(uv_hat - uvz[:,:2], axis=1)
    print("reproj err px  mean/med/max:", err.mean(), np.median(err), err.max())

class DynamicReplicaDataset(BaseDataset):
    """
    DynamicReplica dataset loader for VGGT training pipeline.
    
    This dataset loads DynamicReplica sequences which contain:
    - RGB images (images/f14caa-3_obj_source_right-XXXXX.png)
    - Depth images (depths/f14caa-3_obj_source_right_XXXXX.geometric.png)
    - Optical flow (flow_forward/f14caa-3_obj_source_right_XXXXX.png)
    - Instance maps (instance_id_maps/f14caa-3_obj_source_right_XXXXX.png)
    - Masks (masks/f14caa-3_obj_source_right_XXXXX.png)
    """
    
    def __init__(
        self,
        common_conf,
        dataset_location: str = '/data/dynamicreplica',
        annotations_file: str = '/workspace/data/kaichen/data/dynreplica/dynamic_stereo/dynamic_replica_data/train/frame_annotations_train.jgz',
        split: str = 'train',
        sequence_names: Optional[List[str]] = None,  
        strides: List[int] = [1, 2, 3, 4],
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
        Initialize DynamicReplica dataset.
        
        Args:
            common_conf: Common configuration object
            dataset_location: Path to DynamicReplica dataset root
            annotations_file: Path to camera annotations .jgz file
            split: Dataset split ('train' or 'val')
            sequence_names: List of sequence names to load
            strides: List of temporal strides to use
            clip_step: Step size for sampling clips
            min_num_images: Minimum number of images per sequence
            len_train: Training dataset length
            len_test: Test dataset length  
            quick: Quick mode for testing (uses subset)
            verbose: Verbose logging
            dist_type: Distribution type for stride sampling
        """
        print(f'Loading DynamicReplica dataset from {dataset_location}...')
        super().__init__(common_conf=common_conf)
        
        # Dataset configuration
        self.dataset_location = dataset_location
        if 'train' in dataset_location:
            self.annotations_file = f'/workspace/data/kaichen/data/dynreplica/dynamic_stereo/dynamic_replica_data/train/frame_annotations_train.jgz'
        elif 'val' in dataset_location:
            self.annotations_file = f'/workspace/data/kaichen/data/dynreplica/dynamic_stereo/dynamic_replica_data/val/frame_annotations_valid.jgz'
        elif 'test' in dataset_location:
            self.annotations_file = f'/workspace/data/kaichen/data/dynreplica/dynamic_stereo/dynamic_replica_data/test/frame_annotations_test.jgz'
        self.split = split
        self.strides = strides
        self.clip_step = clip_step
        self.min_num_images = min_num_images
        self.quick = quick
        self.verbose = verbose
        self.dist_type = dist_type
        
        # Load camera annotations
        self.camera_annotations = self._load_camera_annotations()

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
        self.track_num = getattr(common_conf, 'track_num', 256)  # Default to 256 tracks
        self.depth_eps = 1e-5
        # Load sequences and organize data
        self._load_sequences()
        
        # Apply stride distribution if specified
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)
            
        print(f'Loaded {len(self.sequence_list)} sequences with {len(self.clip_data)} clips')
        
        self.ENABLE_TRACK = False

    def _discover_sequences(self) -> List[str]:
        """
        Auto-discover all valid DynamicReplica sequence directories.
        
        DynamicReplica sequences contain images and depths directories.
        Image files follow the pattern: seqname-XXXX.png
        
        Returns:
            List of sequence names (directory names) that contain required files
        """
        return self._discover_sequences_generic(
            dataset_location=self.dataset_location,
            required_subdirs=['images', 'depths'],
            required_files=[],  # No specific files required at sequence level
            image_subdir='images',
            image_pattern='*.png',  # DynamicReplica uses seqname-XXXX.png pattern
            min_num_images=self.min_num_images,
            verbose=self.verbose
        )

    def _load_sequences(self):
        """Load and organize DynamicReplica sequences."""
        
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
                
            # Check for required directories
            images_path = os.path.join(seq_path, 'images')
            depths_path = os.path.join(seq_path, 'depths')
            
            if not (os.path.isdir(images_path) and os.path.isdir(depths_path)):
                if self.verbose:
                    print(f'  Skipping {seq_name}: missing images or depths directory')
                continue
                
            # Get available RGB files (pattern: f14caa-3_obj_source_right-XXXXX.png)
            rgb_files = sorted([
                f for f in os.listdir(images_path) 
                if f.startswith(seq_name + '-') and f.endswith('.png')
            ])
            
            if len(rgb_files) < self.min_num_images:
                if self.verbose:
                    print(f'  Skipping {seq_name}: insufficient frames ({len(rgb_files)})')
                continue
            
            # Extract frame indices from RGB files
            frame_indices = []
            for rgb_file in rgb_files:
                try:
                    # Extract frame index from pattern: f14caa-3_obj_source_right-XXXX.png
                    frame_idx_str = rgb_file.split('-')[-1].split('.')[0]
                    frame_idx = int(frame_idx_str)
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
                'rgb_files': rgb_files,
                'seq_name_prefix': seq_name  # Store the sequence name prefix for file naming
            }
            
            # Generate clips for this sequence
            self._generate_clips_for_sequence(seq_name, num_frames, frame_indices)
            
        print(f'Successfully loaded {len(self.sequence_list)} sequences')

    def _load_camera_annotations(self) -> Dict[str, Dict]:
        """
        Load camera annotations from .jgz file.
        Returns:
            Dictionary mapping (sequence_name, camera_name, frame_number) -> annotation data
        """
        print(f'Loading camera annotations from {self.annotations_file}...')
        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(f"Camera annotations file not found: {self.annotations_file}")
        try:
            # Load gzipped JSON file
            with gzip.open(self.annotations_file, 'rt', encoding='utf-8') as f:
                annotations_list = json.load(f)
            # Convert list to dictionary for fast lookup
            # Key format: (sequence_name, camera_name, frame_number)
            annotations_dict = {}
            for annotation in annotations_list:
                key = (
                    annotation['sequence_name'],
                    annotation.get('camera_name', 'unknown'),
                    annotation['frame_number'])
                annotations_dict[key] = annotation
            print(f'Loaded {len(annotations_dict)} camera annotation entries')
            return annotations_dict
        except Exception as e:
            print(f'Error loading camera annotations: {e}')
            return {}

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
        seq_name_prefix = seq_meta['seq_name_prefix']
        # Extend indices if img_per_seq > len(base_frame_indices)
        if img_per_seq:
            frame_indices, FLAG = self._contiguous_window_impr(available_frames, base_frame_indices, img_per_seq)
        else:
            frame_indices = base_frame_indices

        stride = frame_indices[1] - frame_indices[0]
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
        processed_tracks = []
        processed_track_masks = []
        # Create synthetic tracks similar to Odyssey's approach
        if self.ENABLE_TRACK:
            if stride == 1 and FLAG:
                tracks, track_masks = self._extract_synthetic_tracks(
                    seq_meta, frame_indices)
                tracks = tracks.transpose(1, 0, 2)
                track_masks = track_masks.transpose(1, 0)
            else:
                tracks = np.zeros((len(frame_indices), self.track_num, 2), dtype=np.float32)
                track_masks = np.zeros((len(frame_indices), self.track_num), dtype=bool)
        else:
            tracks = None
            track_masks = None

        for num, frame_idx in enumerate(frame_indices):
            # Load image (pattern: f14caa-3_obj_source_right-XXXX.png)
            rgb_path = os.path.join(
                seq_meta['seq_path'], 'images', f'{seq_name_prefix}-{frame_idx:04d}.png')
            image = read_image_cv2(rgb_path)
            if image is None:
                raise ValueError(f"Could not load image: {rgb_path}")
            # Handle BGRA format for DynamicReplica
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            # Load depth (pattern: f14caa-3_obj_source_right_XXXX.geometric.png)
            depth_map = None
            if self.load_depth:
                depth_path = os.path.join(
                    seq_meta['seq_path'], 'depths', f'{seq_name_prefix}_{frame_idx:04d}.geometric.png')
                if os.path.exists(depth_path):
                    depth_map = self._load_16bit_png_depth(depth_path)
                    depth_mask = depth_map < self.depth_eps
                    depth_map[depth_mask] = self.depth_eps
            #print(depth_path, '===========depth_path===========', depth_map.shape, '===========depth_map.shape===========')
            #print(rgb_path, '===========depth_path===========', image.shape, '===========image.shape===========')
            # Extract camera parameters from annotations
            camera_name = 'unknown'
            if 'source_left' in seq_name_prefix:
                camera_name = 'left'
            elif 'source_right' in seq_name_prefix:
                camera_name = 'right'
            # Look up camera annotation for this frame
            seq_name1, _ = seq_name.split("obj", 1)
            seq_name1 = seq_name1 + "obj"
            annotation_key = (seq_name1, camera_name, frame_idx)
            annotation = self.camera_annotations.get(annotation_key)
            # Extract viewpoint data
            viewpoint = annotation['viewpoint']
            rotation = viewpoint['R']
            translation = viewpoint['T']
            image_size = annotation['image']['size'] 
            intri_opencv = convert_ndc_to_pixel_intrinsics(viewpoint, image_size)
            extri_opencv = build_w2c_4(rotation, translation)
            original_size = np.array(image.shape[:2])
            original_track = tracks[num] if tracks is not None else None
            # Process image using VGGT pipeline
            (   processed_image,
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
                filepath=rgb_path,)
            # print(world_coords_points.shape, '===========world_coords_points===========', extri_opencv.shape, '===========processed_extri===========')
            # check = check_coord_system(world_coords_points.reshape(-1, 3)[point_mask.reshape(-1)], extri_opencv)
            # print(check, '===========check===========')
            # matrix_check(processed_image, processed_depth, processed_intri, processed_extri)
            # print(processed_extri, '===========processed_extri===========')
            # print(processed_depth.max(),"===========processed_depth.max", processed_depth.mean(), "===========", processed_depth.min())

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
            processed_track_masks = np.zeros((len(images), self.track_num))
        # Return in VGGT format
        return {
            "seq_name": f"dynamicreplica_{seq_name}",
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

    def _load_16bit_png_depth(self, depth_png: str) -> np.ndarray:
        """Load Dynamic Replica depth PNG encoded as float16 (meters) into float32."""
        with Image.open(depth_png) as depth_pil:
            depth = (np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32).reshape((depth_pil.size[1], depth_pil.size[0])))
        return depth

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

    def _load_flow_png(self, path: str, target_shape):
        """
        读取 KITTI 风格 16-bit PNG 光流：
        u = (png[...,0] - 32768)/64, v = (png[...,1] - 32768)/64
        自动 resize 到 target_shape，并按比例缩放位移。
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.dtype != np.uint16 or img.ndim != 3 or img.shape[2] < 2:
            return None
        u = img[:, :, 0].astype(np.float32)
        v = img[:, :, 1].astype(np.float32)
        flow = np.stack([(u - 32768.0) / 64.0, (v - 32768.0) / 64.0], axis=2)  # (H,W,2)

        Ht, Wt = target_shape
        H0, W0 = flow.shape[:2]
        if (H0 != Ht) or (W0 != Wt):
            sx = Wt / float(W0); sy = Ht / float(H0)
            flow = cv2.resize(flow, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
            flow[:, :, 0] *= sx
            flow[:, :, 1] *= sy
        return flow

    def flowreader(self,flow_path):
        with Image.open(flow_path) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            flow = np.frombuffer(
                np.array(depth_pil, dtype=np.uint16), dtype=np.float16
            ).astype(np.float32).reshape((depth_pil.size[1], depth_pil.size[0]))
        flow_res = np.stack([flow[:,:flow.shape[1]//2], flow[:,flow.shape[1]//2:]],axis=-1)
        return flow_res
    
    def _init_points_grid(self, w, h, num_tracks):
        """
        规则网格初始化，尽量均匀覆盖；不随机。
        """
        if num_tracks <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        # 估计列/行数，留一定边距防越界
        cols = max(1, int(np.ceil(np.sqrt(num_tracks * (w / (h + 1e-6))))))
        rows = max(1, int(np.ceil(num_tracks / cols)))
        xs = np.linspace(8, max(8, w - 9), cols, dtype=np.float32)
        ys = np.linspace(8, max(8, h - 9), rows, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        return pts[:num_tracks]

    def _sample_flow_nn(self, flow_hw2: np.ndarray, xy_n2: np.ndarray, mask_hw: np.ndarray = None):
        """
        用最近邻采样光流(更简单更快) + 可选mask约束:
        - flow_hw2: (H, W, 2)，像素位移 (u, v)
        - xy_n2:    (N, 2)，像素坐标 (x, y)
        - mask_hw:  (H, W)，可选，>0 表示光流有效
        返回: disp_n2: (N, 2); valid_n: (N,)
        """
        assert flow_hw2.ndim == 3 and flow_hw2.shape[2] >= 2
        H, W = flow_hw2.shape[:2]
        x = np.rint(xy_n2[:, 0]).astype(np.int32)
        y = np.rint(xy_n2[:, 1]).astype(np.int32)
        valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        disp = np.zeros((xy_n2.shape[0], 2), dtype=np.float32)
        if np.any(valid):
            disp[valid] = flow_hw2[y[valid], x[valid], :2]
            # 如果传了mask，要求 mask=1 的点才有效
            if mask_hw is not None:
                mask_ok = mask_hw[y[valid], x[valid]] > 0
                tmp = np.zeros_like(valid)
                tmp[valid] = mask_ok
                valid = tmp
            bad = ~np.isfinite(disp).all(axis=1)
            valid = valid & (~bad)
            disp[~valid] = 0.0
        return disp, valid

    def _extract_synthetic_tracks(self, seq_meta: Dict, frame_indices: List[int]):
        """
        用前向光流生成简易轨迹（最近邻采样）：
        - 轨迹平面尺寸 = 第一张可用光流的 (H, W)
        - 若后续某帧光流尺寸不同，则 resize 到 (H, W) 并按比例缩放位移
        - 起点：规则网格
        - 无效：置 0，mask=False
        """
        N = int(self.track_num)
        T = len(frame_indices)
        tracks = np.zeros((N, T, 2), dtype=np.float32)
        masks  = np.zeros((N, T), dtype=bool)
        flow_forward_dir = os.path.join(seq_meta['seq_path'], 'flow_forward')
        flow_forward_mask_dir = os.path.join(seq_meta['seq_path'], 'flow_forward_mask')
        seq_name_prefix  = seq_meta['seq_name_prefix']
        # 先找第一张可用的光流，确定 H、W
        fp = os.path.join(flow_forward_dir, f'{seq_name_prefix}_{frame_indices[0]:04d}.png')
        flow = self.flowreader(fp)
        H, W = flow.shape[:2]
        # 起点：规则网格（以 H,W 为准）
        start_xy = self._init_points_grid(W, H, N).astype(np.float32)
        tracks[:, 0, :] = start_xy
        masks[:, 0] = True
        # 用第一张光流推进（如果它并不是 ti=1 的那张，我们也只从有光流的 ti 开始推进）
        for ti in range(1, T):
            prev_xy    = tracks[:, ti - 1, :].copy()
            prev_valid = masks[:, ti - 1].copy()
            # 默认无效
            tracks[:, ti, :] = 0.0
            masks[:, ti] = False
            prev_idx = frame_indices[ti - 1]
            fp = os.path.join(flow_forward_dir, f'{seq_name_prefix}_{prev_idx:04d}.png')
            flow = self.flowreader(fp)
            fp_mask = os.path.join(flow_forward_mask_dir, f'{seq_name_prefix}_{prev_idx:04d}.png')
            mask_hw = np.array(Image.open(fp_mask))
            disp, samp_valid = self._sample_flow_nn(flow, prev_xy, mask_hw)
            good = prev_valid & samp_valid
            if np.any(good):
                cur_xy = prev_xy[good] + disp[good]
                inb = (cur_xy[:, 0] >= 0) & (cur_xy[:, 0] < W) & \
                    (cur_xy[:, 1] >= 0) & (cur_xy[:, 1] < H)
                idx = np.where(good)[0][inb]
                if idx.size > 0:
                    tracks[idx, ti, :] = cur_xy[inb]
                    masks[idx, ti] = True
        # 清 NaN/Inf
        bad = ~np.isfinite(tracks).all(axis=2)
        if np.any(bad):
            tracks[bad] = 0.0
            masks[bad] = False
        return tracks, masks

T = np.array([[0, -1, 0],
              [0, 0, -1],
              [1, 0, 0]])

def build_w2c_1(rotation, translation, input_convention: str = "world_to_camera"):
    # 错误
    """
    reproj err px  mean/med/max: 0.0014177499385983443 6.877210492598751e-06 1.7975406157481109
    reproj err px  mean/med/max: 0.0004919260792608015 5.729807110057891e-06 1.2135114370936697
    reproj err px  mean/med/max: 7.4255592742712165e-06 6.639260873481224e-06 3.119919298740216e-05
    reproj err px  mean/med/max: 6.619161878800264e-06 5.940166954222068e-06 2.9191652512471124e-05
    reproj err px  mean/med/max: 8.808325151818636e-06 8.017237904956653e-06 2.9908518888514434e-05
    """
    R = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float32).reshape(3)
    if input_convention == "world_to_camera":
        # W2C in OpenGL camera frame -> apply camera-basis change on the LEFT
        R_wc_cv = T @ R
        t_wc_cv = T @ t
    elif input_convention == "camera_to_world":
        # C2W in OpenGL camera frame -> first convert to OpenCV camera frame,
        R_cw_cv = R @ T           # change camera basis on the RIGHT for C2W
        t_cw_cv = t               # position is in world coords; unchanged
        R_wc_cv = R_cw_cv.T
        t_wc_cv = -R_wc_cv @ t_cw_cv
    else:
        raise ValueError(f"Unknown input_convention: {input_convention}")
    T_cw = np.eye(4, dtype=np.float32)
    T_cw[:3, :3] = R_wc_cv
    T_cw[:3, 3]  = t_wc_cv
    return T_cw

def build_w2c_2(R, t, input_convention: str = "world_to_camera"):
    """
    reproj err px  mean/med/max: 7.4255592742712165e-06 6.639260873481224e-06 3.119919298740216e-05
    reproj err px  mean/med/max: 6.619161878800264e-06 5.940166954222068e-06 2.9191652512471124e-05
    reproj err px  mean/med/max: 8.808325151818636e-06 8.017237904956653e-06 2.9908518888514434e-05
    reproj err px  mean/med/max: 0.0007295784322285104 1.1477803985316165e-05 1.7914122948981723
    """
    T_cw = np.eye(4, dtype=np.float32)
    T_cw[:3,:3] = np.array(R).T
    T_cw[:3,3] = -np.array(R).T @ np.array(t)
    return T_cw

def build_w2c_3(rotation, translation,
                    input_convention: str = "world_to_camera") -> np.ndarray:
    """
    reproj err px  mean/med/max: 8.053808848259911e-06 6.840353000421476e-06 4.4000820998140896e-05
    reproj err px  mean/med/max: 0.00022193060929452207 5.3415553579053744e-06 0.5392328918148023
    reproj err px  mean/med/max: 7.636845709825266e-06 6.778956481548194e-06 3.276455595571716e-05
    reproj err px  mean/med/max: 7.03172396835381e-06 6.049693694409601e-06 4.2078985261905054e-05
    """
    R = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float32).reshape(3, )
    R_wc, t_wc = R, t
    T_cw = np.eye(4, dtype=np.float32)
    T_cw[:3, :3] = R_wc
    T_cw[:3, 3]  = t_wc
    return T_cw

def build_w2c_4(rotation, translation,
                    input_convention: str = "world_to_camera") -> np.ndarray:
    """
    reproj err px  mean/med/max: 8.053808848259911e-06 6.840353000421476e-06 4.4000820998140896e-05
    reproj err px  mean/med/max: 0.00022193060929452207 5.3415553579053744e-06 0.5392328918148023
    reproj err px  mean/med/max: 7.636845709825266e-06 6.778956481548194e-06 3.276455595571716e-05
    reproj err px  mean/med/max: 7.03172396835381e-06 6.049693694409601e-06 4.2078985261905054e-05
    Qianqian Wang
    """
    R = torch.tensor(rotation, dtype=torch.float)
    T = torch.tensor(translation, dtype=torch.float)
    R_pytorch3d = R.clone()
    T_pytorch3d = T.clone()
    T_pytorch3d[..., :2] *= -1
    R_pytorch3d[..., :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.T
    pose = np.eye(4)
    if input_convention == "world_to_camera":
        pose[:3, :3] = R.numpy()
        pose[:3, 3] = tvec.numpy()
    elif input_convention == "camera_to_world":
        pose[:3, :3] = R.numpy().T
        pose[:3, 3] = -R.numpy().T @ tvec.numpy()
    return pose

def convert_ndc_to_pixel_intrinsics(entry_viewpoint, image_size):
    image_height, image_width = image_size
    f_x_ndc, f_y_ndc = map(float, entry_viewpoint['focal_length'])
    c_x_ndc, c_y_ndc = map(float, entry_viewpoint['principal_point'])
    # Compute half image size
    half_image_size_wh_orig = np.array([image_width, image_height]) / 2.0
    # Determine rescale factor based on intrinsics_format
    if entry_viewpoint['intrinsics_format'].lower() == "ndc_norm_image_bounds":
        rescale = half_image_size_wh_orig  # [image_width/2, image_height/2]
    elif entry_viewpoint['intrinsics_format'].lower() == "ndc_isotropic":
        rescale = np.min(half_image_size_wh_orig)  # scalar value
    else:
        raise ValueError(f"Unknown intrinsics format: {intrinsics_format}")
    # Convert principal point from NDC to pixel coordinates
    principal_point_px = half_image_size_wh_orig - np.array([c_x_ndc, c_y_ndc]) * rescale
    focal_length_px = np.array([f_x_ndc, f_y_ndc]) * rescale
    # Construct the intrinsics matrix in pixel coordinates
    K_pixel = np.array([
        [focal_length_px[0], 0,                principal_point_px[0]],
        [0,                 focal_length_px[1], principal_point_px[1]],
        [0,                 0,                 1]
    ])
    return K_pixel

def convert_ndc_to_pixel_intrinsics2(entry_viewpoint, image_size):
    image_height, image_width = image_size
    f_x_ndc, f_y_ndc = map(float, entry_viewpoint['focal_length'])
    c_x_ndc, c_y_ndc = map(float, entry_viewpoint['principal_point'])
    # 注意：PyTorch3D 的 OpenCV 约定是 (W-1)/2, (H-1)/2 作为像素中心
    half_image_size_wh_orig = np.array([(image_width - 1) / 2.0,
                                        (image_height - 1) / 2.0])
    fmt = entry_viewpoint['intrinsics_format'].lower()
    if fmt == "ndc_norm_image_bounds":
        rescale = half_image_size_wh_orig  # [ (W-1)/2 , (H-1)/2 ]
    elif fmt == "ndc_isotropic":
        rescale = np.min(half_image_size_wh_orig)  # scalar value
    else:
        raise ValueError(f"Unknown intrinsics format: {fmt}")
    # focal length to pixel
    focal_length_px = np.array([f_x_ndc, f_y_ndc]) * rescale
    # principal point to pixel
    principal_point_px = half_image_size_wh_orig - np.array([c_x_ndc, c_y_ndc]) * rescale
    K_pixel = np.array([
        [focal_length_px[0], 0.0,              principal_point_px[0]],
        [0.0,                focal_length_px[1], principal_point_px[1]],
        [0.0,                0.0,              1.0]
    ], dtype=np.float32)
    return K_pixel

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
    print("Testing DynamicReplica dataset...")
    
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
            self.track_num = 256
            self.augs = {'scales': [0.8, 1.2]}
    
    config = SimpleConfig()
    
    # Create dataset
    dataset = DynamicReplicaDataset(
        common_conf=config,
        dataset_location='/Users/gracechen/Desktop/4D_FM/dynamicrepl',  # test with your local path
        annotations_file='/workspace/data/kaichen/data/dynreplica/dynamic_stereo/dynamic_replica_data/train/frame_annotations_train.jgz',  # update path as needed
        sequence_names=['f14caa-3_obj_source_right'],
        quick=True,
        verbose=True,
        len_train=100
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test getting a sample
        sample = dataset.get_data(seq_index=0, img_per_seq=2, aspect_ratio=1.0)
        print(f"Sample keys: {sample.keys()}")
        print(f"Images shape: {[img.shape for img in sample['images']]}")
        print(f"IDs: {sample['ids']}")
        print("Test completed successfully!")
    else:
        print("No data found - check dataset path")