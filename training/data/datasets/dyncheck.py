# dyncheck.py
import sys
import os
import glob
import random
import numpy as np
import cv2
import json
import logging
import re
from typing import Optional, List, Dict, Any
import os.path as osp
import pdb
# Add paths for imports
sys.path.append('../../..')
import math
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
    elif T_cw.shape != (4,4):
        raise ValueError(f"T_cw must be 3x4 or 4x4, got {T_cw.shape}")
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

class DyncheckDataset(BaseDataset):
    """
    DYNCHECK dataset loader for VGGT testing pipeline.
    
    This dataset loads DYNCHECK sequences which contain:
    - RGB images (category/rgb/*.png)
    - Depth maps (category/depth/*.npy)
    - Camera parameters (category/camera/*.json)
    - Covisible masks (category/covisible/*.png)
    - Scene metadata (category/metadata.json, scene.json, extra.json)
    """
    def __init__(
        self,
        common_conf,
        dataset_location: str = '/data/dyncheck',
        split: str = 'val',  # testing only
        sequence_names: Optional[List[str]] = None,
        strides: List[int] = [1, 2, 3, 4],
        clip_step: int = 1,
        min_num_images: int = 4,
        len_train: int = 1000,
        len_test: int = 1000,
        quick: bool = False,
        verbose: bool = False,
        dist_type: Optional[str] = None,
        resolution: str = '1x',
        **kwargs
    ):
        """
        Initialize DYNCHECK dataset.
        Args:
            common_conf: Common configuration object
            dataset_location: Path to DYNCHECK dataset root
            split: Dataset split (only 'val' for testing)
            sequence_names: List of sequence names to load (for compatibility)
            strides: List of temporal strides to use
            clip_step: Step size for sampling clips
            min_num_images: Minimum number of images per sequence
            len_train: Training dataset length (unused for testing)
            len_test: Test dataset length
            quick: Quick mode for testing (uses subset)
            verbose: Verbose logging
            dist_type: Distribution type for stride sampling
            resolution: Resolution subdirectory to use ('1x' or '2x', defaults to '2x')
        """
        print(f'Loading DYNCHECK dataset from {dataset_location}...')
        print(f'Note: This dataset is for testing purposes only')
        print(f'Using resolution: {resolution}')
        super().__init__(common_conf=common_conf)
        # Dataset configuration
        self.dataset_location = dataset_location
        self.split = split
        self.sequence_names = sequence_names
        self.strides = strides
        self.clip_step = clip_step
        self.min_num_images = min_num_images
        self.quick = quick
        self.verbose = False
        self.dist_type = dist_type
        self.resolution = resolution
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
        self.img_size = getattr(common_conf, 'img_size', 224)
        if self.verbose:
            print(f"Image size set to: {self.img_size}")
            print(f"Testing dataset for DYNCHECK evaluation")
        # Load sequences and organize data
        self._load_sequences()
        # Apply stride distribution if specified
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)
            
        print(f'Loaded {len(self.sequence_list)} sequences with {len(self.clip_data)} clips')
        # Validate dataset
        self._validate_dataset()

    def _load_sequences(self):
        """Load and organize DYNCHECK sequences."""
        # Initialize storage
        self.sequence_list = []
        self.sequence_metadata = {}
        self.clip_data = []
        # Detect dataset structure and load accordingly
        self._load_simple_structure()
        print(f'Successfully loaded {len(self.sequence_list)} sequences')

    def _load_simple_structure(self):
        """Load DYNCHECK scenes from simple structure (category/rgb/ directly)."""
        if not os.path.exists(self.dataset_location):
            print(f'Dataset location does not exist: {self.dataset_location}')
            return
        # Find all category directories
        category_dirs = []
        for item in os.listdir(self.dataset_location):
            item_path = os.path.join(self.dataset_location, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if this directory contains rgb/
                rgb_check = os.path.join(item_path, 'rgb')
                if os.path.exists(rgb_check):
                    category_dirs.append(item)
        if self.verbose:
            print(f'Found {len(category_dirs)} category directories in simple structure')
        for category in sorted(category_dirs):
            category_path = os.path.join(self.dataset_location, category)
            seq_name = category  # Use category name directly
            if self.verbose:
                print(f'Processing simple structure sequence: {seq_name} at {category_path}')
            scene_data = self._load_scene_data_simple(category_path, seq_name)
            if scene_data:
                self.sequence_list.append(seq_name)
                self.sequence_metadata[seq_name] = scene_data
                self._generate_clips_for_sequence(seq_name, len(scene_data['frame_info']), scene_data['frame_info'])

    def _load_scene_data_simple(self, scene_path: str, seq_name: str):
        """Load data for a single DYNCHECK scene using simple structure."""
        # In simple structure, metadata files might not exist - that's okay
        metadata_file = os.path.join(scene_path, 'metadata.json')
        scene_file = os.path.join(scene_path, 'scene.json')
        extra_file = os.path.join(scene_path, 'extra.json')
        metadata = {}
        scene_info = {}
        extra_info = {}
        # Try to load metadata files, but don't fail if they don't exist
        for file_path, data_dict, name in [
            (metadata_file, metadata, 'metadata'),
            (scene_file, scene_info, 'scene'),
            (extra_file, extra_info, 'extra')
        ]:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        loaded_data = json.load(f)
                        data_dict.update(loaded_data)
                    if self.verbose:
                        print(f'  Loaded {name}.json')
                except Exception as e:
                    if self.verbose:
                        print(f'  Warning: Could not load {name}.json for {seq_name}: {e}')
        # Get available RGB files - try with and without resolution subdirectories
        rgb_path = os.path.join(scene_path, 'rgb')
        rgb_files = []
        # Try resolution subdirectories first
        for res in [self.resolution]:
            rgb_res_path = os.path.join(rgb_path, res)
            if os.path.exists(rgb_res_path):
                rgb_files = sorted([
                    f for f in os.listdir(rgb_res_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                if rgb_files:
                    rgb_path = rgb_res_path
                    if self.verbose:
                        print(f'  Using RGB resolution: {res}')
                    break
        # If no resolution subdirectory works, try direct rgb path
        if not rgb_files and os.path.exists(rgb_path):
            rgb_files = sorted([
                f for f in os.listdir(rgb_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            if self.verbose and rgb_files:
                print(f'  Using direct RGB path (no resolution subdirectory)')
        if len(rgb_files) < self.min_num_images:
            if self.verbose:
                print(f'  Insufficient RGB images: {len(rgb_files)} < {self.min_num_images}')
            return None
        frame_info = []
        for rgb_file in rgb_files:
            try:
                basename = os.path.splitext(rgb_file)[0]  # Remove extension
                # Try different filename parsing strategies
                frame_id = None
                # Strategy 1: filename contains underscore (frame_123.png)
                if '_' in basename:
                    parts = basename.split('_')
                    for part in reversed(parts):  # Check from end to start
                        try:
                            frame_id = int(part)
                            break
                        except ValueError:
                            continue
                # Strategy 2: filename is just a number (123.png)
                if frame_id is None:
                    try:
                        frame_id = int(basename)
                    except ValueError:
                        pass
                # Strategy 3: extract any number from filename
                if frame_id is None:
                    numbers = re.findall(r'\d+', basename)
                    if numbers:
                        frame_id = int(numbers[-1])  # Use last number found
                if frame_id is None:
                    if self.verbose:
                        print(f'  Warning: Could not parse frame ID from {rgb_file}, skipping')
                    continue
                # Verify the RGB file actually exists
                rgb_path_check = os.path.join(rgb_path, rgb_file)
                if not os.path.exists(rgb_path_check):
                    continue
                # Check depth file existence if depth loading is enabled
                depth_file_exists = True
                if self.load_depth:
                    depth_file_exists = self._check_depth_file_exists(scene_path, rgb_file, basename)
                    if not depth_file_exists and self.verbose:
                        print(f'  Note: No depth file found for {rgb_file}')
                # Include frame regardless of depth availability (create placeholder if needed)
                if basename[0] != '0':
                    continue
                frame_info.append({
                    'frame_id': frame_id,
                    'filename': rgb_file,
                    'full_id': basename,
                    'has_depth': depth_file_exists
                })
            except Exception as e:
                if self.verbose:
                    print(f'  Warning: Error processing {rgb_file}: {e}')
                continue
        # Sort by frame_id
        frame_info = sorted(frame_info, key=lambda x: x['frame_id'])
        if len(frame_info) < self.min_num_images:
            if self.verbose:
                print(f'  Insufficient valid frames: {len(frame_info)} < {self.min_num_images}')
            return None
        # Check resource availability
        depth_path = os.path.join(scene_path, 'depth')
        depth_available = os.path.exists(depth_path)
        camera_path = os.path.join(scene_path, 'camera')
        camera_files_available = os.path.exists(camera_path)
        # Return scene data
        return {
            'seq_path': scene_path,
            'rgb_path': rgb_path,  # Store the resolved RGB path
            'num_frames': len(frame_info),
            'frame_info': frame_info,
            'metadata': metadata,
            'scene_info': scene_info,
            'extra_info': extra_info,
            'depth_available': depth_available,
            'camera_files_available': camera_files_available
        }

    def _check_depth_file_exists(self, scene_path: str, rgb_file: str, basename: str) -> bool:
        """Check if corresponding depth file exists for an RGB file."""
        # Generate possible depth filenames
        depth_candidates = []
        frame_number = rgb_file.split('_', 1)[1]  # Get part after first underscore
        depth_candidates.append(f"0_{frame_number}".replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
        
    def _load_depth_for_frame(self, seq_meta: Dict, frame_info: Dict) -> np.ndarray:
        """Load depth map for a specific frame, handling different structures and naming conventions."""
        rgb_filename = frame_info['filename']
        basename = frame_info['full_id']
        depth_filename = f"{basename}.npy"
        # Try to find depth file in different locations
        depth_base_path = os.path.join(seq_meta['seq_path'], 'depth')
        depth_path = os.path.join(depth_base_path, self.resolution, depth_filename)     
        if os.path.exists(depth_path):
            depth_map = np.load(depth_path).astype(np.float32)
            if self.verbose:
                print(f"   Found depth: {depth_path}")
            return depth_map

    def _generate_clips_for_sequence(self, seq_name: str, num_frames: int, frame_info: List[Dict]):
        """Generate testing clips for the sequence with different strides."""
        # Create frame indices from frame_info for compatibility
        frame_indices = list(range(num_frames))
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

    def convert_pt3d_RT_to_opencv(self, Rot, Trans):
        """
        Convert Point3D extrinsic matrices to OpenCV convention.
        Args:
            Rot: 3D rotation matrix in Point3D format
            Trans: 3D translation vector in Point3D format
        Returns:
            extri_opencv: 3x4 extrinsic matrix in OpenCV format
        """
        rot_pt3d = np.array(Rot)
        trans_pt3d = np.array(Trans)
        trans_pt3d[:2] *= -1
        rot_pt3d[:, :2] *= -1
        rot_pt3d = rot_pt3d.transpose(1, 0)
        extri_opencv = np.hstack((rot_pt3d, trans_pt3d[:, None]))
        return extri_opencv

    def _load_camera_parameters(self, seq_meta: Dict, frame_info_list: List[Dict], target_size=None) -> tuple:
        """Load camera parameters for the specified frames."""
        intrinsics = []
        extrinsics = []
        for frame_info in frame_info_list:
            frame_id = frame_info['frame_id']
            full_id = frame_info['full_id']
            # Try to load from individual camera files first
            camera_file = os.path.join(
                seq_meta['seq_path'], 'camera', f'{full_id}.json'
            )
            if os.path.exists(camera_file):
                with open(camera_file, 'r') as f:
                    cam_data = json.load(f)
                # Extract intrinsics and extrinsics from DYNCHECK format
                focal_length = cam_data['focal_length']
                principal_point = cam_data['principal_point']  # [cx, cy]
                # Construct intrinsic matrix
                original_intrinsic = np.array([
                    [focal_length, 0, principal_point[0]],
                    [0, focal_length, principal_point[1]],
                    [0, 0, 1]
                ], dtype=np.float32)
                # Extract position and orientation for extrinsics
                position = np.array(cam_data.get('position', [0, 0, 0]))
                orientation = np.array(cam_data.get('orientation', np.eye(3)))
                extrinsic = np.block(
                    [
                        [orientation, -orientation @ position[:, None]],
                        [np.zeros((1, 3)), np.ones((1, 1))],
                    ]
                ).astype(np.float32)
                intrinsics.append(original_intrinsic)
                extrinsics.append(extrinsic)
        return intrinsics, extrinsics

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
        # Validate inputs and handle edge cases
        if len(self.clip_data) == 0:
            raise ValueError("No clips available in dataset")
        if self.inside_random or seq_index is None:
            seq_index = random.randint(0, len(self.clip_data) - 1)
        # Ensure seq_index is within valid range
        seq_index = max(0, min(seq_index, len(self.clip_data) - 1))
        # Get clip information
        clip_info = self.clip_data[seq_index]
        seq_name = clip_info['seq_name']
        base_frame_indices = clip_info['frame_indices']
        available_frames = clip_info['available_frames']
        # Get sequence metadata
        if seq_name not in self.sequence_metadata:
            raise ValueError(f"Sequence {seq_name} not found in metadata")
        seq_meta = self.sequence_metadata[seq_name]
        # Validate frame indices
        max_frame_idx = len(seq_meta['frame_info']) - 1
        if any(idx > max_frame_idx for idx in base_frame_indices):
            # Fix invalid frame indices
            base_frame_indices = [min(idx, max_frame_idx) for idx in base_frame_indices]
        # Ensure base_frame_indices is not empty
        if img_per_seq:
            frame_indices, FLAG = self._contiguous_window_impr(available_frames, base_frame_indices, img_per_seq)
        else:
            frame_indices = base_frame_indices
        # Get corresponding frame info
        try:
            frame_info_list = [seq_meta['frame_info'][i] for i in frame_indices]
        except IndexError as e:
            raise ValueError(f"Frame index out of range: {e}")
        # Get target image shape
        target_image_shape = self.get_target_shape(aspect_ratio)
        # Load camera parameters
        try:
            intrinsics, extrinsics = self._load_camera_parameters(
                seq_meta, frame_info_list, target_image_shape
            )
        except Exception as e:
            # Fallback to default parameters
            if self.verbose:
                print(f"Warning: Using default camera parameters due to error: {e}")
            intrinsics = [self._create_default_intrinsics_for_dyncheck(target_image_shape) 
                         for _ in frame_info_list]
            extrinsics = [self._create_default_extrinsics() for _ in frame_info_list]
        # Load and process images
        images = []
        depths = []
        processed_extrinsics = []
        processed_intrinsics = []
        cam_points = []
        world_points = []
        point_masks = []
        for i, frame_info in enumerate(frame_info_list):
            # Load RGB image (using resolution subdirectory)
            rgb_path = os.path.join(
                seq_meta['rgb_path'], frame_info['filename']
            )
            # Verify file exists before loading
            if not os.path.exists(rgb_path):
                raise FileNotFoundError(f"RGB file not found: {rgb_path}")
            image = read_image_cv2(rgb_path)
            if image is None:
                raise ValueError(f"Could not load image: {rgb_path}")
            # Load depth map
            depth_map = None
            if self.load_depth and seq_meta['depth_available']:
                depth_map = self._load_depth_for_frame(seq_meta, frame_info)
                if depth_map is not None:
                    try:
                        # Apply threshold to remove invalid depth values
                        depth_map = threshold_depth_map(depth_map, max_percentile=98)
                        if self.verbose:
                            print(f"   Loaded depth shape: {depth_map.shape}, valid pixels: {np.sum(depth_map > 0)}")
                    except Exception as e:
                        if self.verbose:
                            print(f"   Error processing depth map: {e}")
                        depth_map = None
            original_size = np.array(image.shape[:2])            
            # Process image using VGGT pipeline
            (
                processed_image,
                processed_depth,
                processed_extri,
                processed_intri,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extrinsics[i],
                intrinsics[i],
                original_size,
                target_image_shape,
                track=None,
                filepath=rgb_path,
                # resize_ours=True,
            )
            # check = check_coord_system(world_coords_points.reshape(-1, 3)[point_mask.reshape(-1)], processed_extri)
            # print(check, '===========check===========')
            # matrix_check(processed_image, processed_depth, processed_intri, processed_extri)
            # print(np.mean(processed_depth), '===========processed_depth===========', np.min(processed_depth), '===========np.min(processed_depth)===========', np.max(processed_depth), '===========np.max(processed_depth)===========')
            images.append(processed_image)
            depths.append(processed_depth)
            processed_extrinsics.append(processed_extri)
            processed_intrinsics.append(processed_intri)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
        # DYNCHECK doesn't have trajectory data for testing
        tracks = np.zeros((len(images), self.track_num, 2))
        track_masks = np.zeros((len(images), self.track_num))
        # Return in VGGT format
        return {
            "seq_name": f"dyncheck_{seq_name}",
            "ids": np.array([info['frame_id'] for info in frame_info_list]),
            "images": images,
            "depths": depths,
            "extrinsics": processed_extrinsics,
            "intrinsics": processed_intrinsics,
            "abandon_pose": False,
            "abandon_geometry": False,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "tracks": tracks,
            "track_masks": track_masks,
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

    def _validate_dataset(self):
        """Validate that the dataset has been loaded correctly and has usable data."""
        if len(self.sequence_list) == 0:
            print("Warning: No sequences found in dataset")
            return False
        if len(self.clip_data) == 0:
            print("Warning: No clips generated from sequences")
            return False
        # Check that at least one sequence has valid data
        valid_sequences = 0
        for seq_name in self.sequence_list:
            if seq_name in self.sequence_metadata:
                seq_meta = self.sequence_metadata[seq_name]
                if seq_meta.get('num_frames', 0) >= self.min_num_images:
                    valid_sequences += 1
        if valid_sequences == 0:
            print("Warning: No sequences with sufficient frames found")
            return False
        if self.verbose:
            print(f"Dataset validation passed: {valid_sequences}/{len(self.sequence_list)} sequences are valid")
        return True


if __name__ == "__main__":
    # Simple test
    print("Testing DYNCHECK dataset...")
    
    # Create a simple config object
    class SimpleConfig:
        def __init__(self):
            self.img_size = 224
            self.patch_size = 14
            self.training = False  # Testing only
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
    dataset = DyncheckDataset(
        common_conf=config,
        dataset_location='/workspace/data/kaichen/data/test/DYNCHECK',  # Updated to use the main directory
        quick=True,
        verbose=True,
        len_test=100,
        resolution='2x'  # Use 2x resolution by default
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test getting a sample
        sample = dataset.get_data(seq_index=0, img_per_seq=4, aspect_ratio=1.0)
        print(f"Sample keys: {sample.keys()}")
        print(f"Images shape: {[img.shape for img in sample['images']]}")
        print(f"IDs: {sample['ids']}")
        print("Test completed successfully!")
    else:
        print("No data found - check dataset path") 