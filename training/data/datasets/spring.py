import sys
import os
import glob
import random
import numpy as np
import cv2
import h5py
import logging
from typing import Optional, List, Dict, Any
import os.path as osp
from PIL import Image
from matplotlib import cm, colors
import open3d as o3d
import struct

import pdb
sys.path.append('../../..')

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

class SpringDataset(BaseDataset):
    
    def __init__(
        self,
        common_conf,
        dataset_location: str = '/data/spring',
        split: str = 'train',
        sequence_names: Optional[List[str]] = None,  
        strides: List[int] = [1, 2, 3],
        clip_step: int = 2,
        min_num_images: int = 8,
        len_train: int = 10000,
        len_test: int = 1000,
        quick: bool = False,
        verbose: bool = False,
        dist_type: Optional[str] = None,
        baseline: float = 0.065,  # Stereo baseline in meters
        use_zip_format: bool = None,  # Auto-detect format if None
        apply_coord_transform: bool = False,
        eyes: str = "left",
        **kwargs
    ):
        print(f'Loading Spring dataset from {dataset_location}...')
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
        self.baseline_ori = baseline
        self.apply_coord_transform = apply_coord_transform
        self.eyes = eyes # right
        print(f'Using {self.eyes} eyes', '==========================================')
        # Auto-detect format if not specified (must be done before sequence discovery)
        if use_zip_format is None:
            self.use_zip_format = self._detect_format()
        else:
            self.use_zip_format = use_zip_format
            
        if self.verbose:
            format_type = "ZIP" if self.use_zip_format else "Sample"
            print(f"Using {format_type} format")
        
        # Discover sequences after format is determined
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
        # For ZIP format, we don't discover individual sequences  
        if self.use_zip_format:
            return []
            
        # For sample format, look in the split subdirectory
        split_path = os.path.join(self.dataset_location, self.split)
        if not os.path.exists(split_path):
            if self.verbose:
                print(f'Split directory does not exist: {split_path}')
            return []
            
        return self._discover_sequences_generic(
            dataset_location=split_path,
            required_subdirs=[f'frame_{self.eyes}', f'disp1_{self.eyes}'],
            required_files=[],  # No specific files required at sequence level
            image_subdir=f'frame_{self.eyes}',
            image_pattern=f'frame_{self.eyes}_*.png',
            min_num_images=self.min_num_images,
            verbose=self.verbose
        )

    def _detect_format(self) -> bool:
        try:
            # Check for ZIP format indicators
            zip_files = [
                f'{self.split}_frame_{self.eyes}.zip',
                f'{self.split}_disp1_{self.eyes}.zip',
                f'{self.split}_cam_data.zip'
            ]
            
            zip_count = sum(1 for zip_file in zip_files 
                           if os.path.exists(os.path.join(self.dataset_location, zip_file)))
            
            # Check for sample format indicators
            sample_path = os.path.join(self.dataset_location, self.split)
            sample_exists = os.path.exists(sample_path)
            
            if zip_count >= 2:  # At least 2 zip files found
                return True
            elif sample_exists:
                return False
            else:
                # Default to sample format
                if self.verbose:
                    print("Could not detect format, defaulting to sample format")
                return False
                
        except Exception as e:
            if self.verbose:
                print(f"Error detecting format: {e}, defaulting to sample format")
            return False

    def _load_sequences(self):
        if self.use_zip_format:
            self._load_sequences_zip()
        else:
            self._load_sequences_sample()

    def _load_sequences_sample(self):
        """Load sequences from sample directory structure."""
        # Initialize storage
        self.sequence_list = []
        self.sequence_metadata = {}
        self.clip_data = []
        
        # Process each sequence
        for seq_name in self.sequence_names:
            seq_path = os.path.join(self.dataset_location, self.split, seq_name)
            
            if not os.path.exists(seq_path):
                if self.verbose:
                    print(f'Sequence path does not exist: {seq_path}')
                continue
                
            if self.verbose:
                print(f'Processing sequence: {seq_name}')
                
            # Check for required directories
            frame_all_path = os.path.join(seq_path, f'frame_{self.eyes}')
            disp1_all_path = os.path.join(seq_path, f'disp1_{self.eyes}')
            cam_data_path = os.path.join(seq_path, 'cam_data')
            
            if not (os.path.isdir(frame_all_path) and os.path.isdir(disp1_all_path)):
                if self.verbose:
                    print(f'  Skipping {seq_name}: missing frame_{self.eyes} or disp1_{self.eyes} directory')
                continue
                
            rgb_files = sorted([
                f for f in os.listdir(frame_all_path) 
                if f.startswith(f'frame_{self.eyes}_') and f.endswith('.png')
            ])
            
            if len(rgb_files) < self.min_num_images:
                if self.verbose:
                    print(f'  Skipping {seq_name}: insufficient frames ({len(rgb_files)})')
                continue
            
            # Extract frame indices from RGB files
            frame_indices = []
            for rgb_file in rgb_files:
                try:
                    frame_idx_str = rgb_file.split('_')[-1].split('.')[0]
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
            
            # Load camera parameters
            camera_params = self._load_camera_parameters(cam_data_path, num_frames)
            
            # Store sequence metadata
            self.sequence_list.append(seq_name)
            self.sequence_metadata[seq_name] = {
                'seq_path': seq_path,
                'num_frames': num_frames,
                'frame_indices': frame_indices,
                'rgb_files': rgb_files,
                'camera_params': camera_params,
                'format': 'sample'
            }
            
            # Generate clips for this sequence
            self._generate_clips_for_sequence(seq_name, num_frames, frame_indices)
            
        print(f'Successfully loaded {len(self.sequence_list)} sequences from sample format')

    def _load_sequences_zip(self):
        """Load sequences from ZIP file structure."""
        # Initialize storage
        self.sequence_list = []
        self.sequence_metadata = {}
        self.clip_data = []
        
        # Check for required ZIP files
        required_zips = [
            f'{self.split}_frame_{self.eyes}.zip',
            f'{self.split}_disp1_{self.eyes}.zip'
        ]
        
        missing_zips = []
        for zip_file in required_zips:
            if not os.path.exists(os.path.join(self.dataset_location, zip_file)):
                missing_zips.append(zip_file)
        
        if missing_zips:
            if self.verbose:
                print(f'Missing required ZIP files: {missing_zips}')
            return
        
        # For ZIP format, we create a single sequence representing the entire dataset
        seq_name = f'spring_{self.split}'
        
        if self.verbose:
            print(f'Processing ZIP-based dataset: {seq_name}')
        
        try:
            import zipfile
            
            frame_all_zip = os.path.join(self.dataset_location, f'{self.split}_frame_{self.eyes}.zip')
            with zipfile.ZipFile(frame_all_zip, 'r') as zf:
                frame_files = [f for f in zf.namelist() if f.endswith('.png')]
                
            # Extract frame indices
            frame_indices = []
            for frame_file in frame_files:
                try:
                    basename = os.path.basename(frame_file)
                    if f'frame_{self.eyes}_' in basename:
                        frame_idx_str = basename.split('_')[-1].split('.')[0]
                    else:
                        frame_idx_str = basename.split('.')[0]
                    frame_idx = int(frame_idx_str)
                    frame_indices.append(frame_idx)
                except (ValueError, IndexError):
                    if self.verbose:
                        print(f'  Warning: Could not parse frame index from {frame_file}')
                    continue
            
            frame_indices = sorted(frame_indices)
            num_frames = len(frame_indices)
            
            if num_frames < self.min_num_images:
                if self.verbose:
                    print(f'  Insufficient frames in ZIP dataset ({num_frames})')
                return
            
            # Load camera parameters from ZIP
            camera_params = self._load_camera_parameters_zip(num_frames)
            
            # Store sequence metadata
            self.sequence_list.append(seq_name)
            self.sequence_metadata[seq_name] = {
                'num_frames': num_frames,
                'frame_indices': frame_indices,
                'camera_params': camera_params,
                'format': 'zip'
            }
            
            # Generate clips for this sequence
            self._generate_clips_for_sequence(seq_name, num_frames, frame_indices)
            
            print(f'Successfully loaded ZIP dataset with {num_frames} frames')
            
        except ImportError:
            if self.verbose:
                print('zipfile module not available, cannot load ZIP format')
        except Exception as e:
            if self.verbose:
                print(f'Error loading ZIP format: {e}')

    def _load_camera_parameters(self, cam_data_path: str, num_frames: int):
        """Load camera parameters from Spring cam_data directory."""
        
        camera_params = {
            'intrinsics': None,
            'extrinsics': None,
            'focal_distances': None
        }
        
        # Load intrinsics
        intrinsics_path = os.path.join(cam_data_path, 'intrinsics.txt')
        if os.path.exists(intrinsics_path):
            try:
                intrinsics_data = np.loadtxt(intrinsics_path)
                if intrinsics_data.ndim == 1:
                    # Single set of intrinsics for all frames
                    fx, fy, cx, cy = intrinsics_data
                    intrinsics = np.array([[fx, 0.0, cx],
                                         [0.0, fy, cy],
                                         [0.0, 0.0, 1.0]], dtype=np.float32)
                    camera_params['intrinsics'] = np.tile(intrinsics[None], (num_frames, 1, 1))
                else:
                    # Different intrinsics per frame
                    intrinsics_list = []
                    for i in range(min(num_frames, intrinsics_data.shape[0])):
                        fx, fy, cx, cy = intrinsics_data[i]
                        intrinsics = np.array([[fx, 0.0, cx],
                                             [0.0, fy, cy],
                                             [0.0, 0.0, 1.0]], dtype=np.float32)
                        intrinsics_list.append(intrinsics)
                    
                    # Pad with last intrinsics if needed
                    while len(intrinsics_list) < num_frames:
                        intrinsics_list.append(intrinsics_list[-1])
                    
                    camera_params['intrinsics'] = np.stack(intrinsics_list)
                    
            except Exception as e:
                if self.verbose:
                    print(f"Could not load intrinsics: {e}")
        
        # Load extrinsics if available
        extrinsics_path = os.path.join(cam_data_path, 'extrinsics.txt')
        if os.path.exists(extrinsics_path):
            try:
                extrinsics_data = np.loadtxt(extrinsics_path)
                if self.verbose:
                    print(f"Extrinsics data shape: {extrinsics_data.shape}, total elements: {extrinsics_data.size}")
                
                if extrinsics_data.ndim == 1:
                    total_elements = len(extrinsics_data)
                    if total_elements == 16:
                        # Single extrinsics matrix
                        extrinsics = extrinsics_data.reshape(4, 4).astype(np.float32)
                        camera_params['extrinsics'] = np.tile(extrinsics[None], (num_frames, 1, 1))
                    elif total_elements % 16 == 0:
                        # Multiple extrinsics matrices stored as flattened array
                        num_matrices = total_elements // 16
                        extrinsics_list = []
                        
                        for i in range(min(num_frames, num_matrices)):
                            start_idx = i * 16
                            end_idx = start_idx + 16
                            matrix = extrinsics_data[start_idx:end_idx].reshape(4, 4).astype(np.float32)
                            extrinsics_list.append(matrix)
                        
                        # Pad with last extrinsics if needed
                        while len(extrinsics_list) < num_frames:
                            if len(extrinsics_list) > 0:
                                extrinsics_list.append(extrinsics_list[-1])
                            else:
                                extrinsics_list.append(np.eye(4, dtype=np.float32))
                        
                        camera_params['extrinsics'] = np.stack(extrinsics_list)
                    else:
                        if self.verbose:
                            print(f"Warning: extrinsics data size ({total_elements}) is not a multiple of 16")
                        raise ValueError(f"Invalid extrinsics data size: {total_elements}")
                        
                elif extrinsics_data.ndim == 2:
                    if extrinsics_data.shape[1] == 16:
                        # Each row is a flattened 4x4 matrix
                        extrinsics_list = []
                        for i in range(min(num_frames, extrinsics_data.shape[0])):
                            matrix = extrinsics_data[i].reshape(4, 4).astype(np.float32)
                            extrinsics_list.append(matrix)
                        
                        # Pad with last extrinsics if needed
                        while len(extrinsics_list) < num_frames:
                            if len(extrinsics_list) > 0:
                                extrinsics_list.append(extrinsics_list[-1])
                            else:
                                extrinsics_list.append(np.eye(4, dtype=np.float32))
                        
                        camera_params['extrinsics'] = np.stack(extrinsics_list)
                        
                    elif extrinsics_data.shape == (4, 4):
                        # Single 4x4 extrinsics matrix
                        extrinsics = extrinsics_data.astype(np.float32)
                        camera_params['extrinsics'] = np.tile(extrinsics[None], (num_frames, 1, 1))
                    else:
                        if self.verbose:
                            print(f"Warning: unsupported extrinsics shape {extrinsics_data.shape}")
                        # Try to interpret as sequence of values
                        total_elements = extrinsics_data.size
                        if total_elements % 16 == 0:
                            num_matrices = total_elements // 16
                            flat_data = extrinsics_data.flatten()
                            extrinsics_list = []
                            
                            for i in range(min(num_frames, num_matrices)):
                                start_idx = i * 16
                                end_idx = start_idx + 16
                                matrix = flat_data[start_idx:end_idx].reshape(4, 4).astype(np.float32)
                                extrinsics_list.append(matrix)
                            
                            # Pad with last extrinsics if needed
                            while len(extrinsics_list) < num_frames:
                                if len(extrinsics_list) > 0:
                                    extrinsics_list.append(extrinsics_list[-1])
                                else:
                                    extrinsics_list.append(np.eye(4, dtype=np.float32))
                            
                            camera_params['extrinsics'] = np.stack(extrinsics_list)
                        else:
                            raise ValueError(f"Cannot reshape extrinsics data with {total_elements} elements")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Could not load extrinsics: {e}")
                    print("Using default identity extrinsics")
                # Will be set to default below
        
        # Load focal distances if available
        focal_distance_path = os.path.join(cam_data_path, 'focaldistance.txt')
        if os.path.exists(focal_distance_path):
            try:
                focal_distances = np.loadtxt(focal_distance_path)
                if focal_distances.ndim == 0:
                    focal_distances = np.full(num_frames, focal_distances)
                elif len(focal_distances) < num_frames:
                    # Pad with last value
                    last_val = focal_distances[-1] if len(focal_distances) > 0 else 2.0
                    focal_distances = np.concatenate([
                        focal_distances, 
                        np.full(num_frames - len(focal_distances), last_val)
                    ])
                camera_params['focal_distances'] = focal_distances[:num_frames]
            except Exception as e:
                if self.verbose:
                    print(f"Could not load focal distances: {e}")
        
        # Create default parameters if not loaded
        if camera_params['intrinsics'] is None:
            # Default intrinsics assuming 1920x1080 resolution
            fx = fy = 2000.0  # Reasonable default
            cx, cy = 960.0, 540.0
            default_intrinsics = np.array([[fx, 0.0, cx],
                                         [0.0, fy, cy],
                                         [0.0, 0.0, 1.0]], dtype=np.float32)
            camera_params['intrinsics'] = np.tile(default_intrinsics[None], (num_frames, 1, 1))
            
        if camera_params['extrinsics'] is None:
            default_extrinsics = np.eye(4, dtype=np.float32)
            camera_params['extrinsics'] = np.tile(default_extrinsics[None], (num_frames, 1, 1))
        
        return camera_params

    def _load_camera_parameters_zip(self, num_frames: int):
        """Load camera parameters from ZIP file."""
        camera_params = {
            'intrinsics': None,
            'extrinsics': None,
            'focal_distances': None
        }
        
        try:
            import zipfile
            import tempfile
            
            cam_data_zip = os.path.join(self.dataset_location, f'{self.split}_cam_data.zip')
            
            if os.path.exists(cam_data_zip):
                with zipfile.ZipFile(cam_data_zip, 'r') as zf:
                    # Extract to temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        zf.extractall(temp_dir)
                        
                        # Look for intrinsics.txt
                        intrinsics_files = glob.glob(os.path.join(temp_dir, '**/intrinsics.txt'), recursive=True)
                        if intrinsics_files:
                            camera_params = self._load_camera_parameters(os.path.dirname(intrinsics_files[0]), num_frames)
            
        except Exception as e:
            if self.verbose:
                print(f"Could not load camera parameters from ZIP: {e}")
        
        # Create defaults if loading failed
        if camera_params['intrinsics'] is None:
            fx = fy = 2000.0
            cx, cy = 960.0, 540.0
            default_intrinsics = np.array([[fx, 0.0, cx],
                                         [0.0, fy, cy],
                                         [0.0, 0.0, 1.0]], dtype=np.float32)
            camera_params['intrinsics'] = np.tile(default_intrinsics[None], (num_frames, 1, 1))
            
        if camera_params['extrinsics'] is None:
            default_extrinsics = np.eye(4, dtype=np.float32)
            camera_params['extrinsics'] = np.tile(default_extrinsics[None], (num_frames, 1, 1))
        
        return camera_params

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

    def _load_disparity_and_convert_to_depth(self, disp_path: str, target_shape: tuple, frame_idx: int = None) -> Optional[np.ndarray]:
        """Load disparity from HDF5 file or ZIP and convert to depth."""
        
        try:
            if self.use_zip_format:
                return self._load_disparity_from_zip(frame_idx, target_shape)
            else:
                return self._load_disparity_from_file(disp_path, target_shape)
                
        except Exception as e:
            if self.verbose:
                print(f"Could not load disparity: {e}")
            return None

    def _load_disparity_from_file(self, disp_path: str, target_shape: tuple) -> Optional[np.ndarray]:
        """Load disparity from individual HDF5 file."""
        try:
            with h5py.File(disp_path, 'r') as f:
                disparity_hr = f['disparity'][()]  # High-res disparity
                disparity_hr = np.nan_to_num(disparity_hr, nan=0.0, posinf=0.0, neginf=0.0)
                
                h, w = target_shape
                original_h, original_w = disparity_hr.shape
                scale_x = w / original_w
                
                disparity = cv2.resize(disparity_hr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                disparity = disparity * scale_x  # Scale disparity values for resolution change
                
                # Convert disparity to depth using baseline
                depth = np.zeros_like(disparity)
                valid_mask = disparity > 0
                depth[valid_mask] = self.baseline / disparity[valid_mask]
                
                return depth
                
        except Exception as e:
            if self.verbose:
                print(f"Could not load disparity from {disp_path}: {e}")
            return None

    def _load_disparity_from_zip(self, frame_idx: int, target_shape: tuple) -> Optional[np.ndarray]:
        """Load disparity from ZIP file."""
        try:
            import zipfile
            import tempfile
            
            disp_zip = os.path.join(self.dataset_location, f'{self.split}_disp1_{self.eyes}.zip')
            
            if not os.path.exists(disp_zip):
                return None
                
            with zipfile.ZipFile(disp_zip, 'r') as zf:
                # Look for the specific frame file
                possible_names = [
                    f'disp1_{self.eyes}_{frame_idx:04d}.dsp5',
                    f'{frame_idx:04d}.dsp5'
                ]
                
                disp_file = None
                for name in possible_names:
                    for file_in_zip in zf.namelist():
                        if file_in_zip.endswith(name):
                            disp_file = file_in_zip
                            break
                    if disp_file:
                        break
                
                if disp_file is None:
                    if self.verbose:
                        print(f"Could not find disparity file for frame {frame_idx}")
                    return None
                
                # Extract and read the file
                with tempfile.NamedTemporaryFile(suffix='.dsp5') as temp_file:
                    temp_file.write(zf.read(disp_file))
                    temp_file.flush()
                    
                    return self._load_disparity_from_file(temp_file.name, target_shape)
                    
        except Exception as e:
            if self.verbose:
                print(f"Could not load disparity from ZIP for frame {frame_idx}: {e}")
            return None

    def _load_image_from_zip(self, frame_idx: int) -> Optional[np.ndarray]:
        """Load image from ZIP file."""
        try:
            import zipfile
            
            frame_zip = os.path.join(self.dataset_location, f'{self.split}_frame_{self.eyes}.zip')
            
            if not os.path.exists(frame_zip):
                return None
                
            with zipfile.ZipFile(frame_zip, 'r') as zf:
                # Look for the specific frame file
                possible_names = [
                    f'frame_{self.eyes}_{frame_idx:04d}.png',
                    f'{frame_idx:04d}.png'
                ]
                
                frame_file = None
                for name in possible_names:
                    for file_in_zip in zf.namelist():
                        if file_in_zip.endswith(name):
                            frame_file = file_in_zip
                            break
                    if frame_file:
                        break
                
                if frame_file is None:
                    if self.verbose:
                        print(f"Could not find frame file for frame {frame_idx}")
                    return None
                
                # Extract and read the image
                image_data = zf.read(frame_file)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                return image
                
        except Exception as e:
            if self.verbose:
                print(f"Could not load image from ZIP for frame {frame_idx}: {e}")
            return None

    def _load_semantic_mask(self, seq_path: str, frame_idx: int, target_shape: tuple) -> Optional[np.ndarray]:
        """Load semantic mask from Spring maps."""
        
        h, w = target_shape
        
        # Try different map types in order of preference
        map_types = [
            (f'rigidmap_FW_{self.eyes}', f'rigidmap_FW_{self.eyes}'),
            (f'detailmap_disp1_{self.eyes}', f'detailmap_disp1_{self.eyes}'),
            (f'skymap_{self.eyes}', f'skymap_{self.eyes}')
        ]
        
        for map_dir, map_prefix in map_types:
            map_path = os.path.join(
                seq_path, 'maps', map_dir, f'{map_prefix}_{frame_idx:04d}.png'
            )
            
            if os.path.exists(map_path):
                try:
                    mask_img = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
                    if mask_img is not None:
                        # Resize to target shape
                        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        if len(mask_img.shape) == 3:
                            raw_mask = mask_img[:, :, 0]
                        else:
                            raw_mask = mask_img
                        
                        # Convert to semantic IDs
                        unique_vals = np.unique(raw_mask)
                        semantic_mask = np.zeros_like(raw_mask, dtype=np.uint8)
                        for i, val in enumerate(unique_vals):
                            semantic_mask[raw_mask == val] = i
                        
                        return semantic_mask
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Could not load mask from {map_path}: {e}")
                    continue
        
        # Fallback: create grid segmentation
        if self.verbose:
            print("No Spring semantic maps found, using grid segmentation")
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(6):
            for j in range(8):
                y_start, y_end = i * h // 6, (i + 1) * h // 6
                x_start, x_end = j * w // 8, (j + 1) * w // 8
                semantic_mask[y_start:y_end, x_start:x_end] = i * 8 + j + 1
        
        return semantic_mask

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
        
        # Create synthetic tracks using optical flow if available
        if stride == 1 and FLAG and self.ENABLE_TRACK:
            tracks, track_masks, source_H, source_W = self._extract_flow_tracks(
                seq_meta, frame_indices)
            tracks = tracks.transpose(1, 0, 2)
            track_masks = track_masks.transpose(1, 0)
        else:
            tracks = np.zeros((len(frame_indices), self.track_num, 2), dtype=np.float32)
            track_masks = np.zeros((len(frame_indices), self.track_num), dtype=bool)

        processed_tracks = []; processed_track_masks = []
        for i, frame_idx in enumerate(frame_indices):
            # Load image
            image = None
            if self.use_zip_format:
                image = self._load_image_from_zip(frame_idx)
            else:
                rgb_path = os.path.join(
                    seq_meta['seq_path'], f'frame_{self.eyes}', f'frame_{self.eyes}_{frame_idx:04d}.png')
                image = read_image_cv2(rgb_path)
            if image is None:
                if self.use_zip_format:
                    raise ValueError(f"Could not load image for frame {frame_idx} from ZIP")
                else:
                    raise ValueError(f"Could not load image: {rgb_path}")
            # Get camera parameters
            camera_params = seq_meta['camera_params']
            # Find the camera parameter index for this frame
            if self.use_zip_format:
                # For ZIP format, frame_idx is the actual index
                frame_idx_in_seq = min(frame_idx, camera_params['intrinsics'].shape[0] - 1)
            else:
                # For sample format, find index in sequence
                frame_idx_in_seq = seq_meta['frame_indices'].index(frame_idx) if frame_idx in seq_meta['frame_indices'] else 0
                frame_idx_in_seq = min(frame_idx_in_seq, camera_params['intrinsics'].shape[0] - 1)
            # Use Spring's baseline formula: baseline = baseline_width * fx
            fx = seq_meta['camera_params']['intrinsics'][frame_idx_in_seq][0,0]
            self.baseline = self.baseline_ori * fx
            # Load depth from disparity
            depth_map = None
            if self.load_depth:
                if self.use_zip_format:
                    depth_map = self._load_disparity_and_convert_to_depth(None, target_image_shape, frame_idx)
                else:
                    disp_path = os.path.join(seq_meta['seq_path'], f'disp1_{self.eyes}', f'disp1_{self.eyes}_{frame_idx:04d}.dsp5')
                    depth_map = self._load_disparity_and_convert_to_depth(disp_path, target_image_shape)
                if depth_map is not None:
                    depth_map = threshold_depth_map(depth_map, max_percentile=98)

            depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            intri_opencv = camera_params['intrinsics'][frame_idx_in_seq]
            if camera_params['extrinsics'] is not None:
                extri_opencv = camera_params['extrinsics'][frame_idx_in_seq]
            else:
                extri_opencv = np.eye(4, dtype=np.float32)
            
            # Spring data is already in OpenCV format (verified by co3d coordinate check)
            if self.apply_coord_transform:
                extri_opencv = SPRING_TO_OPENCV_T @ extri_opencv
                if self.verbose and i == 0:
                    print(f"[Spring] Applied coordinate transformation")
            else:
                if self.verbose and i == 0:
                    print(f"[Spring] Using original coordinates")

            original_size = np.array(image.shape[:2])
            original_track = tracks[i] if tracks is not None else None
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
                filepath=f"frame_{frame_idx:04d}",
            )
            # check = check_coord_system(world_coords_points.reshape(-1, 3)[point_mask.reshape(-1)], processed_extri)
            # print(check, '===========check===========')
            # matrix_check(processed_image, processed_depth, processed_intri, processed_extri)
            # print(np.max(processed_depth), '===========np.max(processed_depth)===========', np.min(processed_depth), '===========np.min(processed_depth)===========', np.mean(processed_depth), '===========np.mean(processed_depth)===========')
            
            images.append(processed_image)
            depths.append(processed_depth)
            extrinsics.append(processed_extri)
            intrinsics.append(processed_intri)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            processed_tracks.append(processed_track)
            processed_track_masks.append(processed_track_mask&track_masks[i])
        return {
            "seq_name": f"spring_{seq_name}",
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

    def _extract_flow_tracks(self, seq_meta: Dict, frame_indices: List[int]):
        """
        Extract tracks using optical flow from Spring dataset (.flo5):
        - 起点：规则网格（基于第一张可用光流的 (H,W)）
        - 推进：前向光流 + 最近邻采样（更稳更快）
        - 无效：tracks 写 0，mask=False（不沿用上一帧位置）
        - 保证无 NaN/Inf
        """
        seq_path = seq_meta['seq_path']
        flow_fw_path = os.path.join(seq_path, f'flow_FW_{self.eyes}')
        N = int(self.track_num)
        T = len(frame_indices)
        # ---- 找到第一张可用的 flow 以确定 H, W ----
        t_prev = frame_indices[0]
        flow_path = os.path.join(flow_fw_path, f'flow_FW_{self.eyes}_{t_prev:04d}.flo5')
        first_flow = self._load_flow_from_file(flow_path, None)  # 先不强制尺寸
        H, W = first_flow.shape[:2]
        # ---- 起点：规则网格 ----
        start_xy = self._init_points_grid(W, H, N).astype(np.float32)  # (N,2)
        tracks = np.zeros((N, T, 2), dtype=np.float32)
        masks  = np.zeros((N, T), dtype=bool)
        tracks[:, 0, :] = start_xy
        masks[:, 0] = True
        # ---- 时间推进（最近邻采样，与 _extract_synthetic_tracks 一致的“无效写0”策略）----
        for ti in range(1, T):
            t_prev = frame_indices[ti - 1]
            flow_path = os.path.join(flow_fw_path, f'flow_FW_{self.eyes}_{t_prev:04d}.flo5')
            flow = self._load_flow_from_file(flow_path, None)
            prev_xy    = tracks[:, ti - 1, :].copy()
            prev_valid = masks[:, ti - 1].copy()
            # 默认本帧无效、位移为 0
            tracks[:, ti, :] = 0.0
            masks[:, ti] = False
            if (flow is not None) and (flow.ndim == 3) and (flow.shape[2] >= 2):
                # 如果当前 flow 尺寸与参考 (H,W) 不一致，resize 并按比例缩放位移
                flow_use = flow
                disp, samp_valid = self._sample_flow_nn(flow_use, prev_xy)  # (N,2), (N,)
                good = prev_valid & samp_valid
                if np.any(good):
                    cur_xy = prev_xy[good] + disp[good]
                    inb = (cur_xy[:, 0] >= 0) & (cur_xy[:, 0] < W) & \
                        (cur_xy[:, 1] >= 0) & (cur_xy[:, 1] < H)
                    idx = np.where(good)[0][inb]
                    if idx.size > 0:
                        tracks[idx, ti, :] = cur_xy[inb]
                        masks[idx, ti] = True
            # else: 无光流，保持默认（全 0、全 False）
        # ---- 清理 NaN/Inf ----
        bad = ~np.isfinite(tracks).all(axis=2)
        if np.any(bad):
            tracks[bad] = 0.0
            masks[bad] = False
        return tracks, masks, H, W

    def _load_flow_from_file(self, flow_path: str, target_shape: tuple):
        with h5py.File(flow_path, "r") as f:
            if "flow" not in f.keys():
                raise IOError(f"File {flow_path} does not have a 'flow' key. Is this a valid flo5 file?")
            return f["flow"][()][::2,::2]

    def _visualize_spring_pointcloud(self, image, depth_map, extri_opencv, intri_opencv, frame_idx, seq_name):
        try:
            if depth_map is None or np.sum(depth_map > 0) < 100:
                print(f"[Spring Viz] Insufficient depth data for frame {frame_idx}")
                return
            height, width = depth_map.shape
            fx, fy = intri_opencv[0, 0], intri_opencv[1, 1]
            cx, cy = intri_opencv[0, 2], intri_opencv[1, 2]
            # Get valid depth pixels
            valid_mask = (depth_map > 0) & np.isfinite(depth_map)
            y_coords, x_coords = np.where(valid_mask)
            z = depth_map[valid_mask]
            color_point = image[y_coords, x_coords]
            X_cam = (x_coords - cx) * z / fx
            Y_cam = (y_coords - cy) * z / fy
            Z_cam = z.copy()
            points_camera = np.stack([X_cam, Y_cam, Z_cam], axis=1)
            # Transform to world coordinates
            camera_to_world = np.linalg.inv(extri_opencv)
            ones = np.ones((points_camera.shape[0], 1), dtype=points_camera.dtype)
            points_cam_h = np.hstack([points_camera, ones])
            points_world_h = (camera_to_world @ points_cam_h.T).T
            points_world = points_world_h[:, :3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_camera.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(color_point.astype(np.float64)/255.)
            o3d.io.write_point_cloud(f"spring_camera_{seq_name}_frame_{frame_idx:04d}.ply", pcd, write_ascii=True)
            print(f"[Spring] Saved camera-space point cloud to: spring_camera_{seq_name}_frame_{frame_idx:04d}.ply")
            # Verification
            u_proj = fx * (points_camera[:, 0] / points_camera[:, 2]) + cx
            v_proj = fy * (points_camera[:, 1] / points_camera[:, 2]) + cy
            u_orig, v_orig = x_coords.astype(np.float32), y_coords.astype(np.float32) 
            reproj_err = np.sqrt((u_proj - u_orig)**2 + (v_proj - v_orig)**2)
            print(f"[verify] pixel reprojection error  mean={reproj_err.mean():.3f}px  "
                  f"median={np.median(reproj_err):.3f}px  max={reproj_err.max():.3f}px  N={reproj_err.size}")
            X_bp = (x_coords - cx) * z / fx
            Y_bp = (y_coords - cy) * z / fy
            Z_bp = z.copy()
            Pc_bp = np.stack([X_bp, Y_bp, Z_bp], axis=1)
            Pc_gt = points_camera
            cam_err = np.linalg.norm(Pc_bp - Pc_gt, axis=1)
            print(f"[verify] camera-frame round-trip error  mean={cam_err.mean():.6e} m  "
                  f"median={np.median(cam_err):.6e} m  max={cam_err.max():.6e} m")
            ones = np.ones((Pc_bp.shape[0], 1), dtype=Pc_bp.dtype)
            Pc_bp_h = np.hstack([Pc_bp, ones])
            Pw_bp_h = (camera_to_world @ Pc_bp_h.T).T
            Pw_bp   = Pw_bp_h[:, :3]
            Pw_gt   = points_world
            world_err = np.linalg.norm(Pw_bp - Pw_gt, axis=1)
            print(f"[verify] world-frame round-trip error  mean={world_err.mean():.6e} m  "
                  f"median={np.median(world_err):.6e} m  max={world_err.max():.6e} m")
            color_bp = image[y_coords, x_coords]
            pcd_bp = o3d.geometry.PointCloud()
            pcd_bp.points = o3d.utility.Vector3dVector(Pw_bp.astype(np.float64))
            pcd_bp.colors = o3d.utility.Vector3dVector((color_bp.astype(np.float64) / 255.0))
            world_ply_path = f"spring_world_{seq_name}_frame_{frame_idx:04d}.ply"
            o3d.io.write_point_cloud(world_ply_path, pcd_bp, write_ascii=True)
            print(f"[verify] Saved back-projected world points to {world_ply_path}")
            # Print coordinate ranges for verification
            print(f"[Spring Stats] Point cloud statistics:")
            print(f"  - Total points: {len(points_world)}")
            print(f"  - Camera coords range: X[{points_camera[:, 0].min():.2f}, {points_camera[:, 0].max():.2f}]")
            print(f"                         Y[{points_camera[:, 1].min():.2f}, {points_camera[:, 1].max():.2f}]")
            print(f"                         Z[{points_camera[:, 2].min():.2f}, {points_camera[:, 2].max():.2f}]")
            print(f"  - World coords range:  X[{points_world[:, 0].min():.2f}, {points_world[:, 0].max():.2f}]")
            print(f"                         Y[{points_world[:, 1].min():.2f}, {points_world[:, 1].max():.2f}]")
            print(f"                         Z[{points_world[:, 2].min():.2f}, {points_world[:, 2].max():.2f}]")
            print(f"  - Depth range: [{z.min():.2f}, {z.max():.2f}] meters")
        except Exception as e:
            print(f"[Spring Viz] Error in visualization: {e}")
            import traceback
            traceback.print_exc()


def depth_to_rgb(
    depth: np.ndarray,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    invalid_value: float | None = 0.0,
    bg_color: tuple[int, int, int] | None = (0, 0, 0),
    percentiles: tuple[float, float] | None = (2.0, 98.0),
) -> np.ndarray:
    """
    Convert a (H, W) depth map to an RGB uint8 image using a Matplotlib colormap.

    Args:
        depth: 2D array of shape (H, W).
        cmap: Matplotlib colormap name (e.g., "magma", "viridis", "turbo").
        vmin, vmax: Value range for color mapping. If None, computed from valid pixels.
        invalid_value: Depth value to treat as invalid (e.g., 0). Set to None to ignore.
        bg_color: RGB tuple (0..255) for invalid pixels. Set to None to leave as-is.
        percentiles: If vmin/vmax not given, compute them from these percentiles of valid data.

    Returns:
        rgb: (H, W, 3) uint8 array.
    """
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array")
    d = depth.astype(np.float32, copy=False)
    # Define validity: finite and not equal to invalid_value (if provided)
    valid = np.isfinite(d)
    if invalid_value is not None:
        valid &= d != np.float32(invalid_value)
    if not np.any(valid):
        raise ValueError("No valid depth values found.")
    # Determine vmin/vmax
    vals = d[valid]
    if vmin is None or vmax is None:
        if percentiles is not None:
            lo, hi = np.percentile(vals, percentiles)
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
        else:
            vmin = vals.min() if vmin is None else vmin
            vmax = vals.max() if vmax is None else vmax
    if vmin == vmax:
        vmax = vmin + 1e-6  # avoid divide-by-zero in normalization
    # Map to colors
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.get_cmap(cmap)
    rgba = mapper(norm(d))  # (H, W, 4) floats in [0,1]
    # Convert to uint8 RGB
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    # Fill invalids with background color if requested
    if bg_color is not None:
        rgb[~valid] = np.array(bg_color, dtype=np.uint8)
    return rgb

def save_depth_as_png(
    image: np.ndarray,
    depth: np.ndarray,
    out_path: str = "image.png",
    **depth_to_rgb_kwargs,
) -> str:
    """
    Convert a (H, W) depth map to RGB and save as a PNG.
    Pass any keyword args through to depth_to_rgb (e.g., cmap, vmin, vmax,
    invalid_value, bg_color, percentiles).
    """
    rgb = depth_to_rgb(depth, **depth_to_rgb_kwargs) + image  # (H,W,3) uint8   
    Image.fromarray(rgb).save(out_path)
    return out_path


if __name__ == "__main__":
    # Simple test
    print("Testing Spring dataset...")
    
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
    dataset = SpringDataset(
        common_conf=config,
        dataset_location='/Users/gracechen/Desktop/4D_FM/spring',  # test with your local path
        sequence_names=['0001'],
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