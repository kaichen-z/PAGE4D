import sys
import os
import glob
import random
import numpy as np
import cv2
import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import open3d as o3d
from tqdm import tqdm
import json
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from PIL import Image
import pdb
# CRITICAL: Configure TensorFlow to not use GPU memory before importing
import tensorflow as tf
from ..dataset_util import depth_to_world_coords_points
# Force TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

# Add paths for imports
sys.path.append('../../..')

# Local imports
from ..base_dataset import BaseDataset
from ..dataset_util import get_stride_distribution, read_image_cv2, threshold_depth_map

# Waymo Open Dataset imports
try:
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import frame_utils
    from waymo_open_dataset.utils import transform_utils
    from waymo_open_dataset.utils import range_image_utils
    from waymo_open_dataset.utils import camera_segmentation_utils
    from waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2
except ImportError:
    print("Warning: waymo-open-dataset not installed. Install with: pip install waymo-open-dataset-tf-2-11-0")

# Camera name mapping
CAMERA_NAME_TO_IDX = {
    open_dataset.CameraName.FRONT: 0,
    open_dataset.CameraName.FRONT_LEFT: 1,
    open_dataset.CameraName.FRONT_RIGHT: 2,
    open_dataset.CameraName.SIDE_LEFT: 3,
    open_dataset.CameraName.SIDE_RIGHT: 4,
}

IDX_TO_CAMERA_NAME = {v: k for k, v in CAMERA_NAME_TO_IDX.items()}

T = np.array([[0, -1, 0, 0],
              [0, 0, -1, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 1]])

def apply_conjugate_on_points(points_3d, E, T):
    """
    points_3d: (N,3) 世界坐标
    extrinsics: (4,4) camera-from-world (X_c = E @ X_w)
    T: (4,4)   你左乘在 extrinsics 前面的那个变换（E' = T @ E）
    返回:
      points_3d_new: (N,3)，使得
      proj(points_3d_new, extrinsics) == proj(points_3d, T @ extrinsics)
    """
    N = points_3d.shape[0]
    ones = np.ones((N, 1), dtype=points_3d.dtype)
    pts_h = np.hstack([points_3d, ones])              # (N,4) 行向量齐次
    E_inv = np.linalg.inv(E)
    M = E_inv @ T @ E                                  # 共轭: E^{-1} T E
    pts_h_new = (M @ pts_h.T).T                        # (N,4)
    # 规一化（以防 T 或 E 有投影成分导致 w!=1）
    w = pts_h_new[:, 3:4]
    w = np.where(np.abs(w) < 1e-8, 1.0, w)
    pts_h_new = pts_h_new / w
    points_3d_new = pts_h_new[:, :3]
    return points_3d_new

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

def transform_points(points_3d, extrinsics, T):
    # points_3d: (N,3) 世界坐标
    # extrinsics: (4,4) camera-from-world (X_c = E @ X_w)
    # T: (4,4)   你左乘在 extrinsics 前面的那个变换（E' = T @ E）
    N = points_3d.shape[0]
    ones = np.ones((N, 1), dtype=points_3d.dtype)
    pts_h = np.hstack([points_3d, ones])                # (N,4)
    M = np.linalg.inv(extrinsics) @ T @ extrinsics      # 共轭：把相机系左乘T吸收到世界点
    pts_h_new = (M @ pts_h.T).T                         # (N,4)
    new_points_3d = pts_h_new[:, :3] / pts_h_new[:, 3:] # 还原到3D
    return new_points_3d

class WaymoDataset(BaseDataset):
    """
    Waymo Open Dataset loader for VGGT training pipeline.
    
    This dataset loads Waymo sequences from tfrecord files which contain:
    - Camera images from 5 synchronized views
    - Camera calibration (intrinsics/extrinsics)
    - 2D/3D object labels with tracking IDs
    - LiDAR data (optional, for depth)
    - Vehicle pose data
    
    OPTIMIZED VERSION: Forces TensorFlow to use CPU only to avoid GPU memory conflicts.
    """
    
    def __init__(
        self,
        common_conf,
        dataset_location: str = '/workspace/data/kaichen/data/waymo/archived_files/training',
        split: str = 'train',
        sequence_names: Optional[List[str]] = None,  
        camera_names: List[str] = ['FRONT'],  # Which cameras to use
        strides: List[int] = [1, 2, 3, 4],
        clip_step: int = 2,
        min_num_images: int = 8,
        len_train: int = 10000,
        len_test: int = 1000,
        quick: bool = False,
        verbose: bool = False,
        dist_type: Optional[str] = None,
        use_lidar_depth: bool = True,  # Whether to project LiDAR to camera for depth
        cache_frames_in_memory: bool = False,  # NEW: Control frame caching
        **kwargs
    ):
        """
        Initialize Waymo dataset.
        
        Args:
            common_conf: Common configuration object
            dataset_location: Path to Waymo dataset root (contains training_XXXX folders)
            split: Dataset split ('train' or 'val')
            sequence_names: List of tfrecord filenames to load. If None, auto-discovers all.
            camera_names: List of camera names to use (FRONT, FRONT_LEFT, etc.)
            strides: List of temporal strides to use
            clip_step: Step size for sampling clips
            min_num_images: Minimum number of images per sequence
            len_train: Training dataset length
            len_test: Test dataset length  
            quick: Quick mode for testing (uses subset)
            verbose: Verbose logging
            dist_type: Distribution type for stride sampling
            use_lidar_depth: Whether to use LiDAR projection for depth maps
            cache_frames_in_memory: Whether to cache parsed frames in memory (uses more RAM but faster)
        """
        print(f'Loading Waymo dataset from {dataset_location}...')
        print(f'TensorFlow GPU devices: {tf.config.list_physical_devices("GPU")}')  # Should be empty
        print(f'TensorFlow will use CPU only to avoid GPU memory conflicts')
        
        super().__init__(common_conf=common_conf)
        
        # Dataset configuration
        self.dataset_location = dataset_location
        self.split = split
        self.camera_names = camera_names
        self.strides = strides
        self.clip_step = clip_step
        self.min_num_images = min_num_images
        self.quick = quick
        self.verbose = verbose
        self.dist_type = dist_type
        self.use_lidar_depth = use_lidar_depth
        self.cache_frames_in_memory = False
        
        # Auto-discover sequences if not specified
        if sequence_names is None:
            self.sequence_names = self._discover_sequences()
            if self.verbose:
                print(f'Auto-discovered {len(self.sequence_names)} tfrecord files')
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
        Auto-discover all Waymo tfrecord files.
        
        Returns:
            List of tfrecord file paths
        """
        tfrecord_files = []
        
        # Search in training_XXXX subdirectories
        for subdir in glob.glob(os.path.join(self.dataset_location, 'training_*')):
            if os.path.isdir(subdir):
                # Find all tfrecord files in this subdirectory
                pattern = os.path.join(subdir, '*_with_camera_labels.tfrecord')
                tfrecords = glob.glob(pattern)
                tfrecord_files.extend(tfrecords)
        
        # If no subdirectories, search in root
        if not tfrecord_files:
            pattern = os.path.join(self.dataset_location, '*_with_camera_labels.tfrecord')
            tfrecord_files = glob.glob(pattern)
        
        if self.verbose:
            print(f'Found {len(tfrecord_files)} tfrecord files')
            
        # Quick mode - use subset
        if self.quick and len(tfrecord_files) > 5:
            tfrecord_files = tfrecord_files[:5]
            
        return sorted(tfrecord_files)

    def _load_sequences2(self):
        """Load and organize Waymo sequences from tfrecords."""
        # Initialize storage
        self.sequence_list = []
        self.sequence_metadata = {}
        self.clip_data = []
        self.frame_cache = {} if self.cache_frames_in_memory else None
        # Process each tfrecord file
        for tfrecord_path in tqdm(self.sequence_names[::250]):
            if not os.path.exists(tfrecord_path):
                if self.verbose:
                    print(f'TFRecord does not exist: {tfrecord_path}')
                continue
            # Extract sequence name from path
            seq_name = os.path.basename(tfrecord_path).replace('_with_camera_labels.tfrecord', '')
            if self.verbose:
                print(f'Processing sequence: {seq_name}')
            # Load tfrecord and extract metadata
            try:
                # Count frames and extract first frame for metadata
                dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
                frame_count = 0
                first_frame = None
                for data in dataset:
                    if first_frame is None:
                        frame = open_dataset.Frame()
                        frame.ParseFromString(bytearray(data.numpy()))
                        first_frame = frame
                        # Optionally cache the first frame
                        if self.cache_frames_in_memory:
                            if seq_name not in self.frame_cache:
                                self.frame_cache[seq_name] = {}
                            self.frame_cache[seq_name][0] = frame
                    frame_count += 1
                if frame_count < self.min_num_images:
                    if self.verbose:
                        print(f'  Skipping {seq_name}: insufficient frames ({frame_count})')
                    continue
                # Store sequence metadata
                self.sequence_list.append(seq_name)
                self.sequence_metadata[seq_name] = {
                    'tfrecord_path': tfrecord_path,
                    'num_frames': frame_count,
                    'frame_indices': list(range(frame_count)),
                    'first_frame': first_frame if not self.cache_frames_in_memory else None,
                }
                # Generate clips for this sequence
                self._generate_clips_for_sequence(seq_name, frame_count, list(range(frame_count)))
            except Exception as e:
                if self.verbose:
                    print(f'  Error loading {seq_name}: {e}')
                continue
        print(f'Successfully loaded {len(self.sequence_list)} sequences')

    def _load_sequences(self):
        """Load and organize Waymo sequences from prebuilt manifest (fast)."""
        self.sequence_list = []
        self.sequence_metadata = {}
        self.clip_data = []
        self.frame_cache = {} if self.cache_frames_in_memory else None
        manifest_path = os.path.join(self.dataset_location, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run the preprocessor to build it once.")
        with open(manifest_path, 'r') as f:
            man = json.load(f)
        # 你原来有 self.sequence_names（路径列表）；如果你想按它筛选，做一次交集
        wanted = set([os.path.basename(p).replace('_with_camera_labels.tfrecord','')
                    for p in self.sequence_names]) if self.sequence_names else set(man.keys())
        for seq_name in sorted(man.keys()):
            if seq_name not in wanted:
                continue
            meta = man[seq_name]
            tfrecord_path = meta["tfrecord_path"]
            frame_count   = meta["num_frames"]
            if not os.path.exists(tfrecord_path):
                if self.verbose:
                    print(f"TFRecord missing, skip: {tfrecord_path}")
                continue
            if frame_count < self.min_num_images:
                if self.verbose:
                    print(f"Skipping {seq_name}: insufficient frames ({frame_count})")
                continue
            self.sequence_list.append(seq_name)
            self.sequence_metadata[seq_name] = {
                'tfrecord_path': tfrecord_path,
                'num_frames': frame_count,
                'frame_indices': list(range(frame_count)),
                'first_frame': None,}
            self._generate_clips_for_sequence(seq_name, frame_count, list(range(frame_count)))
        print(f"Successfully loaded {len(self.sequence_list)} sequences (fast manifest)")

    def _generate_clips_for_sequence(self, seq_name: str, num_frames: int, frame_indices: List[int]):
        """Generate training clips for a sequence with different strides."""
        for stride in self.strides:
            max_start_idx = num_frames - 2 * stride
            for start_idx in range(0, max_start_idx + 1, self.clip_step):
                # Generate frame indices for this clip (2 frames minimum)
                clip_frame_indices = [
                    frame_indices[start_idx],
                    frame_indices[min(start_idx + stride, num_frames - 1)]]
                # Store clip information
                self.clip_data.append({
                    'seq_name': seq_name,
                    'frame_indices': clip_frame_indices,
                    'start_idx': start_idx,
                    'stride': stride,
                    'available_frames': frame_indices,
                    'camera_name': self.camera_names[0], })

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

    def _get_frame_from_cache_or_load(self, seq_name: str, frame_idx: int, tfrecord_path: str) -> open_dataset.Frame:
        """Get frame from cache or load from tfrecord."""
        # Check cache first if enabled
        if self.cache_frames_in_memory:
            if seq_name in self.frame_cache and frame_idx in self.frame_cache[seq_name]:
                return self.frame_cache[seq_name][frame_idx]
        # Load from tfrecord
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
        for i, data in enumerate(dataset):
            if i == frame_idx:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                # Cache if enabled
                if self.cache_frames_in_memory:
                    if seq_name not in self.frame_cache:
                        self.frame_cache[seq_name] = {}
                    self.frame_cache[seq_name][frame_idx] = frame
                return frame
        raise ValueError(f"Frame {frame_idx} not found in {tfrecord_path}")

    def _extract_image(self, frame: open_dataset.Frame, camera_name: int) -> np.ndarray:
        """
        Extract image from Waymo frame for specified camera.
        Args:
            frame: Waymo Frame object
            camera_name: Camera name enum value
        Returns:
            image: RGB image as numpy array
        """
        for image in frame.images:
            if image.name == camera_name:
                img_bytes = tf.image.decode_jpeg(image.image).numpy()
                if len(img_bytes.shape) == 2:
                    img_bytes = cv2.cvtColor(img_bytes, cv2.COLOR_GRAY2RGB)
                elif img_bytes.shape[2] == 4:
                    img_bytes = cv2.cvtColor(img_bytes, cv2.COLOR_BGRA2RGB)
                return img_bytes
        raise ValueError(f"Image not found for camera {camera_name}")

    def _extract_camera_calibration(
        self,
        frame: open_dataset.Frame,
        camera_name: int,
        target_shape: tuple[int, int] | None = None,  # (H, W) of the image you actually use
        crop_xy: tuple[float, float] = (0.0, 0.0),    # (dx, dy) crop offset after resize, if any
    ):
        """
        Returns:
            intri_opencv: (4,4) K in the SAME resolution domain as your final image
            extri_opencv: (4,4) vehicle(world)->camera matrix  (world-to-camera)
        """
        # 1) Find calibration for this camera
        calib = None
        for c in frame.context.camera_calibrations:
            if c.name == camera_name:
                calib = c
                break
        assert calib is not None, f"Calibration for camera {camera_name} not found"
        # Native image size from Waymo calibration (height, width)
        native_H = int(calib.height)
        native_W = int(calib.width)
        # 2) Intrinsics from Waymo (OpenCV style)
        fx, fy, cx, cy = calib.intrinsic[0], calib.intrinsic[1], calib.intrinsic[2], calib.intrinsic[3]
        K = np.array([[fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
        # 3) Resize correction
        if target_shape is not None:
            H_tgt, W_tgt = int(target_shape[0]), int(target_shape[1])
            sx = W_tgt / float(native_W)
            sy = H_tgt / float(native_H)
            S = np.array([[sx, 0.0, 0.0],
                        [0.0, sy, 0.0],
                        [0.0, 0.0, 1.0]], dtype=np.float64)
            K = S @ K
            dx, dy = crop_xy
            K[0, 2] -= dx
            K[1, 2] -= dy
        # 4) 转成 (4,4)
        K_h = np.eye(4, dtype=np.float64)
        K_h[:3, :3] = K
        intri_opencv = K_h
        # 5) Extrinsics
        T_cam_to_veh = np.array(calib.extrinsic.transform, dtype=np.float64).reshape(4, 4)
        T_veh_to_cam = np.linalg.inv(T_cam_to_veh)   # vehicle->camera
        extri_opencv = T_veh_to_cam

        # 6) 畸变参数（Waymo顺序：[k1,k2,p1,p2,k3,k4,k5,k6]，有多少给多少）
        dist = np.array([calib.intrinsic[4], calib.intrinsic[5], calib.intrinsic[6], calib.intrinsic[7]])
        return intri_opencv, extri_opencv, dist
    
    def points2img2(self, points, extrinsics, intrinsics):
        T = intrinsics @ extrinsics
        proj = (T[:3, :3] @ points.T + T[:3, [3]]).T
        proj[:, :2] /= proj[:, [2]]
        return proj
    
    def points2img(self,points, extrinsics, intrinsics, distCoeffs=None):
        """
        Project 3D points into image plane with optional 4-coefficient distortion.
        Args:
            points:     (N,3) points in vehicle/world frame
            extrinsics: (4,4) vehicle/world -> camera  (R|t) in the camera frame
            intrinsics: (3,3) or (4,4) K (must match your image resolution domain)
            distCoeffs: None or iterable of 4 floats [k1, k2, p1, p2]
        Returns:
            proj: (N,3) -> [u, v, Zc]  (pixel x, pixel y, camera-depth Zc)
        """
        pts = np.asarray(points, dtype=np.float64)
        T   = np.asarray(extrinsics, dtype=np.float64)
        Kin = np.asarray(intrinsics, dtype=np.float64)

        # --- Normalize K to 3x3 (allow caller to pass 4x4 homogeneous K_h) ---
        if Kin.shape == (4, 4):
            K = Kin[:3, :3].copy()
        elif Kin.shape == (3, 3):
            K = Kin.copy()
        else:
            raise ValueError(f"intrinsics must be 3x3 or 4x4, got {Kin.shape}")

        if T.shape != (4, 4):
            raise ValueError(f"extrinsics must be 4x4 world/veh->cam, got {T.shape}")

        # --- World/veh -> camera ---
        R = T[:3, :3]
        t = T[:3, 3]
        P_cam = (R @ pts.T + t.reshape(3, 1)).T   # (N,3)
        X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]

        # Avoid divide-by-zero; keep true Z for output
        eps = 1e-12
        Zsafe = np.where(Z > eps, Z, eps)
        xn = X / Zsafe
        yn = Y / Zsafe

        # --- Optional radial+tangential distortion (OpenCV k1,k2,p1,p2) ---
        if distCoeffs is not None:
            dc = np.asarray(distCoeffs, dtype=np.float64).ravel()
            if dc.size != 4:
                raise ValueError(f"distCoeffs must be length 4 [k1,k2,p1,p2], got {dc}")
            k1, k2, p1, p2 = dc
            r2 = xn*xn + yn*yn
            radial = 1.0 + k1*r2 + k2*(r2*r2)
            x_tan = 2.0*p1*xn*yn + p2*(r2 + 2.0*xn*xn)
            y_tan = p1*(r2 + 2.0*yn*yn) + 2.0*p2*xn*yn
            xd = xn*radial + x_tan
            yd = yn*radial + y_tan
        else:
            xd, yd = xn, yn

        # --- Pixels ---
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = fx*xd + cx
        v = fy*yd + cy

        proj = np.stack([u, v, Z], axis=1)
        return proj

    def _extract_lidar_depth(self, frame: open_dataset.Frame, camera_name: int, 
                           ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """ Project LiDAR points to camera view to create depth map and return camera-to-world extrinsics.
        Args:
            frame: Waymo Frame object
            camera_name: Camera name enum value
            intrinsics: Camera intrinsic matrix
            extrinsics: World-to-camera extrinsic matrix (4x4)
            image_shape: (height, width) of target image
        Returns:
            Tuple of:
                - depth_map: 2D array (H, W) with depth values (z in camera frame)
                - extrinsics: 3x4 camera-to-world matrix for depth_to_world_coords_points"""        
        if not self.use_lidar_depth or len(frame.lasers) == 0: return None
        image = self._extract_image(frame, camera_name)
        intrinsics, extrinsics, _ = self._extract_camera_calibration(frame, camera_name, target_shape=(image.shape[:2]))
        range_images, camera_projections, _, range_image_top_pose = \
            frame_utils.parse_range_image_and_camera_projection(frame)
        points, cp_points = \
            frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose)
        # ------- Applying mask to the points -------
        points_all = np.concatenate(points, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
        mask = tf.equal(cp_points_all_tensor[..., 0], camera_name)
        cp_points_camera = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
        if cp_points_camera.shape[0] == 0: return None
        points_3d = points_all[mask.numpy()]
        # ------- END NEW ------------------------------------------------------------------
        # proj_ours = self.points2img2(points_3d, np.linalg.inv(extrinsics), intrinsics@T)
        # print(T.shape, '===========1==========', points_3d.shape)
        #proj_ours = self.points2img2(points_3d, T@extrinsics, intrinsics)
        #proj_ours = self.points2img2( f(T, points_3d, extrinsics), extrinsics, intrinsics)
        new_points_3d = transform_points(points_3d, extrinsics, T)
        proj_ours = self.points2img2(new_points_3d, extrinsics, intrinsics)
        # ------- Do Projection -------
        height, width = image.shape[:2]
        pixels_f = cp_points_camera[..., 1:3].numpy().astype(np.float64)  # (N,2) [u,v] in native domain
        u_f, v_f = pixels_f[:, 0], pixels_f[:, 1]
        valid_mask = (u_f >= 0) & (u_f < width) & (v_f >= 0) & (v_f < height) & np.isfinite(u_f) & np.isfinite(v_f)
        u_f, v_f = u_f[valid_mask], v_f[valid_mask]
        z = proj_ours[valid_mask, 2]
        x = np.rint(u_f).astype(np.int32)
        y = np.rint(v_f).astype(np.int32)
        x = np.clip(x, 0, width  - 1)
        y = np.clip(y, 0, height - 1)
        depth_map = np.zeros((height, width), dtype=np.float32)
        depth_map[y, x] = z.astype(np.float32)
        if np.any(depth_map > 0):
            depth_map = threshold_depth_map(depth_map, max_percentile=98)
        # points, _, colors = depth_to_world_points(depth_map, intrinsics, extrinsics, image)
        # print(np.min(depth_map), np.max(depth_map), '===========')
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        # pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        # o3d.io.write_point_cloud("text3_.ply", pcd, write_ascii=True)
        # pdb.set_trace()
        # print(pixels_f[valid_mask, :2][:5])
        # print('=======SHANF=======XIA')
        # print(proj_ours[valid_mask, :2][:5].astype(np.int32))
        # du = float(np.mean(np.abs(proj_ours[valid_mask,0] - u_f)))
        # dv = float(np.mean(np.abs(proj_ours[valid_mask,1] - v_f)))
        # print("mean offset (u,v) =", du, dv)
        # pdb.set_trace()
        # # ===== 新增：用 3D-2D 对应直接拟合一个新的 K' =====
        # Kp_3x3, _ = fit_pinhole_K_from_corresp(points_3d[valid_mask], extrinsics, pixels_f[valid_mask, :2], trim_ratio=0.2, tie_fx_fy=False)
        # intrinsics_refit = intrinsics.copy()
        # intrinsics_refit[:3, :3] = Kp_3x3
        # proj_ours_refit = self.points2img2(points_3d, extrinsics, intrinsics_refit@T)
        # print(proj_ours_refit[valid_mask, :2][:5].astype(np.int32), '===============')
        # # ------- END NEW ------------------------------------------------------------------
        # # Optional: write colored point cloud (camera-space)
        # color_point = image[y, x]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(new_points_3d[valid_mask].astype(np.float64))
        # pcd.colors = o3d.utility.Vector3dVector(color_point.astype(np.float64)/255.)
        # o3d.io.write_point_cloud("text3_.ply", pcd, write_ascii=True)
        # pdb.set_trace()
        # ================== VERIFICATION & BACK-PROJECTION (append-only) ==================
        # 1) Reproject check: camera points -> pixels (compare with Waymo cp pixels x,y)
        # fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        # cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        # u_proj = fx * (points_camera[valid_mask, 0] / points_camera[valid_mask, 2]) + cx
        # v_proj = fy * (points_camera[valid_mask, 1] / points_camera[valid_mask, 2]) + cy
        # reproj_err = np.sqrt((u_proj - u_f)**2 + (v_proj - v_f)**2)
        # print(f"[verify] pixel reprojection error  mean={reproj_err.mean():.3f}px  "
        #     f"median={np.median(reproj_err):.3f}px  max={reproj_err.max():.3f}px  N={reproj_err.size}")
        # pdb.set_trace()

        # # 2) Back-project: (x,y,Z) -> camera 3D using intrinsics, then camera -> world using extrinsics
        # X_bp = (x - cx) * z / fx
        # Y_bp = (y - cy) * z / fy
        # Z_bp = z.copy()
        # Pc_bp = np.stack([X_bp, Y_bp, Z_bp], axis=1)            # (N,3), camera-frame back-projection
        # Pc_gt = points_camera[valid_mask]                        # (N,3)
        # cam_err = np.linalg.norm(Pc_bp - Pc_gt, axis=1)
        # print(f"[verify] camera-frame round-trip error  mean={cam_err.mean():.6e} m  "
        #     f"median={np.median(cam_err):.6e} m  max={cam_err.max():.6e} m")

        # # Camera -> world using your camera_to_world (4x4)
        # ones = np.ones((Pc_bp.shape[0], 1), dtype=Pc_bp.dtype)
        # Pc_bp_h = np.hstack([Pc_bp, ones])                       # (N,4)
        # Pw_bp_h = (camera_to_world @ Pc_bp_h.T).T                # (N,4)
        # Pw_bp   = Pw_bp_h[:, :3]                                 # (N,3)
        # Pw_gt   = points_3d[valid_mask]                          # (N,3)
        # world_err = np.linalg.norm(Pw_bp - Pw_gt, axis=1)
        # print(f"[verify] world-frame round-trip error  mean={world_err.mean():.6e} m  "
        #     f"median={np.median(world_err):.6e} m  max={world_err.max():.6e} m")

        # # 3) Save the back-projected world points as text4.ply
        # color_bp = image[y, x]                                   # (N,3), uint8
        # pcd_bp = o3d.geometry.PointCloud()
        # pcd_bp.points = o3d.utility.Vector3dVector(Pw_bp.astype(np.float64))
        # pcd_bp.colors = o3d.utility.Vector3dVector((color_bp.astype(np.float64) / 255.0))
        # o3d.io.write_point_cloud("text4.ply", pcd_bp, write_ascii=True)
        # print("[verify] Saved back-projected world points to text4.ply")
        # pdb.set_trace()
        return depth_map, intrinsics, extrinsics, image
        # except Exception as e:
        #     if self.verbose:
        #         print(f"Could not extract LiDAR depth: {e}")
        #     return None
    def _extract_tracks_from_labels(self, frames: List[open_dataset.Frame], 
                                camera_name: int, frame_indices: List[int],
                                target_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract 2D tracks from Waymo camera labels.
        Args:
            frames: List of Waymo Frame objects
            camera_name: Camera name enum value
            frame_indices: Indices of frames to extract
            target_shape: Target image shape for track coordinates  (H, W)
        Returns:
            tracks: (N, T, 2) array of track positions (float32, in target image pixels)
            track_masks: (N, T) array of track visibility (bool)
        """
        try:
            target_h, target_w = target_shape
            # ====== 1) 原图尺寸（Waymo 相机原始分辨率）======
            def get_source_hw(frame0, cam_name):
                for cc in frame0.context.camera_calibrations:
                    if cc.name == cam_name:
                        return int(cc.height), int(cc.width)
                return target_h, target_w  # 兜底
            source_h, source_w = get_source_hw(frames[0], camera_name)
            # ====== 2) source→target 等比缩放 + 居中黑边 映射 ======
            scale = min(target_w / float(source_w), target_h / float(source_h))
            new_w, new_h = int(round(source_w * scale)), int(round(source_h * scale))
            x0 = (target_w - new_w) // 2
            y0 = (target_h - new_h) // 2
            def map_src_to_tgt(x_src: float, y_src: float) -> Tuple[float, float]:
                x_t = x_src * scale + x0
                y_t = y_src * scale + y0
                return x_t, y_t
            # ====== 3) 收集每帧的 labels，并汇总 ID ======
            all_track_ids = set()
            frame_labels = {}
            for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
                labels_for_frame = []
                for cam_labels in frame.camera_labels:
                    if cam_labels.name == camera_name:
                        for obj in cam_labels.labels:
                            labels_for_frame.append({
                                'id': obj.id,
                                'box': obj.box,  # 像素坐标，源图系
                                'type': obj.type
                            })
                            all_track_ids.add(obj.id)
                        break
                frame_labels[i] = labels_for_frame
            if not all_track_ids:
                return None, None
            track_ids = sorted(list(all_track_ids))
            target_track_num = self.track_num
            num_frames = len(frame_indices)
            # ====== 4) 选 ID：超出则保留观测最多的 ======
            if len(track_ids) > target_track_num:
                track_counts = {}
                for i in range(num_frames):
                    for label in frame_labels[i]:
                        tid = label['id']
                        track_counts[tid] = track_counts.get(tid, 0) + 1
                track_ids = sorted(track_ids, key=lambda tid: (-track_counts.get(tid, 0), tid))
                track_ids = track_ids[:target_track_num]
            # ====== 5) 构建 (N,T,2) 与 (N,T) ======
            num_tracks = len(track_ids)
            tracks = np.zeros((num_tracks, num_frames, 2), dtype=np.float32)
            track_masks = np.zeros((num_tracks, num_frames), dtype=bool)
            id2idx = {tid: i for i, tid in enumerate(track_ids)}
            # 填真实轨迹
            for frame_i in range(num_frames):
                for label in frame_labels[frame_i]:
                    tid = label['id']
                    if tid in id2idx:
                        ti = id2idx[tid]
                        box = label['box']
                        x_src = float(box.center_x)
                        y_src = float(box.center_y)
                        x_t, y_t = map_src_to_tgt(x_src, y_src)
                        if 0 <= x_t < target_w and 0 <= y_t < target_h:
                            tracks[ti, frame_i, 0] = x_t
                            tracks[ti, frame_i, 1] = y_t
                            track_masks[ti, frame_i] = True
            # ====== 6) 数量不够：选择“复制真实ID补足”或“零填充” ======
            oversample = True   # <--- 想只要真实轨迹且补满：设 True；不想复制：改 False
            if num_tracks < target_track_num:
                deficit = target_track_num - num_tracks
                if oversample and num_tracks > 0:
                    # 有放回采样已有真实ID，纯复制，不加噪
                    idxs = np.random.default_rng(seed=0).choice(num_tracks, size=deficit, replace=True)
                    tracks = np.concatenate([tracks, tracks[idxs]], axis=0)
                    track_masks = np.concatenate([track_masks, track_masks[idxs]], axis=0)
                else:
                    # 直接零填充（全 0 + mask=False）
                    pad_tracks = np.zeros((deficit, num_frames, 2), dtype=np.float32)
                    pad_masks = np.zeros((deficit, num_frames), dtype=bool)
                    tracks = np.concatenate([tracks, pad_tracks], axis=0)
                    track_masks = np.concatenate([track_masks, pad_masks], axis=0)
            # ====== 7) 清 NaN/Inf ======
            nan_mask = np.isnan(tracks).any(axis=2)
            inf_mask = np.isinf(tracks).any(axis=2)
            invalid_mask = nan_mask | inf_mask
            track_masks = track_masks & (~invalid_mask)
            tracks = np.where(np.isnan(tracks) | np.isinf(tracks), 0.0, tracks)
            return tracks, track_masks
        except Exception as e:
            if self.verbose:
                print(f"Could not extract tracks from labels: {e}")
            return None, None

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
        camera_name_str = clip_info['camera_name']
        # Convert camera name to enum
        camera_name = getattr(open_dataset.CameraName, camera_name_str, open_dataset.CameraName.FRONT)
        # Get sequence metadata
        seq_meta = self.sequence_metadata[seq_name]
        # Extend indices if img_per_seq > len(base_frame_indices)
        if img_per_seq:
            frame_indices, FLAG = self._contiguous_window_impr(available_frames, base_frame_indices, img_per_seq)
        else:
            frame_indices = base_frame_indices
        # Get target image shape
        target_image_shape = self.get_target_shape(aspect_ratio)
        # Load frames efficiently (either from cache or tfrecord)
        frames = []
        dataset = tf.data.TFRecordDataset(seq_meta['tfrecord_path'], compression_type='')
        for frame_idx in frame_indices:
            for i, data in enumerate(dataset):
                if i == frame_idx:
                    frame = open_dataset.Frame()
                    frame.ParseFromString(bytearray(data.numpy()))
                    frames.append(frame)
                    break
        images_raw = []; depths_raw = []; extrinsics_raw = []; intrinsics_raw = []
        for num, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
            # Extract depth from LiDAR projection
            depth_map, intrinsic, extrinsic, image = self._extract_lidar_depth(frame, camera_name)
            T_veh_to_world = np.array(frame.pose.transform, dtype=np.float64).reshape(4,4)  # 车辆→世界
            T_world_to_veh = np.linalg.inv(T_veh_to_world)
            extrinsic = T @ (extrinsic@T_world_to_veh)
            intrinsic = intrinsic[:3, :3]
            original_size = np.array(image.shape[:2])
            images_raw.append(image)
            depths_raw.append(depth_map)
            extrinsics_raw.append(extrinsic)
            intrinsics_raw.append(intrinsic)
        if self.ENABLE_TRACK:
            tracks, track_masks = self._extract_tracks_from_labels(
                frames, camera_name, frame_indices, original_size)
        elif not self.ENABLE_TRACK:
            tracks = None
            track_masks = None

        if tracks is not None:
            tracks = tracks.transpose(1, 0, 2)  # (N, T, 2) -> (T, N, 2)
            track_masks = track_masks.transpose(1, 0)  # (N, T) -> (T, N)
        else:
            tracks = np.zeros((len(frames), self.track_num, 2))
            track_masks = np.zeros((len(frames), self.track_num))
        images = []; depths = []; extrinsics = []; intrinsics = []
        cam_points = []; world_points = []; point_masks = []; processed_tracks = []; processed_track_masks = []
        for num, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
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
                images_raw[num],
                depths_raw[num],
                extrinsics_raw[num],
                intrinsics_raw[num],
                original_size,
                target_image_shape,
                track=tracks[num],
                filepath=f"waymo_{seq_name}_frame{frame_idx:04d}",
                resize_ours=True,
            )
            # check = check_coord_system(world_coords_points.reshape(-1, 3)[point_mask.reshape(-1)], processed_extri)
            # print(check, '===========check===========')
            # matrix_check(processed_image, processed_depth, processed_intri, processed_extri)

            # print(np.mean(depths_raw[num]), '1===========', np.max(depths_raw[num]), np.sum(depths_raw[num]>0))
            # print(np.mean(processed_depth), '2===========', np.max(processed_depth), np.sum(processed_depth>0))
            # print(extrinsics_raw[num], '===========', intrinsics_raw[num])
            # print(processed_extri, '===========', processed_intri)
            # print(point_mask.shape, '===========', world_coords_points.shape)
            # points, _, _ = depth_to_world_coords_points(depths_raw[num], extrinsics_raw[num], intrinsics_raw[num], eps=1e-8)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3).astype(np.float64))
            # o3d.io.write_point_cloud(f"text3_{seq_name[:4]}_{frame_idx:04d}.ply", pcd, write_ascii=True)

            # print(point_mask.shape, '===========2222222')
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(world_coords_points.reshape(-1, 3).astype(np.float64))
            # o3d.io.write_point_cloud(f"text3_{seq_name[:4]}_{frame_idx:04d}_world1.ply", pcd, write_ascii=True)

            # world_coords_points2, cam_coords_points2, point_mask2 = depth_to_world_coords_points(processed_depth, processed_extri, processed_intri, eps=1e-8)
            # print(type(world_coords_points2), type(cam_coords_points2), type(point_mask2), '===========3333333', world_coords_points2.shape)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(world_coords_points2.reshape(-1, 3).astype(np.float64))
            # o3d.io.write_point_cloud(f"text3_{seq_name[:4]}_{frame_idx:04d}_world3.ply", pcd, write_ascii=True)
            # pdb.set_trace()
            images.append(processed_image)
            depths.append(processed_depth)
            extrinsics.append(processed_extri)
            intrinsics.append(processed_intri)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            processed_tracks.append(processed_track)
            processed_track_masks.append(processed_track_mask*track_masks[num])
            # Clear frame from memory if not caching
            if not self.cache_frames_in_memory:
                del frame  # Help garbage collection
        # Return in VGGT format
        return {
            "seq_name": f"waymo_{seq_name}",
            "ids": np.array(frame_indices),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "abandon_pose": False, # This one is for Dataset w/o pose.
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
        
        Args:
            frame_indices: List of frame indices
            
        Returns:
            Normalized temporal features for each frame in [-1, 1] range
        """
        if not frame_indices:
            return np.array([])
            
        frame_indices_array = np.array(frame_indices)
        if len(frame_indices_array) > 1:
            min_frame = frame_indices_array.min()
            max_frame = frame_indices_array.max()
            frame_range = max_frame - min_frame
            if frame_range > 0:
                temporal_features = 2.0 * (frame_indices_array - min_frame) / frame_range - 1.0
            else:
                temporal_features = np.full_like(frame_indices_array, 0.0, dtype=np.float32)
        else:
            temporal_features = np.full_like(frame_indices_array, 0.0, dtype=np.float32)
            
        return temporal_features.astype(np.float32)

def depth_to_world_points(depth, K, E_w2c, rgb=None, depth_valid_min=1e-6, flatten=True):
    """
    Back-project a depth map to a 3D point cloud in WORLD coordinates.

    Args:
        depth (H, W): depth map in meters (camera Z).
        K (3, 3): camera intrinsics (fx, fy, cx, cy).
        E_w2c (4, 4): world->camera extrinsic (OpenCV convention).
        rgb (H, W, 3) or None: optional colors aligned with depth (uint8 or float).
        depth_valid_min (float): minimum valid depth threshold.
        flatten (bool): if True, returns Nx3 (and Nx3 colors if provided). If False, returns HxWx3.
    Returns:
        Pw: (N, 3) world points (or HxWx3 if flatten=False).
        mask: (N,) boolean valid mask (or HxW if flatten=False).
        colors: (N, 3) float in [0,1] if rgb provided (or HxWx3 if flatten=False), else None.
    """
    assert depth.ndim == 2, "depth must be HxW"
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Build pixel grid
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H, W)
    # Valid mask
    mask = np.isfinite(depth) & (depth > depth_valid_min)
    # Back-project to CAMERA coordinates
    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    Pc = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
    # Transform CAMERA -> WORLD using inverse of world->camera
    E_c2w = np.linalg.inv(E_w2c)
    R_c2w = E_c2w[:3, :3]
    t_c2w = E_c2w[:3, 3]
    Pw = (Pc @ R_c2w.T) + t_c2w  # (H, W, 3)
    # Colors (optional)
    colors = None
    if rgb is not None:
        rgb = rgb.astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0  # normalize to [0,1]
        colors = rgb
    if flatten:
        Pw = Pw[mask]
        mask_out = mask.reshape(-1)
        if colors is not None:
            colors = colors.reshape(-1, 3)[mask_out]
        return Pw, mask_out, colors
    else:
        return Pw, mask, colors

def fit_pinhole_K_from_corresp(points_3d,            # (N,3) 车辆/世界坐标系
                               T_veh2cam_4x4,       # (4,4) 世界->相机 外参（与你投影时使用的保持一致）
                               pixels_uv,           # (N,2) Waymo cp_points 浮点像素 (u,v)，不要round
                               trim_ratio=0.2,      # 截尾比例，去掉最差的20%再拟合一次
                               tie_fx_fy=False,     # 如需要可约束 fx=fy
                               min_Z=1e-6):
    """从 3D-2D 对应直接拟合 K' = [[fx,0,cx],[0,fy,cy],[0,0,1]]（无畸变假设）"""
    points_3d = np.asarray(points_3d, dtype=np.float64)
    T = np.asarray(T_veh2cam_4x4, dtype=np.float64)
    uv = np.asarray(pixels_uv, dtype=np.float64)
    # 世界->相机
    R = T[:3, :3]; t = T[:3, 3]
    Pcam = (R @ points_3d.T + t.reshape(3,1)).T  # (N,3)
    X, Y, Z = Pcam[:,0], Pcam[:,1], Pcam[:,2]
    # 过滤
    Zsafe = np.where(Z <= min_Z, min_Z, Z)
    x = X / Zsafe; y = Y / Zsafe
    u = uv[:,0];   v = uv[:,1]
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v) & (Z > 0)
    x, y, u, v = x[m], y[m], u[m], v[m]
    if x.size < 20:
        raise RuntimeError("Not enough valid correspondences to fit K'.")
    # 线性最小二乘
    def solve_once(x, y, u, v, tie_fx_fy):
        if tie_fx_fy:
            # 约束 fx=fy=f：把 u,v 两式一起解 [f, cx, cy]
            # u = f*x + cx, v = f*y + cy
            A = np.concatenate([np.stack([x, np.ones_like(x), np.zeros_like(x)], axis=1),
                                np.stack([y, np.zeros_like(y), np.ones_like(y)], axis=1)], axis=0)
            b = np.concatenate([u, v], axis=0)
            f, cx, cy = np.linalg.lstsq(A, b, rcond=None)[0]
            fx = fy = f
        else:
            Au = np.stack([x, np.ones_like(x)], axis=1)   # [fx, cx]
            Av = np.stack([y, np.ones_like(y)], axis=1)   # [fy, cy]
            fx, cx = np.linalg.lstsq(Au, u, rcond=None)[0]
            fy, cy = np.linalg.lstsq(Av, v, rcond=None)[0]
        return fx, fy, cx, cy
    fx, fy, cx, cy = solve_once(x, y, u, v, tie_fx_fy)
    # 误差 + 截尾重拟合
    u_hat = fx * x + cx
    v_hat = fy * y + cy
    err = np.sqrt((u_hat - u)**2 + (v_hat - v)**2)
    if trim_ratio > 0 and err.size > 50:
        keep = err <= np.quantile(err, 1.0 - trim_ratio)
        fx, fy, cx, cy = solve_once(x[keep], y[keep], u[keep], v[keep], tie_fx_fy)
        u_hat = fx * x + cx
        v_hat = fy * y + cy
        err = np.sqrt((u_hat - u)**2 + (v_hat - v)**2)
    Kp = np.array([[fx, 0.0, cx],
                   [0.0, fy, cy],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    stats = dict(mean_err=float(err.mean()), med_err=float(np.median(err)), n=int(err.size))
    return Kp, stats

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

def discover_tfrecords(root):
    files = glob.glob(os.path.join(root, 'training_*', '*_with_camera_labels.tfrecord'))
    if not files:
        files = glob.glob(os.path.join(root, '*_with_camera_labels.tfrecord'))
    return sorted(files)
def count_frames(tfrecord_path):
    ds = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    n = 0
    for _ in ds:
        n += 1
    return n
def build_manifest(root, manifest_path):
    tfrecords = discover_tfrecords(root)
    man = {}
    for p in tqdm(tfrecords):
        seq = os.path.basename(p).replace('_with_camera_labels.tfrecord','')
        n = count_frames(p)
        man[seq] = {"tfrecord_path": p, "num_frames": n}
        print(f"[manifest] {seq}: {n} frames")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(man, f)
    print(f"[manifest] wrote {manifest_path} ({len(man)} sequences)")

if __name__ == "__main__":    
    build_manifest('/workspace/data/kaichen/data/waymo/archived_files/training', '/workspace/data/kaichen/data/waymo/archived_files/training/manifest.json')
    # Create a simple config object
    # class SimpleConfig:
    #     def __init__(self):
    #         self.img_size = 224
    #         self.patch_size = 14
    #         self.training = True
    #         self.rescale = True
    #         self.rescale_aug = False
    #         self.landscape_check = False
    #         self.debug = False
    #         self.get_nearby = True
    #         self.load_depth = True
    #         self.inside_random = False
    #         self.augs = {'scales': [0.8, 1.2]}
    #         self.track_num = 256  # Number of tracks to extract
    
    # config = SimpleConfig()
    
    # # Test auto-discovery
    # print("\n=== Testing auto-discovery ===")
    # dataset = WaymoDataset(
    #     common_conf=config,
    #     dataset_location='/workspace/data/kaichen/data/waymo/archived_files/training',
    #     sequence_names=None,  # Auto-discover
    #     camera_names=['FRONT'],  # Use front camera
    #     quick=True,  # Quick mode for testing
    #     verbose=True,
    #     len_train=100,
    #     use_lidar_depth=True,
    #     cache_frames_in_memory=False  # Don't cache to save RAM
    # )
    
    # print(f"Dataset length: {len(dataset)}")
    
    # if len(dataset) > 0:
    #     # Test getting a sample
    #     print("\n=== Testing data loading ===")
    #     sample = dataset.get_data(seq_index=0, img_per_seq=4, aspect_ratio=1.0)
    #     print(f"Sample keys: {sample.keys()}")
    #     print(f"Images shape: {[img.shape for img in sample['images']]}")
    #     print(f"IDs: {sample['ids']}")
        
    #     if sample['tracks'] is not None and len(sample['tracks']) > 0:
    #         print(f"Tracks shape: {[t.shape if t is not None else None for t in sample['tracks']]}")
    #         print(f"Track masks shape: {sample['track_masks'].shape if sample['track_masks'] is not None else None}")
        
    #     if sample['depths'][0] is not None:
    #         print(f"Depth shape: {sample['depths'][0].shape}")
    #         print(f"Depth range: [{np.min(sample['depths'][0]):.2f}, {np.max(sample['depths'][0]):.2f}]")
        
    #     # Test the corrected depth generation
    #     print("\n=== Testing corrected depth generation ===")
    #     dataset.test_depth_generation(seq_index=0)
        
    #     print("Test completed successfully!")
    # else:
    #     print("No data found - check dataset path and ensure tfrecord files exist")



