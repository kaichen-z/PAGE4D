# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os.path as osp
import os
import logging
import shutil
import open3d as o3d

import cv2
import random
import numpy as np
from typing import Optional, List, Dict, Any


from data.dataset_util import *
from data.base_dataset import BaseDataset


SEEN_CATEGORIES = [
    "apple",
    "banana",
    "baseballbat",
    "baseballglove",
    "bicycle",
    "bottle",
    "broccoli",
    "cake",
    "car",
    "couch",
    "cup",
    "donut",
    "frisbee",
    "handbag",
    "hotdog",
    "kite",
    "microwave",
    "motorcycle",
    "parkingmeter",
    "pizza",
    "sandwich",
    "skateboard",
    "stopsign",
    "toaster",
    "toybus",
    "toyplane",
    "toytrain",
    "tv",
    "bowl",
    "carrot",
    "hairdryer",
    "laptop",
    "wineglass",
]

def check_coord_system(points_3d_world, extrinsics):
    """
    简单判断相机坐标系是 OpenCV (+Z forward) 还是 OpenGL (-Z forward)
    只看 z 坐标的符号。
    Args:
        points_3d_world: (N,3)
        extrinsics: (4,4) world->camera
    Returns:
        'opencv' or 'opengl' and the mean z
    """
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

def extend_extrinsics_batch(extrinsics_list):
    """
    Convert list to batch array, extend, then convert back to list
    More efficient for large lists
    """
    import numpy as np
    
    # Convert list to batch array (N, 3, 4)
    extrinsics_batch = np.stack(extrinsics_list, axis=0)
    N = extrinsics_batch.shape[0]
    
    # Create batch of 4x4 matrices
    extrinsics_4x4_batch = np.zeros((N, 4, 4))
    extrinsics_4x4_batch[:, :3, :] = extrinsics_batch  # Copy 3x4 parts
    extrinsics_4x4_batch[:, 3, 3] = 1.0  # Set bottom-right to 1
    
    # Convert back to list
    return [extrinsics_4x4_batch[i] for i in range(N)]

def save_depth_as_rgb(depth_map, out_path="depth_vis.png", cmap=cv2.COLORMAP_JET, max_depth=None):
    """
    Visualize depth map and save as RGB PNG.
    Args:
        depth_map (np.ndarray): Depth in float (H,W). Values can be in meters, mm, etc.
        out_path (str): Path to save the RGB visualization.
        cmap: OpenCV colormap (default JET).
        max_depth (float or None): Max depth for normalization. If None, uses depth_map.max().
    """
    # Handle invalid values
    depth = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    if max_depth is None:
        max_depth = np.percentile(depth[depth > 0], 98) if np.any(depth > 0) else 1.0
    # Normalize to 0–255
    depth_norm = np.clip(depth / max_depth, 0, 1)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    # Apply colormap (BGR)
    depth_color = cv2.applyColorMap(depth_uint8, cmap)
    # Save as PNG
    cv2.imwrite(out_path, depth_color)
    print(f"[OK] Depth visualization saved: {out_path}")
    return depth_color
class Co3dDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        CO3D_DIR: str = None,
        CO3D_ANNOTATION_DIR: str = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        sequence_names: Optional[List[str]] = None,  
        strides: List[int] = [1, 2, 3, 4, 5, 6, 7, 8],
    ):
        """
        Initialize the Co3dDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            CO3D_DIR (str): Directory path to CO3D data.
            CO3D_ANNOTATION_DIR (str): Directory path to CO3D annotations.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If CO3D_DIR or CO3D_ANNOTATION_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if CO3D_DIR is None or CO3D_ANNOTATION_DIR is None:
            raise ValueError("Both CO3D_DIR and CO3D_ANNOTATION_DIR must be specified.")

        category = sorted(SEEN_CATEGORIES)

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence = [] # set any invalid sequence names here

        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"CO3D_DIR is {CO3D_DIR}")

        self.CO3D_DIR = CO3D_DIR
        self.CO3D_ANNOTATION_DIR = CO3D_ANNOTATION_DIR

        total_frame_num = 0

        for c in category:
            for split_name in split_name_list:
                annotation_file = osp.join(
                    self.CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz"
                )

                try:
                    with gzip.open(annotation_file, "r") as fin:
                        annotation = json.loads(fin.read())
                except FileNotFoundError:
                    logging.error(f"Annotation file not found: {annotation_file}")
                    continue

                for seq_name, seq_data in annotation.items():
                    if len(seq_data) < min_num_images:
                        continue
                    if seq_name in self.invalid_sequence:
                        continue
                    total_frame_num += len(seq_data)
                    self.data_store[seq_name] = seq_data

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Co3D Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Co3D Data dataset length: {len(self)}")

        print(f"Successfully loaded {len(self.sequence_list)} Co3D sequences")
        self.ENABLE_TRACK = False

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.
        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
            
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]

        if ids is None:
            ids = np.random.choice(
                len(metadata), img_per_seq, replace=self.allow_duplicate_img
            )

        annos = [metadata[i] for i in ids]

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for anno in annos:
            filepath = anno["filepath"]

            image_path = osp.join(self.CO3D_DIR, filepath)
            image = read_image_cv2(image_path)

            if self.load_depth:
                depth_path = image_path.replace("/images", "/depths") + ".geometric.png"
                depth_map = read_depth(depth_path, 1.0)

                mvs_mask_path = image_path.replace(
                    "/images", "/depth_masks"
                ).replace(".jpg", ".png")
                mvs_mask = cv2.imread(mvs_mask_path, cv2.IMREAD_GRAYSCALE) > 128
                depth_map[~mvs_mask] = 0

                depth_map = threshold_depth_map(
                    depth_map, min_percentile=-1, max_percentile=98
                )
            else:
                depth_map = None
            # shutil.copy(depth_path, f"/workspace/code/12_4d/VGGT-4D_T/training/data_visualization/co3d_one_depth0.png")
            # save_depth_as_rgb(depth_map, f"/workspace/code/12_4d/VGGT-4D_T/training/data_visualization/co3d_one_depth1.png")
            original_size = np.array(image.shape[:2])
            extri_opencv = np.array(anno["extri"])
            intri_opencv = np.array(anno["intri"])
            # print(depth_map.shape, '==============depth_map==============', extri_opencv.shape, "===================", intri_opencv.shape)
            # points_world = reconstruct_pointcloud_world(depth_map, intri_opencv, extri_opencv)
            # save_ply(points_world, f"/workspace/code/12_4d/VGGT-4D_T/training/data_visualization/co3d_one_depth2.ply")
            # print(points_world.shape, '==============points_world==============')
            # pdb.set_trace()
            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=filepath,
                resize_ours=True,
            )
            # save_depth_as_rgb(depth_map, f"/workspace/code/12_4d/VGGT-4D_T/training/data_visualization/co3d_one_depth2.png")
            # print(world_coords_points.shape, '===========world_coords_points===========', extri_opencv.shape, '===========processed_extri===========')
            # check = check_coord_system(world_coords_points.reshape(-1, 3)[point_mask.reshape(-1)], extri_opencv)
            # print(check, '===========check===========')
            # matrix_check(image, depth_map, intri_opencv, extri_opencv)

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        set_name = "co3d"
        extrinsics = extend_extrinsics_batch(extrinsics)
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "abandon_pose": False,
            "abandon_geometry": False,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            "tracks": np.zeros((len(images), self.track_num, 2)),
            "track_masks": np.zeros((len(images), self.track_num)),
            "temporal_features": self._compute_temporal_features(ids),
        }
        return batch

    def _compute_temporal_features(self, frame_indices: List[int]) -> np.ndarray:
        """
        Compute temporal features for the given frame indices.
        Args: frame_indices: List of frame indices (e.g., [0, 100, 200, 300])
        Returns: np.ndarray: Normalized temporal features for each frame in [-1, 1] range
        """
        frame_indices_array = np.zeros(len(frame_indices))
        return frame_indices_array.astype(np.float32)
    
def reconstruct_pointcloud_world(
    depth_map: np.ndarray,          # (H, W) float32, meters
    intri_opencv: np.ndarray,       # (3, 3)
    extri_opencv_3x4: np.ndarray,   # (3, 4), usually [R|t]
    assume_extri_world_to_cam: bool = True
):
    H, W = depth_map.shape[:2]

    # Intrinsics
    fx, fy = intri_opencv[0, 0], intri_opencv[1, 1]
    cx, cy = intri_opencv[0, 2], intri_opencv[1, 2]

    # Homogenize extrinsic to 4x4
    extri_4x4 = np.eye(4, dtype=np.float32)
    extri_4x4[:3, :4] = extri_opencv_3x4  # [R|t] in the top rows

    # Pixel grid
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    z = depth_map.astype(np.float32)  # meters
    valid = np.isfinite(z) & (z > 0)

    # Backproject to camera coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Build homogeneous camera points (4,N)
    cam_pts = np.stack([x[valid], y[valid], z[valid], np.ones_like(z[valid])], axis=0)

    # Convert to world coordinates
    if assume_extri_world_to_cam:
        # extri maps world -> cam, so invert to get cam -> world
        cam_to_world = np.linalg.inv(extri_4x4)
        world_pts = cam_to_world @ cam_pts
    else:
        # extri already cam -> world
        world_pts = extri_4x4 @ cam_pts

    world_pts = world_pts[:3].T  # (M,3), only valid pixels
    return world_pts

def save_ply(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)