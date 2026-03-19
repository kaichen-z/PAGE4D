import sys
import os
import numpy as np
import cv2
import json
from typing import Optional
import argparse
from tqdm import tqdm
import imageio
import matplotlib
from PIL import Image
from dataset_util import crop_image_depth_and_intrinsic_by_pp, resize_image_depth_and_intrinsic

#nohup python VGGT-4D/training/data/preprocess_kubric_tracks.py --dataset_location /shared/ssd_14T/gaspard/output --visualize --overwrite --preprocess_tracks > preprocess_logs.txt 2>&1 &

def plot_2d_tracks(video, points, visibles, infront_cameras=None, tracks_leave_trace=16, show_occ=False):
  """Visualize 2D point trajectories."""
  num_frames, num_points = points.shape[:2]

  # Precompute colormap for points
  color_map = matplotlib.colormaps.get_cmap('hsv')
  cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)
  point_colors = np.zeros((num_points, 3), dtype=np.uint8)
  for i in range(num_points):
    point_colors[i] = (np.array(color_map(cmap_norm(i)))[:3] * 255).astype(np.uint8)

  if infront_cameras is None:
    infront_cameras = np.ones_like(visibles).astype(bool)

  frames = []
  for t in range(num_frames):
    # Ensure frame is a proper numpy array with correct dtype
    frame = video[t].copy()
    if not isinstance(frame, np.ndarray):
      print(f"Warning: Frame {t} is not a numpy array, skipping")
      continue
    if frame.dtype != np.uint8:
      frame = frame.astype(np.uint8)

    # Draw tracks on the frame
    line_tracks = points[max(0, t - tracks_leave_trace) : t + 1]
    line_visibles = visibles[max(0, t - tracks_leave_trace) : t + 1]
    line_infront_cameras = infront_cameras[max(0, t - tracks_leave_trace) : t + 1]
    
    if line_tracks.shape[0] > 1:
        for s in range(line_tracks.shape[0] - 1):
            img = frame.copy()

            for i in range(num_points):
                if line_visibles[s, i] and line_visibles[s + 1, i]:  # visible
                    x1, y1 = int(round(line_tracks[s, i, 0])), int(round(line_tracks[s, i, 1]))
                    x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(round(line_tracks[s + 1, i, 1]))
                    cv2.line(frame, (x1, y1), (x2, y2), tuple(point_colors[i].tolist()), 1, cv2.LINE_AA)
                elif show_occ and line_infront_cameras[s, i] and line_infront_cameras[s + 1, i]:  # occluded
                    x1, y1 = int(round(line_tracks[s, i, 0])), int(round(line_tracks[s, i, 1]))
                    x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(round(line_tracks[s + 1, i, 1]))
                    cv2.line(frame, (x1, y1), (x2, y2), tuple(point_colors[i].tolist()), 1, cv2.LINE_AA)

            alpha = (s + 1) / (line_tracks.shape[0] - 1)
            frame = cv2.addWeighted(frame, alpha, img, 1 - alpha, 0)

    # Draw end points on the frame
    for i in range(num_points):
      if visibles[t, i]:  # visible
        x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
        cv2.circle(frame, (x, y), 2, tuple(point_colors[i].tolist()), -1, cv2.LINE_AA)
      elif show_occ and infront_cameras[t, i]:  # occluded
        x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
        cv2.circle(frame, (x, y), 2, tuple(point_colors[i].tolist()), 1, cv2.LINE_AA)

    frames.append(frame)
  
  if frames:
    frames = np.stack(frames)
  else:
    frames = np.array([])
  return frames

def get_intrinsic(focal_length, sensor_width, width, height):
    """Calculate the intrinsic parameters matrix of the camera."""
    f_x = focal_length / sensor_width * width
    sensor_height = sensor_width * height / width
    f_y = focal_length / sensor_height * height
    p_x = width / 2.
    p_y = height / 2.
    return np.array([
        [f_x, 0, p_x],
        [0, f_y, p_y],
        [0, 0, 1],
    ])

def batch_quaternion_to_rotation_matrix(quaternions):
    """Convert a batch of quaternions to rotation matrices."""
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    q0, q1, q2, q3 = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    rot = np.zeros(quaternions.shape[:-1] + (3, 3))
    rot[..., 0, 0] = 1 - 2 * (q2**2 + q3**2)
    rot[..., 0, 1] = 2 * (q1*q2 - q0*q3)
    rot[..., 0, 2] = 2 * (q0*q2 + q1*q3)
    rot[..., 1, 0] = 2 * (q1*q2 + q0*q3)
    rot[..., 1, 1] = 1 - 2 * (q1**2 + q3**2)
    rot[..., 1, 2] = 2 * (q2*q3 - q0*q1)
    rot[..., 2, 0] = 2 * (q1*q3 - q0*q2)
    rot[..., 2, 1] = 2 * (q0*q1 + q2*q3)
    rot[..., 2, 2] = 1 - 2 * (q1**2 + q2**2)
    return rot

def batch_get_matrix_world(rotations, translations):
    """Batch version of the homogeneous transformation matrix."""
    transforms = np.zeros(rotations.shape[:-2] + (4, 4), dtype=np.float32)
    transforms[..., :3, :3] = rotations
    transforms[..., :3, 3] = translations
    transforms[..., 3, 3] = 1
    return transforms

def camera2image(point3d, intrinsic):
    """Project 3D point in camera coordinate to [0, 1] image plane."""
    proj = point3d @ intrinsic.T
    z_proj = proj[..., 2:3]
    z_proj[z_proj == 0] = 1e-6
    image_coords = proj[..., :2] / z_proj
    z = proj[..., 2]
    return image_coords, z

def batch_bilinear_interpolate(im, x, y):
    """Bilinear interpolation for batch of images."""
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[-1] - 1)
    x1 = np.clip(x1, 0, im.shape[-1] - 1)
    y0 = np.clip(y0, 0, im.shape[-2] - 1)
    y1 = np.clip(y1, 0, im.shape[-2] - 1)

    b = np.arange(im.shape[0])[:, None, None]
    im_a = im[b, y0, x0]
    im_b = im[b, y1, x0]
    im_c = im[b, y0, x1]
    im_d = im[b, y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * im_a + wb * im_b + wc * im_c + wd * im_d

def sample_grid_points(height, width, stride=1):
    """Return [H/stride, W/stride, 2] grid points with x,y order."""
    grid = np.mgrid[stride//2:height:stride, stride//2:width:stride].transpose(1, 2, 0)
    return grid[..., ::-1]

def read_image_cv2(path: str) -> Optional[np.ndarray]:
    """Read image with OpenCV."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def extract_tracks_for_scene(
    seq_path: str,
    preprocess_tracks: bool = False,
    img_size: int = 224,
    patch_size: int = 14
):
    """
    Extracts 2D tracks and visibility masks for an entire scene.
    This function is now based exactly on data_processing_kubric from test_kubric_movi_e.py.
    """
    metadata_path = os.path.join(seq_path, "metadata.json")
    if not os.path.isfile(metadata_path):
        print(f"  Skipping {os.path.basename(seq_path)}: missing metadata.json")
        return None, None

    with open(metadata_path, "r") as f:
        seq_meta = json.load(f)

    rgba_files = sorted([f for f in os.listdir(seq_path) if f.startswith("rgba_") and f.endswith(".png")])
    if not rgba_files:
        print(f"  Skipping {os.path.basename(seq_path)}: no rgba frames found.")
        return None, None

    frame_indices = sorted([int(f.split("_")[-1].split(".")[0]) for f in rgba_files])

    try:
        num_frames = len(frame_indices)
        if num_frames < 2:
            print(f"  Skipping {os.path.basename(seq_path)}: need at least 2 frames.")
            return None, None

        # Load first image to get dimensions
        first_img_path = os.path.join(seq_path, f"rgba_{frame_indices[0]:05d}.png")
        image = read_image_cv2(first_img_path)
        if image is None: 
            return None, None
        height, width = image.shape[:2]

        # Load depth maps for all frames
        depth_range = seq_meta.get('metadata', {}).get('depth_range', [0.1, 100.0])
        depths = []
        for frame_idx in frame_indices:
            depth_path = os.path.join(seq_path, f"depth_{frame_idx:05d}.tiff")
            if os.path.exists(depth_path):
                depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_raw is not None:
                    if depth_raw.dtype == np.uint16:
                        # Convert exactly like in test script: from 16-bit to meters
                        depth_map = depth_raw.astype(np.float32) / 65535 * (depth_range[1] - depth_range[0]) + depth_range[0]
                    else:
                        # Already float (meters)
                        depth_map = depth_raw.astype(np.float32)
                    depths.append(depth_map)
                else: 
                    depths.append(np.zeros((height, width), dtype=np.float32))
            else: 
                depths.append(np.zeros((height, width), dtype=np.float32))
        depths = np.array(depths)  # Shape: (num_frames, height, width)

        # Load segmentation masks
        masks = []
        for frame_idx in frame_indices:
            mask_path = os.path.join(seq_path, f"segmentation_{frame_idx:05d}.png")
            if os.path.exists(mask_path):
                img_pil = Image.open(mask_path)
                mask = np.array(img_pil)  # Direct conversion from PIL palette image
                masks.append(mask)
            else:
                masks.append(np.zeros((height, width), dtype=np.uint8))
        masks = np.array(masks)  # Shape: (num_frames, height, width)

        # Retrieve list of object instances (needed for ID remapping below)
        instances = seq_meta.get("instances", [])

        # Get camera intrinsic
        cam_meta = seq_meta["camera"]
        focal_length = cam_meta.get("focal_length", 35.0)
        sensor_width = cam_meta.get("sensor_width", 32.0)
        intrinsic = get_intrinsic(focal_length, sensor_width, width, height)

        # Get camera poses for all frames
        camera_positions = []
        camera_quaternions = []
        available_frames_cam = len(cam_meta.get("positions", []))
        if available_frames_cam == 0: 
            return None, None
        for frame_idx in frame_indices:
            meta_idx = min(frame_idx, available_frames_cam - 1)
            camera_positions.append(cam_meta["positions"][meta_idx])
            camera_quaternions.append(cam_meta["quaternions"][meta_idx])
        camera_positions = np.array(camera_positions)  # Shape: (num_frames, 3)
        camera_quaternions = np.array(camera_quaternions)  # Shape: (num_frames, 4)

        # Convert camera quaternions to rotation matrices and poses
        camera_rotations = batch_quaternion_to_rotation_matrix(camera_quaternions)
        camera_rotations = camera_rotations @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # flip y and z axis
        camera_poses = batch_get_matrix_world(camera_rotations, camera_positions)
        camera_poses = np.linalg.inv(camera_poses)  # world to camera extrinsics

        # Object poses from metadata 
        # Build object poses arrays matching the test script format
        object_positions = []  # Will be (num_frames, num_objects, 3)
        object_quaternions = []  # Will be (num_frames, num_objects, 4)
        
        for frame_idx in frame_indices:
            frame_obj_pos = []
            frame_obj_quat = []
            for instance in instances:
                available_pos = len(instance.get("positions", []))
                available_quat = len(instance.get("quaternions", []))
                if available_pos == 0 or available_quat == 0: 
                    continue
                pos_idx = min(frame_idx, available_pos - 1)
                quat_idx = min(frame_idx, available_quat - 1)
                frame_obj_pos.append(instance["positions"][pos_idx])
                frame_obj_quat.append(instance["quaternions"][quat_idx])
            object_positions.append(frame_obj_pos)
            object_quaternions.append(frame_obj_quat)

        if not object_positions or not object_positions[0]:
            print(f"  Skipping {os.path.basename(seq_path)}: no object positions found in instances.")
            return None, None
            
        object_positions = np.array(object_positions)  # Shape: (num_frames, num_objects, 3)
        object_quaternions = np.array(object_quaternions)  # Shape: (num_frames, num_objects, 4)

        # Convert object quaternions to rotation matrices and poses
        object_rotations = batch_quaternion_to_rotation_matrix(object_quaternions)
        object_poses = batch_get_matrix_world(object_rotations, object_positions)
        
        # Add background (identity) to object poses - CRUCIAL step from test script
        identity = np.tile(np.eye(4)[None, None], (num_frames, 1, 1, 1))
        object_poses = np.concatenate((identity, object_poses), axis=1)  # add background to object poses

        # Build a robust mapping raw_segmentation_id -> object_index (+1)



        # Pre-compute camera matrix for projection
        K = intrinsic
        cam_extr_0 = camera_poses[0]  # world→camera (frame 0)


        # Backproject first frame to 3D - EXACTLY like in test script
        h, w = height, width
        v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        uv_homogeneous = np.stack([u.flatten(), v.flatten(), np.ones_like(u.flatten())])

        K_inv = np.linalg.inv(intrinsic)
        camera_plane_flat = (K_inv @ uv_homogeneous).T            # [H*W, 3]    (raw ray vectors)
        dirs = camera_plane_flat / np.linalg.norm(camera_plane_flat, axis=1, keepdims=True)
        camera_coords_flat = dirs * depths[0].flatten()[:, None]   # radial depth

        camera_coords = camera_coords_flat.reshape(h, w, 3)
        z_buffers = depths  # = camera_coords[...,2]

        # Compute object transformations - EXACTLY like in test script
        poses = camera_poses[:, None] @ object_poses  # Shape: (num_frames, num_objects, 4, 4)
        relative_poses = np.einsum('tkcd, kde -> tkce', poses, np.linalg.inv(poses[0]))
        num_objects = relative_poses.shape[1]
        one_hot_masks = (masks[0][None, None, ...] == np.arange(num_objects)[None, :, None, None])
        dense_poses = np.einsum('tkhw, tkce -> thwce', one_hot_masks, relative_poses)  # Shape: (num_frames, height, width, 4, 4)

        # Transform points to other frames - EXACTLY like in test script
        points4d = np.concatenate([camera_coords, np.ones_like(camera_coords[..., :1])], axis=-1)  # Homogeneous
        proj_camera_coords = np.einsum('thwcd, hwd -> thwc', dense_poses, points4d)  # Shape: (num_frames, height, width, 4)
        
        # Project to image coordinates
        image_coords_xy, image_coords_z = camera2image(proj_camera_coords[..., :3], intrinsic)  # Shape: (num_frames, height, width, 2)

        # Visibility check - EXACTLY like in test script
        interpolate_z_buffers = batch_bilinear_interpolate(z_buffers, image_coords_xy[..., 0], image_coords_xy[..., 1])
        visible_all = (image_coords_z <= interpolate_z_buffers * 1.01 ) & \
            (image_coords_xy[..., 0] >= 0) & (image_coords_xy[..., 0] < width) & \
            (image_coords_xy[..., 1] >= 0) & (image_coords_xy[..., 1] < height)

        # Sample grid points and extract trajectories - EXACTLY like in test script
        grid = sample_grid_points(height, width, 8)
        grid = grid.reshape(-1, 2)
        
        # Extract trajectories for grid points
        tracks_2d = image_coords_xy[:, grid[:, 1], grid[:, 0]]  # Shape: (num_frames, N, 2)
        track_visibility = visible_all[:, grid[:, 1], grid[:, 0]]  # Shape: (num_frames, N)
        
        # Transpose to get format expected by VGGT: (N, T, 2) and (N, T)
        tracks = tracks_2d.transpose(1, 0, 2)  # Shape: (N, num_frames, 2)
        track_masks = track_visibility.transpose(1, 0)  # Shape: (N, num_frames)

        if preprocess_tracks:
            print("  Preprocessing tracks...")
            # This logic mimics BaseDataset.process_one_image without random augmentation
            
            # 1. Calculate target shape
            aspect_ratio = width / height
            short_size = int(img_size * aspect_ratio)
            if short_size % patch_size != 0:
                short_size = (short_size // patch_size) * patch_size
            target_shape = np.array([short_size, img_size])
            
            processed_tracks_all_frames = []

            for t in range(num_frames):
                track_t = tracks_2d[t] # Use (N, 2) tracks for one frame
                intrinsic_t = intrinsic.copy()
                
                # We need a dummy image just for shape info for the util functions
                dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
                original_size = np.array([height, width])
                
                # No random augmentation, so aug_size is original_size
                aug_size = original_size
                
                # --- Mimic process_one_image ---
                # 1. First crop
                _, _, intr_1, track_1 = crop_image_depth_and_intrinsic_by_pp(
                    dummy_image, None, intrinsic_t, aug_size, track=track_t,
                )
                
                # 2. Resize
                image_after_crop = np.zeros(aug_size.tolist() + [3], dtype=np.uint8)
                _, _, intr_2, track_2 = resize_image_depth_and_intrinsic(
                    image_after_crop, None, intr_1, target_shape, aug_size, track=track_1,
                    rescale_aug=False
                )
                
                # 3. Final crop
                image_after_resize = np.zeros(target_shape.tolist() + [3], dtype=np.uint8)
                _, _, _, final_track = crop_image_depth_and_intrinsic_by_pp(
                    image_after_resize, None, intr_2, target_shape, track=track_2, strict=True
                )
                
                processed_tracks_all_frames.append(final_track)

            processed_tracks_2d = np.stack(processed_tracks_all_frames) # (T, N, 2)
            tracks = processed_tracks_2d.transpose(1, 0, 2) # (N, T, 2)

        return tracks, track_masks

    except Exception as e:
        print(f"Error extracting tracks for {os.path.basename(seq_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main(dataset_location: str, visualize: bool, overwrite: bool, preprocess_tracks: bool, img_size: int, patch_size: int):
    """Preprocess all scenes in the dataset to extract tracks."""
    scene_names = sorted([d for d in os.listdir(dataset_location)
                           if os.path.isdir(os.path.join(dataset_location, d)) and d.startswith('scene_')])
                           
    for scene_name in tqdm(scene_names, desc="Processing scenes"):
        scene_path = os.path.join(dataset_location, scene_name)
        
        tracks_path = os.path.join(scene_path, 'tracks_pp.npy' if preprocess_tracks else 'tracks.npy')
        track_masks_path = os.path.join(scene_path, 'track_masks_pp.npy' if preprocess_tracks else 'track_masks.npy')
        
        tracks, track_masks = None, None
        recompute_tracks = overwrite or not (os.path.exists(tracks_path) and os.path.exists(track_masks_path))

        if recompute_tracks:
            print(f"  Computing tracks for {scene_name}...")
            tracks, track_masks = extract_tracks_for_scene(
                scene_path, preprocess_tracks, img_size, patch_size
            )
            if tracks is not None and track_masks is not None:
                np.save(tracks_path, tracks)
                np.save(track_masks_path, track_masks)
                print(f"  Saved tracks ({tracks.shape}) and track_masks ({track_masks.shape}) to {scene_path}")
        else:
            print(f"  Tracks already exist for {scene_name}, loading them.")
            tracks = np.load(tracks_path)
            track_masks = np.load(track_masks_path)

        if tracks is not None and track_masks is not None:
            if visualize:
                output_video_path = os.path.join(scene_path, 'tracks_visualization_2d_pp.mp4' if preprocess_tracks else 'tracks_visualization_2d.mp4')
                if overwrite or not os.path.exists(output_video_path):
                    print(f"  Generating 2D track visualization for {scene_name}...")
                    
                    rgba_files = sorted([f for f in os.listdir(scene_path) if f.startswith("rgba_") and f.endswith(".png")])
                    num_track_frames = tracks.shape[1]
                    
                    # Load video frames with proper error handling
                    video_frames = []
                    for f in rgba_files[:num_track_frames]:
                        frame = read_image_cv2(os.path.join(scene_path, f))
                        if frame is not None:
                            video_frames.append(frame)
                        else:
                            print(f"  Warning: Failed to load frame {f}, skipping visualization")
                            break
                    
                    # Only proceed if we have all frames
                    if len(video_frames) == num_track_frames:
                        if preprocess_tracks:
                            # Also preprocess the visualization frames to match tracks
                            aspect_ratio = video_frames[0].shape[1] / video_frames[0].shape[0]
                            short_size = int(img_size * aspect_ratio)
                            if short_size % patch_size != 0:
                                short_size = (short_size // patch_size) * patch_size
                            target_shape = np.array([short_size, img_size])
                            
                            resized_frames = []
                            for frame in video_frames:
                                 # Simplified preprocessing for viz: just resize and crop
                                resized = cv2.resize(frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LANCZOS4)
                                h, w = resized.shape[:2]
                                start_x = max(0, (w - target_shape[1]) // 2)
                                start_y = max(0, (h - target_shape[0]) // 2)
                                cropped = resized[start_y:start_y+target_shape[0], start_x:start_x+target_shape[1]]
                                resized_frames.append(cropped)
                            video_frames = resized_frames

                        video = np.stack(video_frames)
                        
                        points_for_plot = tracks.transpose(1, 0, 2)
                        visibles_for_plot = track_masks.transpose(1, 0)
                        
                        video_viz = plot_2d_tracks(video, points_for_plot, visibles_for_plot)
                        
                        with imageio.get_writer(output_video_path, format='FFMPEG', fps=10, codec='libx264') as writer:
                            for frame in video_viz:
                                writer.append_data(frame)

                        print(f"  Saved visualization to {output_video_path}")
                    else:
                        print(f"  Skipping visualization for {scene_name} due to missing frames")
                else:
                    print(f"  Visualization already exists for {scene_name}, skipping.")
        else:
            print(f"  Failed to extract tracks for {scene_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Kubric dataset to extract tracks.")
    parser.add_argument('--dataset_location', type=str, required=True,
                        help="Path to the Kubric dataset root (e.g., /path/to/output).") # /shared/ssd_14T/gaspard/output for ex
    parser.add_argument('--visualize', action='store_true', help="Generate 2D track visualizations.")
    parser.add_argument('--overwrite', action='store_true', help="Recompute tracks and visualizations even if they already exist.")
    parser.add_argument('--preprocess_tracks', action='store_true', help="Apply preprocessing (resizing/cropping) to the tracks.")
    parser.add_argument('--img_size', type=int, default=224, help="Target image size for preprocessing.")
    parser.add_argument('--patch_size', type=int, default=14, help="Patch size for calculating target shape.")

    args = parser.parse_args()
    
    main(args.dataset_location, args.visualize, args.overwrite, args.preprocess_tracks, args.img_size, args.patch_size) 