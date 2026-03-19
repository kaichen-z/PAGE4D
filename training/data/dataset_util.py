import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import pdb

def read_image_cv2(path: str, rgb: bool = True) -> np.ndarray:
    """
    Reads an image from disk using OpenCV, returning it as an RGB image array (H, W, 3).

    Args:
        path (str): File path to the image.
        rgb (bool): If True, convert the image to RGB.

    Returns:
        np.ndarray or None: A numpy array of shape (H, W, 3) if successful,
            or None if the file does not exist or could not be read.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"File does not exist or is empty: {path}")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image={path}. Retrying...")
        img = cv2.imread(path)
        if img is None:
            print("Retry failed.")
            return None

    if rgb:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def threshold_depth_map(
    depth_map: np.ndarray,
    max_percentile: float = 99,
    min_percentile: float = 1,
    max_depth: float = -1,
) -> np.ndarray:
    """
    Thresholds a depth map using percentile-based limits and optional maximum depth clamping.

    Args:
        depth_map (np.ndarray): Input depth map (H, W).
        max_percentile (float): Upper percentile (0-100). Values above this will be set to zero.
        min_percentile (float): Lower percentile (0-100). Values below this will be set to zero.
        max_depth (float): Absolute maximum depth. If > 0, any depth above this is set to zero.

    Returns:
        np.ndarray: Depth map (H, W) after thresholding.
    """
    if depth_map is None:
        return None

    depth_map = depth_map.astype(float, copy=True)

    # Optional clamp by max_depth
    if max_depth > 0:
        depth_map[depth_map > max_depth] = 0.0

    # Percentile-based thresholds
    depth_max_thres = (
        np.nanpercentile(depth_map, max_percentile) if max_percentile > 0 else None
    )
    depth_min_thres = (
        np.nanpercentile(depth_map, min_percentile) if min_percentile > 0 else None
    )

    # Apply the thresholds if they are > 0
    if depth_max_thres is not None and depth_max_thres > 0:
        depth_map[depth_map > depth_max_thres] = 0.0
    if depth_min_thres is not None and depth_min_thres > 0:
        depth_map[depth_map < depth_min_thres] = 0.0

    return depth_map


def crop_image_depth_and_intrinsic_by_pp(
    image, depth_map, intrinsic, target_size, track=None, filepath=None, strict=False
):
    """
    Crop image and depth map to target size, centered on principal point.
    Updates intrinsic matrix accordingly.
    
    Args:
        image (np.ndarray): Input image (H, W, 3)
        depth_map (np.ndarray): Input depth map (H, W)
        intrinsic (np.ndarray): Camera intrinsic matrix (3, 3)
        target_size (tuple): Target size (height, width)
        track (np.ndarray, optional): Track coordinates to update
        filepath (str, optional): File path for debugging
        strict (bool): If True, enforce exact target size
        
    Returns:
        tuple: (cropped_image, cropped_depth, updated_intrinsic, updated_track)
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Get principal point
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # Calculate crop bounds centered on principal point
    start_x = max(0, int(cx - target_w // 2))
    start_y = max(0, int(cy - target_h // 2))
    end_x = min(w, start_x + target_w)
    end_y = min(h, start_y + target_h)
    
    # Adjust if we hit boundaries
    if end_x - start_x < target_w:
        start_x = max(0, end_x - target_w)
    if end_y - start_y < target_h:
        start_y = max(0, end_y - target_h)
    
    # Crop image and depth
    cropped_image = image[start_y:end_y, start_x:end_x]
    cropped_depth = depth_map[start_y:end_y, start_x:end_x] if depth_map is not None else None
    
    # Update intrinsic matrix
    updated_intrinsic = intrinsic.copy()
    updated_intrinsic[0, 2] -= start_x  # cx
    updated_intrinsic[1, 2] -= start_y  # cy
    
    # Update track if provided
    updated_track = None
    track_mask = None
    if track is not None:
        updated_track = track.copy()
        updated_track[:, 0] -= start_x  # x coordinates
        updated_track[:, 1] -= start_y  # y coordinates

        Hc, Wc = cropped_image.shape[:2]
        inb = ((updated_track[:, 0] >= 0) & (updated_track[:, 0] < Wc) &
            (updated_track[:, 1] >= 0) & (updated_track[:, 1] < Hc))
        track_mask = inb  # 仅改这一行即可：融合原 mask 与越界检测
    return cropped_image, cropped_depth, updated_intrinsic, updated_track, track_mask

def sparse_depth_resize_splat_zmin(depth, target_h, target_w, dilate_radius=0):
    """
    稀疏深度重采样（前向撒点 + Z-buffer最小深度），不做平滑，不引入新值。
    depth: HxW，0/NaN/inf 视为无效
    dilate_radius: 每个落点在目标图上复制为 (2r+1)x(2r+1) 小块；r=0 表示只占1像素
    """
    H, W = depth.shape
    mask = np.isfinite(depth) & (depth > 0)
    if mask.sum() == 0:
        return np.zeros((target_h, target_w), dtype=np.float32)
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs].astype(np.float32)
    # 映射到目标分辨率（最近的整数像素）
    xs_new = np.clip((xs * (target_w / W)).round().astype(np.int32), 0, target_w - 1)
    ys_new = np.clip((ys * (target_h / H)).round().astype(np.int32), 0, target_h - 1)
    out = np.full((target_h, target_w), np.inf, dtype=np.float32)
    if dilate_radius <= 0:
        for x, y, z in zip(xs_new, ys_new, zs):
            if z < out[y, x]:
                out[y, x] = z
    else:
        r = int(dilate_radius)
        for x, y, z in zip(xs_new, ys_new, zs):
            y0, y1 = max(0, y - r), min(target_h, y + r + 1)
            x0, x1 = max(0, x - r), min(target_w, x + r + 1)
            # 用更近的深度覆盖（Z-buffer），不做平均
            patch = out[y0:y1, x0:x1]
            np.minimum(patch, z, out=patch)
    out[np.isinf(out)] = 0.0
    return out

def resize_image_depth_and_intrinsic(
    image, depth_map, intrinsic, target_shape, original_size, track=None,
    safe_bound=4, rescale_aug=True, resize_ours=False,
):
    """
    Resize image and depth map to target shape, updating intrinsic matrix.
    
    Args:
        image (np.ndarray): Input image
        depth_map (np.ndarray): Input depth map
        intrinsic (np.ndarray): Camera intrinsic matrix
        target_shape (tuple): Target shape (height, width)
        original_size (tuple): Original size (height, width)
        track (np.ndarray, optional): Track coordinates to update
        safe_bound (int): Safety margin for resizing
        rescale_aug (bool): Whether to apply rescaling augmentation
        
    Returns:
        tuple: (resized_image, resized_depth, updated_intrinsic, updated_track)
    """
    target_h, target_w = target_shape
    orig_h, orig_w = original_size
    
    # Add safe bound for augmentation
    if rescale_aug:
        target_h += safe_bound
        target_w += safe_bound
    
    # Calculate scaling factors
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w
    
    # Resize image
    resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Resize depth map if available
    resized_depth = None
    if depth_map is not None:
        if not resize_ours:
            resized_depth = cv2.resize(depth_map, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            resized_depth = sparse_depth_resize_splat_zmin(depth_map, target_h, target_w)
    
    # Update intrinsic matrix
    updated_intrinsic = intrinsic.copy()
    updated_intrinsic[0, 0] *= scale_w  # fx
    updated_intrinsic[1, 1] *= scale_h  # fy
    updated_intrinsic[0, 2] *= scale_w  # cx
    updated_intrinsic[1, 2] *= scale_h  # cy
    
    # Update track if provided
    updated_track = None
    if track is not None:
        updated_track = track.copy()
        updated_track[:, 0] *= scale_w  # x coordinates
        updated_track[:, 1] *= scale_h  # y coordinates
    
    return resized_image, resized_depth, updated_intrinsic, updated_track


def depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv, eps=1e-8):
    """
    Convert depth map to world coordinates and camera coordinates.
    Enhanced version with better error handling and coordinate transformations.
    Args:
        depth_map (np.ndarray): Depth map (H, W)
        extri_opencv (np.ndarray): Extrinsic matrix (3, 4) in (world to camera).
        intri_opencv (np.ndarray): Intrinsic matrix (3, 3)
        eps (float): Small epsilon for thresholding valid depth
    Returns:
        tuple: (world_coords_points, cam_coords_points, point_mask)
    """
    if depth_map is None:
        h, w = 224, 224  # Default size
        return np.zeros((h, w, 3)), np.zeros((h, w, 3)), np.zeros((h, w), dtype=bool)
    # Valid depth mask
    point_mask = (depth_map > eps) & np.isfinite(depth_map)
    # Convert depth map to camera coordinates using the new function
    cam_coords_points = depth_to_cam_coords_points(depth_map, intri_opencv)
    
    # Handle extrinsic matrix format - convert to 4x4 if needed
    if extri_opencv.shape == (3, 4):
        # Convert (3,4) to (4,4) format
        extrinsic_4x4 = np.eye(4)
        extrinsic_4x4[:3, :] = extri_opencv
    else:
        extrinsic_4x4 = extri_opencv
    
    # The extrinsic is world to camera, so invert it to transform camera->world
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic_4x4[None])[0]
    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]
    
    # Apply the rotation and translation to the camera coordinates
    h, w = depth_map.shape
    cam_coords_reshaped = cam_coords_points.reshape(-1, 3)
    world_coords_reshaped = (cam_coords_reshaped @ R_cam_to_world.T) + t_cam_to_world
    world_coords_points = world_coords_reshaped.reshape(h, w, 3)

    return world_coords_points, cam_coords_points, point_mask


def rotate_90_degrees(
    image, depth_map, extri_opencv, intri_opencv, clockwise=True, track=None
):
    """
    Rotate image, depth map, and update camera parameters by 90 degrees.
    
    Args:
        image (np.ndarray): Input image
        depth_map (np.ndarray): Input depth map
        extri_opencv (np.ndarray): Extrinsic matrix
        intri_opencv (np.ndarray): Intrinsic matrix
        clockwise (bool): Rotation direction
        track (np.ndarray, optional): Track coordinates to update
        
    Returns:
        tuple: All inputs rotated/updated accordingly
    """
    h, w = image.shape[:2]
    
    # Rotate image
    if clockwise:
        rotated_image = np.transpose(image, (1, 0, 2))
        rotated_image = np.flip(rotated_image, axis=1)
    else:
        rotated_image = np.transpose(image, (1, 0, 2))
        rotated_image = np.flip(rotated_image, axis=0)
    
    # Rotate depth map
    rotated_depth = None
    if depth_map is not None:
        if clockwise:
            rotated_depth = np.transpose(depth_map, (1, 0))
            rotated_depth = np.flip(rotated_depth, axis=1)
        else:
            rotated_depth = np.transpose(depth_map, (1, 0))
            rotated_depth = np.flip(rotated_depth, axis=0)
    
    # Update intrinsic matrix for rotation
    updated_intrinsic = intri_opencv.copy()
    if clockwise:
        # Swap fx/fy and update principal point
        fx, fy = intri_opencv[0, 0], intri_opencv[1, 1]
        cx, cy = intri_opencv[0, 2], intri_opencv[1, 2]
        updated_intrinsic[0, 0] = fy
        updated_intrinsic[1, 1] = fx
        updated_intrinsic[0, 2] = cy
        updated_intrinsic[1, 2] = w - 1 - cx
    else:
        fx, fy = intri_opencv[0, 0], intri_opencv[1, 1]
        cx, cy = intri_opencv[0, 2], intri_opencv[1, 2]
        updated_intrinsic[0, 0] = fy
        updated_intrinsic[1, 1] = fx
        updated_intrinsic[0, 2] = h - 1 - cy
        updated_intrinsic[1, 2] = cx
    
    # Update track if provided
    updated_track = None
    if track is not None:
        updated_track = track.copy()
        if clockwise:
            # (x, y) -> (y, w - 1 - x)
            new_x = track[:, 1]
            new_y = w - 1 - track[:, 0]
            updated_track = np.stack([new_x, new_y], axis=1)
        else:
            # (x, y) -> (h - 1 - y, x)
            new_x = h - 1 - track[:, 1]
            new_y = track[:, 0]
            updated_track = np.stack([new_x, new_y], axis=1)
    
    return rotated_image, rotated_depth, extri_opencv, updated_intrinsic, updated_track 

def get_stride_distribution(strides, dist_type='uniform'):

    # input strides sorted by descreasing order by default
    
    if dist_type == 'uniform':
        dist = np.ones(len(strides)) / len(strides)
    elif dist_type == 'exponential':
        lambda_param = 1.0
        dist = np.exp(-lambda_param * np.arange(len(strides)))
    elif dist_type.startswith('linear'): # e.g., linear_1_2
        try:
            start, end = map(float, dist_type.split('_')[1:])
            dist = np.linspace(start, end, len(strides))
        except ValueError:
            raise ValueError(f'Invalid linear distribution format: {dist_type}')
    else:
        raise ValueError('Unknown distribution type %s' % dist_type)
    # normalize to sum to 1
    return dist / np.sum(dist)

def read_depth(path: str, scale_adjustment=1.0) -> np.ndarray:
    """
    Reads a depth map from disk in either .exr or .png format. The .exr is loaded using OpenCV
    with the environment variable OPENCV_IO_ENABLE_OPENEXR=1. The .png is assumed to be a 16-bit
    PNG (converted from half float).
    Args:
        path (str):
            File path to the depth image. Must end with .exr or .png.
        scale_adjustment (float):
            A multiplier for adjusting the loaded depth values (default=1.0).
    Returns:
        np.ndarray:
            A float32 array (H, W) containing the loaded depth. Zeros or non-finite values
            may indicate invalid regions.
    Raises:
        ValueError:
            If the file extension is not supported.
    """
    if path.lower().endswith(".exr"):
        # Ensure OPENCV_IO_ENABLE_OPENEXR is set to "1"
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
        d[d > 1e9] = 0.0
    elif path.lower().endswith(".png"):
        d = load_16big_png_depth(path)
    else:
        raise ValueError(f'unsupported depth file name "{path}"')
    d = d * scale_adjustment
    d[~np.isfinite(d)] = 0.0
    return d

def load_16big_png_depth(depth_png: str) -> np.ndarray:
    """
    Loads a 16-bit PNG as a half-float depth map (H, W), returning a float32 NumPy array.

    Implementation detail:
      - PIL loads 16-bit data as 32-bit "I" mode.
      - We reinterpret the bits as float16, then cast to float32.

    Args:
        depth_png (str):
            File path to the 16-bit PNG.

    Returns:
        np.ndarray:
            A float32 depth array of shape (H, W).
    """
    from PIL import Image
    with Image.open(depth_png) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def depth_to_cam_coords_points(
    depth_map: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """
    Unprojects a depth map into camera coordinates, returning (H, W, 3).

    Args:
        depth_map (np.ndarray):
            Depth map of shape (H, W).
        intrinsic (np.ndarray):
            3x3 camera intrinsic matrix.
            Assumes zero skew and standard OpenCV layout:
            [ fx   0   cx ]
            [  0  fy   cy ]
            [  0   0    1 ]

    Returns:
        np.ndarray:
            An (H, W, 3) array, where each pixel is mapped to (x, y, z) in the camera frame.
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert (
        intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0
    ), "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)


def closed_form_inverse_se3(se3_matrices: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of SE(3) matrices using closed-form solution.
    
    For SE(3) matrix [R t; 0 1], the inverse is [R^T -R^T*t; 0 1]
    
    Args:
        se3_matrices: Array of shape (N, 4, 4) containing SE(3) matrices
        
    Returns:
        Array of shape (N, 4, 4) containing inverted SE(3) matrices
    """
    inverted = np.zeros_like(se3_matrices)
    
    for i in range(se3_matrices.shape[0]):
        R = se3_matrices[i, :3, :3]
        t = se3_matrices[i, :3, 3]
        
        R_inv = R.T
        t_inv = -R_inv @ t
        
        inverted[i, :3, :3] = R_inv
        inverted[i, :3, 3] = t_inv
        inverted[i, 3, 3] = 1.0
        
    return inverted


def create_tracking_grid(image_shape, grid_size=None, num_points=None):
    """
    Create a uniform grid of 2D tracking points.
    
    Args:
        image_shape (tuple): Image shape (H, W)
        grid_size (tuple, optional): Grid dimensions (grid_h, grid_w)
        num_points (int, optional): Target number of points (if grid_size not specified)
        
    Returns:
        np.ndarray: Array of shape (N, 2) with 2D points (x, y)
    """
    h, w = image_shape[:2]
    
    if grid_size is None:
        if num_points is None:
            num_points = 1024  # Default
        # Create approximately square grid
        aspect_ratio = w / h
        grid_h = int(np.sqrt(num_points / aspect_ratio))
        grid_w = int(grid_h * aspect_ratio)
    else:
        grid_h, grid_w = grid_size
    
    # Create grid coordinates
    y_coords = np.linspace(0, h-1, grid_h)
    x_coords = np.linspace(0, w-1, grid_w)
    
    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    
    return points.astype(np.float32)


def apply_flow_transforms_simple(flow, crop_start_x, crop_start_y, crop_end_x, crop_end_y, 
                                 scale_w, scale_h, target_w, target_h):
    """
    Simplified flow transformation that directly applies crop and resize.
    
    Args:
        flow: Original flow (H, W, 2)
        crop_start_x, crop_start_y: Start coordinates for crop
        crop_end_x, crop_end_y: End coordinates for crop
        scale_w, scale_h: Scaling factors for flow vectors
        target_w, target_h: Final target dimensions
        
    Returns:
        Transformed flow
    """
    if flow is None:
        return None
    
    # Apply crop
    transformed_flow = flow[crop_start_y:crop_end_y, crop_start_x:crop_end_x].copy()
    
    # Resize to target dimensions
    if transformed_flow.shape[:2] != (target_h, target_w):
        transformed_flow = cv2.resize(transformed_flow, (target_w, target_h), 
                                     interpolation=cv2.INTER_LINEAR)
    
    # Scale flow vectors
    transformed_flow[:, :, 0] *= scale_w
    transformed_flow[:, :, 1] *= scale_h
    
    return transformed_flow



def apply_mask_transforms_simple(mask, crop_start_x, crop_start_y, crop_end_x, crop_end_y, 
                                target_w, target_h):
    """
    Apply the same transformations to masks as to optical flow.
    
    Args:
        mask: Original mask (H, W)
        crop_start_x, crop_start_y: Start coordinates for crop
        crop_end_x, crop_end_y: End coordinates for crop
        target_w, target_h: Final target dimensions
        
    Returns:
        Transformed mask
    """
    if mask is None:
        return None
    
    # Apply crop
    transformed_mask = mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x].copy()
    
    # Resize to target dimensions
    if transformed_mask.shape[:2] != (target_h, target_w):
        # Use nearest neighbor interpolation for masks to preserve binary values
        transformed_mask = cv2.resize(transformed_mask.astype(np.uint8), (target_w, target_h), 
                                     interpolation=cv2.INTER_NEAREST)
        transformed_mask = transformed_mask.astype(bool)
    
    return transformed_mask


def track_points_with_flow_and_occlusion(
    points, flow, image_shape, depth_map=None, occlusion_mask=None, invalid_mask=None
):
    """
    Track points using optical flow and compute visibility with occlusion information.
    Args:
        points (np.ndarray): Points to track, shape (N, 2) with (x, y) coordinates
        flow (np.ndarray): Optical flow field, shape (H, W, 2)
        image_shape (tuple): Target image shape (H, W)
        depth_map (np.ndarray, optional): Depth map for visibility checking
        occlusion_mask (np.ndarray, optional): Binary mask of occluded regions (H, W)
        invalid_mask (np.ndarray, optional): Binary mask of invalid regions (H, W)
        
    Returns:
        tuple: (tracked_points, visibility_mask)
            - tracked_points: New point positions (N, 2)
            - visibility_mask: Boolean mask indicating point visibility (N,)
    """
    if flow is None:
        # No flow available, return original points with all visible
        visibility = np.ones(len(points), dtype=bool)
        return points.copy(), visibility
    
    h, w = image_shape[:2]
    flow_h, flow_w = flow.shape[:2]
    
    # Sample flow at point locations using bilinear interpolation
    x_coords = np.clip(points[:, 0], 0, flow_w-1)
    y_coords = np.clip(points[:, 1], 0, flow_h-1)
    
    # Ensure indices are within flow bounds
    x_coords = np.clip(x_coords, 0, flow_w-1)
    y_coords = np.clip(y_coords, 0, flow_h-1)
    
    # Bilinear interpolation with bounds checking
    x0 = np.floor(x_coords).astype(int)
    x1 = np.minimum(x0 + 1, flow_w-1)
    y0 = np.floor(y_coords).astype(int)
    y1 = np.minimum(y0 + 1, flow_h-1)
    
    # Ensure all indices are valid
    x0 = np.clip(x0, 0, flow_w-1)
    x1 = np.clip(x1, 0, flow_w-1)
    y0 = np.clip(y0, 0, flow_h-1)
    y1 = np.clip(y1, 0, flow_h-1)
    
    # Interpolation weights
    wx = x_coords - x0
    wy = y_coords - y0
    
    # Sample flow at four corners with bounds checking
    try:
        flow_00 = flow[y0, x0]  # top-left
        flow_01 = flow[y1, x0]  # bottom-left
        flow_10 = flow[y0, x1]  # top-right
        flow_11 = flow[y1, x1]  # bottom-right
    except IndexError as e:
        print(f"Index error in flow sampling: {e}")
        print(f"Flow shape: {flow.shape}, points range: x=[{x_coords.min():.1f}, {x_coords.max():.1f}], y=[{y_coords.min():.1f}, {y_coords.max():.1f}]")
        print(f"Index ranges: x0=[{x0.min()}, {x0.max()}], y0=[{y0.min()}, {y0.max()}]")
        # Fallback: no movement
        flow_interp = np.zeros((len(points), 2))
    else:
        # Bilinear interpolation
        flow_interp = (
            flow_00 * (1 - wx)[:, None] * (1 - wy)[:, None] +
            flow_01 * (1 - wx)[:, None] * wy[:, None] +
            flow_10 * wx[:, None] * (1 - wy)[:, None] +
            flow_11 * wx[:, None] * wy[:, None]
        )
    
    # Apply flow to get new positions
    tracked_points = points + flow_interp
    
    # Start with basic visibility check (within image bounds)
    visibility = (
        (tracked_points[:, 0] >= 0) &
        (tracked_points[:, 0] < w) &
        (tracked_points[:, 1] >= 0) &
        (tracked_points[:, 1] < h)
    )
    
    # Check occlusion at original point locations
    if occlusion_mask is not None:
        # Points that are currently in occluded regions should become invisible
        original_x = np.clip(points[:, 0], 0, min(w, occlusion_mask.shape[1])-1).astype(int)
        original_y = np.clip(points[:, 1], 0, min(h, occlusion_mask.shape[0])-1).astype(int)
        
        # Check if original points are in occluded regions
        not_occluded = ~occlusion_mask[original_y, original_x]
        visibility = visibility & not_occluded
    
    # Check invalid regions at original point locations  
    if invalid_mask is not None:
        original_x = np.clip(points[:, 0], 0, min(w, invalid_mask.shape[1])-1).astype(int)
        original_y = np.clip(points[:, 1], 0, min(h, invalid_mask.shape[0])-1).astype(int)
        
        # Check if original points are in invalid regions
        not_invalid = ~invalid_mask[original_y, original_x]
        visibility = visibility & not_invalid
    
    # Additional check: if tracked points land in occluded or invalid regions
    valid_tracked_indices = visibility  # Only check points that are still visible
    if np.any(valid_tracked_indices):
        tracked_valid_points = tracked_points[valid_tracked_indices]
        
        # Check occlusion at tracked locations
        if occlusion_mask is not None:
            tx = np.clip(tracked_valid_points[:, 0], 0, min(w, occlusion_mask.shape[1])-1).astype(int)
            ty = np.clip(tracked_valid_points[:, 1], 0, min(h, occlusion_mask.shape[0])-1).astype(int)
            tracked_not_occluded = ~occlusion_mask[ty, tx]
            
            # Update visibility for the subset of valid points
            visibility_subset = visibility[valid_tracked_indices]
            visibility_subset = visibility_subset & tracked_not_occluded
            visibility[valid_tracked_indices] = visibility_subset
        
        # Check invalid regions at tracked locations
        if invalid_mask is not None:
            # Recalculate valid indices after occlusion check
            valid_tracked_indices = visibility
            if np.any(valid_tracked_indices):
                tracked_valid_points = tracked_points[valid_tracked_indices]
                tx = np.clip(tracked_valid_points[:, 0], 0, min(w, invalid_mask.shape[1])-1).astype(int)
                ty = np.clip(tracked_valid_points[:, 1], 0, min(h, invalid_mask.shape[0])-1).astype(int)
                tracked_not_invalid = ~invalid_mask[ty, tx]
                
                visibility_subset = visibility[valid_tracked_indices]
                visibility_subset = visibility_subset & tracked_not_invalid
                visibility[valid_tracked_indices] = visibility_subset
    
    # Optional depth check (more lenient than before)
    if depth_map is not None:
        valid_tracked_indices = visibility
        if np.any(valid_tracked_indices):
            tracked_valid_points = tracked_points[valid_tracked_indices]
            tx = np.clip(tracked_valid_points[:, 0], 0, min(w, depth_map.shape[1])-1).astype(int)
            ty = np.clip(tracked_valid_points[:, 1], 0, min(h, depth_map.shape[0])-1).astype(int)
            
            # More lenient depth check: accept very small positive values too
            valid_depth = depth_map[ty, tx] > -1.0  # Accept depth close to 0 as valid
            
            visibility_subset = visibility[valid_tracked_indices]
            visibility_subset = visibility_subset & valid_depth
            visibility[valid_tracked_indices] = visibility_subset
    
    return tracked_points, visibility