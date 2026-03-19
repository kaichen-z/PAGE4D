import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
from pathlib import Path

# Add training/ to path so data.* imports work from new location (training/data/dataset_validation.py)
_TRAINING_DIR = Path(__file__).resolve().parent.parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

import argparse
import torch
import torch.distributed as dist
import numpy as np
import open3d as o3d
from hydra import initialize, compose
from hydra.utils import instantiate
from PIL import Image
import pdb
# python3 -m data.dataset_validation --config debug  (run from training/) 
import cv2
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
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

def _to_bgr8(img):
    if img is None:
        raise ValueError("Frame is None")
    arr = np.asarray(img)

    # 通道统一到3通道
    if arr.ndim == 2:  # 灰度
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:  # RGBA -> RGB
        arr = arr[..., :3]

    # dtype 统一到 uint8
    if arr.dtype == np.uint8:
        pass
    elif arr.dtype == np.uint16:
        arr = (np.clip(arr, 0, 65535) / 257.0).astype(np.uint8)
    else:
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        m = float(np.max(arr)) if arr.size else 1.0
        if m <= 1.0 + 1e-6:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(arr)

def export_tracks_side_by_side(images_any, tracks_T_N_2, masks_T_N,
                               out_path="tracks_first_last.png",
                               draw_ids=True, radius=3, thickness=2,
                               scale=4, num=32):
    T, N, _ = tracks_T_N_2.shape
    assert len(images_any) >= T and T >= 2, "at least two frames are required"
    img0 = _to_bgr8(images_any[0])
    img1 = _to_bgr8(images_any[T-1])
    H, W = img0.shape[:2]
    # 图像 resize
    Hs, Ws = H * scale, W * scale
    img0 = cv2.resize(img0, (Ws, Hs), interpolation=cv2.INTER_CUBIC)
    img1 = cv2.resize(img1, (Ws, Hs), interpolation=cv2.INTER_CUBIC)
    # 拼接画布
    canvas = np.zeros((Hs, Ws * 2, 3), dtype=np.uint8)
    canvas[:, 0:Ws] = img0
    canvas[:, Ws:Ws * 2] = img1
    # 点坐标也放大
    pts0 = tracks_T_N_2[0, :] * scale
    pts1 = tracks_T_N_2[T - 1, :] * scale
    valid0 = masks_T_N[0, :]
    valid1 = masks_T_N[T - 1, :]
    valid = valid0 & valid1
    col0 = (0, 200, 255)   # 第1帧点：橙黄
    col1 = (0, 255, 0)     # 最后一帧点：绿
    colL = (255, 255, 255) # 连接线：白
    selected = random.sample(range(N), num)  # 从 0..N-1 中选 num 个不重复的
    for i in selected:
        if not valid[i]:
            continue
        x0, y0 = pts0[i]
        x1, y1 = pts1[i]
        x0i, y0i = int(round(x0)), int(round(y0))
        x1i, y1i = int(round(x1)), int(round(y1))
        # 起点
        cv2.circle(canvas, (x0i, y0i), radius, col0, -1, cv2.LINE_AA)
        # 终点（加偏移 Ws）
        cv2.circle(canvas, (x1i + Ws, y1i), radius, col1, -1, cv2.LINE_AA)
        # 画线
        cv2.line(canvas, (x0i, y0i), (x1i + Ws, y1i), colL, thickness, cv2.LINE_AA)
        if draw_ids and i % max(1, N // 50) == 0:
            cv2.putText(canvas, str(i), (x0i + 4, y0i - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(i), (x0i + 4, y0i - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(i), (x1i + Ws + 4, y1i - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(i), (x1i + Ws + 4, y1i - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(out_path, canvas)
    if not ok:
        raise RuntimeError(f"Failed to write image to {out_path}")
    print(f"[OK] Wrote side-by-side track snapshot (scaled x{scale}): {out_path}")

def save_ply(points, colors, filename, point_mask=None):
    """
    Save point cloud to .ply with optional mask.
    Args:
        points: (N,3) torch.Tensor or np.ndarray
        colors: (N,3) torch.Tensor or np.ndarray
        filename: str
        point_mask: (N,) bool array or tensor, optional
    """
    # Convert to numpy
    if torch.is_tensor(points):
        points_visual = points.reshape(-1, 3).cpu().numpy()
    else:
        points_visual = points.reshape(-1, 3)
    if torch.is_tensor(colors):
        points_visual_rgb = colors.reshape(-1, 3).cpu().numpy()
    else:
        points_visual_rgb = colors.reshape(-1, 3)
    # Apply mask if provided
    if point_mask is not None:
        if torch.is_tensor(point_mask):
            point_mask = point_mask.cpu().numpy().astype(bool).reshape(-1)
        else:
            point_mask = np.asarray(point_mask, dtype=bool).reshape(-1)
        points_visual = points_visual[point_mask]
        points_visual_rgb = points_visual_rgb[point_mask]
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

def save_image(image_tensor, filename):
    if torch.is_tensor(image_tensor):
        # Convert from tensor to numpy
        image_np = image_tensor.cpu().numpy()
    else:
        image_np = image_tensor
    
    # Handle different tensor formats
    if len(image_np.shape) == 4:  # (batch, channels, height, width)
        image_np = image_np[0]  # Take first batch
    
    if len(image_np.shape) == 3 and image_np.shape[0] <= 4:  # (channels, height, width)
        image_np = np.transpose(image_np, (1, 2, 0))  # Convert to (height, width, channels)
    
    # Normalize to 0-255 range if needed
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    
    # Handle grayscale or RGB
    if image_np.shape[2] == 1:
        image_np = image_np.squeeze(2)  # Remove channel dimension for grayscale
    elif image_np.shape[2] > 3:
        image_np = image_np[:, :, :3]  # Take first 3 channels for RGB
    
    # Save as PNG
    Image.fromarray(image_np).save(filename)


def plot_tracks_on_images(images, tracks, masks):
    """
    images: (T, H, W, 3)
    tracks: (N, T, 2)
    masks:  (N, T) bool
    """
    N, T, _ = tracks.shape
    vis_frames = []
    colors = plt.cm.hsv(np.linspace(0, 1, N))

    for t in range(T):
        img = images[t].copy()
        for n in range(N):
            if masks[n, t]:
                x, y = tracks[n, t]
                if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]:
                    cv2.circle(img, (int(x), int(y)), 1, (255*colors[n, :3]).astype(np.uint8).tolist(), -1)
        vis_frames.append(img)
    return vis_frames

def save_video_opencv(frames, out_path, fps=10):
    # frames: list or np.ndarray, shape = (N, H, W, 3), dtype=uint8
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 保存 mp4
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # OpenCV 用 BGR
    writer.release()
    print(f"Saved video to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    args = parser.parse_args()
    # Initialize distributed process group for single-process use
    # This is required by DynamicDistributedSampler
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29521')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', 
                                rank=0, world_size=1)

    _CONFIG_DIR = str(Path(__file__).resolve().parent / "datasets" / "config")
    with initialize(version_base=None, config_path=_CONFIG_DIR):
        cfg = compose(config_name=args.config)
    # Override num_workers to 0 to avoid multi-processing issues in validation
    cfg.data.train.num_workers = 0
    
    # Enable frame caching for faster validation
    if hasattr(cfg.data.train, 'cache_frames_in_memory'):
        cfg.data.train.cache_frames_in_memory = True
    if hasattr(cfg.data.train, 'use_lidar_depth'):
        cfg.data.train.use_lidar_depth = False  # Disable LiDAR depth for faster validation
    
    # Force small sequences for validation to speed up
    if hasattr(cfg.data.train, 'img_nums'):
        cfg.data.train.img_nums = [2, 4]  # Force small sequences
    if hasattr(cfg.data.train, 'aspects'):
        cfg.data.train.aspects = [1.0, 1.0]  # Force square aspect ratio
    
    dataset = instantiate(cfg.data.train, _recursive_=False)
    dataset.seed = 47
    
    print("Successfully instantiated dataset")

    dataloader = dataset.get_loader(epoch=0)

    print("Successfully loaded dataloader")
    print("Starting validation loop...")
    
    # Debug: Check what the first batch looks like
    print("Debug: Getting first batch...")

    # first_batch = next(iter(dataloader))
    # print(f"First batch type: {type(first_batch)}")
    # print(f"First batch length: {len(first_batch)}")
    # if len(first_batch) > 0:
    #     print(f"First item type: {type(first_batch[0])}")
    #     print(f"First item: {first_batch[0]}")
    first_batch = next(iter(dataloader))
    print(f"First batch type: {type(first_batch)}")
    print(f"First batch length: {len(first_batch)}")
    # Create data_visualization directory if it doesn't exist
    save_address = "/workspace/code/12_4d/VGGT-4D_T/training/data_visualization2/"
    os.makedirs(save_address, exist_ok=True)

    import time
    total_time = 0
    
    # Reset dataloader for the actual loop
    dataloader = dataset.get_loader(epoch=0)
    start_time = time.time()
    total_time = start_time
    save_all = True
    for data_iter, batch in enumerate(tqdm(dataloader)):
        seq_name = batch["seq_name"][0]
        if save_all:
            directory = os.path.dirname(f"{save_address}{seq_name}_{data_iter}_1.ply")
            os.makedirs(directory, exist_ok=True)
            save_ply(
                batch["world_points"][0, 0].reshape(-1, 3), 
                batch["images"][0, 0].permute(1, 2, 0).reshape(-1, 3), 
                f"{save_address}{seq_name}_{data_iter}_1.ply",
                batch["point_masks"][0, 0].reshape(-1))  
            save_ply(
                batch["world_points"][0, 1].reshape(-1, 3), 
                batch["images"][0, 1].permute(1, 2, 0).reshape(-1, 3), 
                f"{save_address}{seq_name}_{data_iter}_2.ply",
                batch["point_masks"][0, -1].reshape(-1)) 
            save_ply(
                batch["world_points"][0, -1].reshape(-1, 3), 
                batch["images"][0, -1].permute(1, 2, 0).reshape(-1, 3), 
                f"{save_address}{seq_name}_{data_iter}_last.ply",
                batch["point_masks"][0, -1].reshape(-1)) 
            save_ply(
                batch["cam_points"][0, -1].reshape(-1, 3), 
                batch["images"][0, -1].permute(1, 2, 0).reshape(-1, 3), 
                f"{save_address}{seq_name}_{data_iter}_camp_2.ply",
                batch["point_masks"][0, -1].reshape(-1)) 
        batch_time = time.time() - total_time
        total_time += batch_time
        debug_out = os.path.join(f"{save_address}", f"{seq_name}_tracks_first_vs_last.png")
        images_bgr = []
        images_np = []
        for img in batch["images"][0]:  # img shape: (C, H, W)
            img_np = img.permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C)
            images_bgr.append(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            images_np.append(img_np)
        if save_all:
            export_tracks_side_by_side(
                images_bgr, batch["tracks"][0].cpu().numpy(), batch["track_vis_mask"][0].cpu().numpy(),
                out_path=debug_out,
                draw_ids=True, radius=2, thickness=1)
            save_depth_as_rgb(batch["depths"][0,0].cpu().numpy(), f"{save_address}{seq_name}_{data_iter}_depth.png")
        if save_all:
            images_np = np.stack(images_np, axis=0)  # (T,H,W,3)
            print(images_np.shape, '==============images_np==============', batch["tracks"][0].cpu().numpy().shape, batch["track_vis_mask"][0].cpu().numpy().shape)
            vis_frames = plot_tracks_on_images(images_np*255., batch["tracks"][0].permute(1, 0, 2).cpu().numpy(), batch["track_vis_mask"][0].permute(1, 0).cpu().numpy())
            save_video_opencv(vis_frames, f"{save_address}{seq_name}_{data_iter}_tracks.mp4")
        pdb.set_trace()

    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average time per batch: {total_time/4:.2f}s")
    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()