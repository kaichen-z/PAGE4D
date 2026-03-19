import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MODEL_DIR = _PROJECT_ROOT / "model"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))
import math
import cv2
import numpy as np
import torch
import argparse

from metadata import dataset_metadata
from utils import *

from accelerate import PartialState

from tqdm import tqdm
import torch.nn.functional as F

def add_path_to_dust3r(ckpt):
    HERE_PATH = os.path.dirname(os.path.abspath(ckpt))
    # workaround for sibling import
    sys.path.insert(0, HERE_PATH)

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )
    parser.add_argument("--num_mask", type=int, default=0, help="number of mask")
    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument("--size", type=int, default="224")

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )

    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--freeze_state", action="store_true", default=False)
    parser.add_argument("--solve_pose", action="store_true", default=False)
    return parser


def eval_pose_estimation(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path)
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def pointmap_to_depth(point_map: torch.Tensor) -> torch.Tensor:
    """
    将点云 (point_map) 转换为深度图 (depth_map2)
    Args:
        point_map: torch.Tensor, shape (B, H, W, 3) 或 (B, 3, H, W)，
                   表示相机坐标系下的 3D 点 (X,Y,Z)
    Returns:
        depth_map2: torch.Tensor, shape (B, 1, H, W)，表示深度图
    """
    if point_map.ndim == 4 and point_map.shape[-1] == 3:
        # (B, H, W, 3)
        depth_map2 = point_map[..., 2]  # 取 Z
        depth_map2 = depth_map2.unsqueeze(1)  # (B,1,H,W)
    elif point_map.ndim == 4 and point_map.shape[1] == 3:
        # (B, 3, H, W)
        depth_map2 = point_map[:, 2:3, :, :]  # 取 Z 通道
    else:
        raise ValueError(f"Unexpected point_map shape {point_map.shape}")
    return depth_map2

def vggt_inference_single(images: torch.Tensor, model: torch.nn.Module, device="cuda"):
    """
    单窗推理：images 形状 (S, 3, H, W)
    返回 dict：{"images","extrinsic","intrinsic","depth_map","depth_conf"}
    """
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    images = images.to(device)
    model = model.to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            window = images.unsqueeze(0)  # (1, S, 3, H, W)
            aggregated_tokens_list, ps_idx = model.aggregator(window)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, window.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, window, ps_idx)
        point_map, point_conf = model.point_head(aggregated_tokens_list, window, ps_idx)
        point_depth_map = pointmap_to_depth(point_map[0])
        bottom = torch.tensor([0, 0, 0, 1], dtype=extrinsic.dtype, device=extrinsic.device)
        bottom = bottom.view(1, 1, 1, 4).expand(extrinsic.shape[0], extrinsic.shape[1], -1, -1)
        extrinsic = torch.cat([extrinsic, bottom], dim=2)  # [B, S, 4, 4]
    depth_map[depth_map < 0] = 0
    return {
        "images": images,              # (S, 3, H, W)
        "extrinsic": extrinsic[0],     # (S, 4, 4)
        "intrinsic": intrinsic[0],     # (S, ...)
        "depth_map": depth_map[0],     # (S, 1, H, W) 或 (S, H, W)
        "depth_conf": depth_conf[0],   # (S, 1, H, W) 或 (S, H, W)
    }

def eval_pose_estimation_dist(args, model, img_path, save_dir=None, mask_path=None):

    metadata = dataset_metadata.get(args.eval_dataset)
    anno_path = metadata.get("anno_path", None)
    if args.eval_dataset == "dyncheck":
        args.pose_eval_stride = 10
    # args.pose_eval_stride = 1
    print(f"args.pose_eval_stride: {args.pose_eval_stride}")
    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    with distributed_state.split_between_processes(seq_list) as seqs:
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        load_img_size = args.size
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        bug = False
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get(
                    "mask_path_seq_func", lambda mask_path, seq: None
                )
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                filelist = [
                    os.path.join(dir_path, name) for name in os.listdir(dir_path)
                ]
                filelist.sort()
                filelist = filelist[:: args.pose_eval_stride]

                load_img_size = 518
                views = prepare_input(
                    filelist,
                    size=load_img_size,
                    crop=not args.no_crop,)

                outputs = vggt_inference_single(views, model, device)
                
                # Clear cache after inference to free memory
                torch.cuda.empty_cache()
                (
                    colors,
                    pts3ds_self,
                    pts3ds_other,
                    conf_self,
                    conf_other,
                    cam_dict,
                    pr_poses,
                ) = prepare_output(outputs)
                
                # Clear cache after prepare_output
                torch.cuda.empty_cache()

                pred_traj = get_tum_poses(pr_poses)
                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_tum_poses(pr_poses, f"{save_dir}/{seq}/pred_traj.txt")
                save_focals(cam_dict, f"{save_dir}/{seq}/pred_focal.txt")
                save_intrinsics(cam_dict, f"{save_dir}/{seq}/pred_intrinsics.txt")
                # save_depth_maps(pts3ds_self,f'{save_dir}/{seq}', conf_self=conf_self)
                # save_conf_maps(conf_self,f'{save_dir}/{seq}')
                # save_rgb_imgs(colors,f'{save_dir}/{seq}')

                gt_traj_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                traj_format = metadata.get("traj_format", None)

                if args.eval_dataset == "sintel":
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file, stride=args.pose_eval_stride
                    )
                elif traj_format is not None:
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file,
                        traj_format=traj_format,
                        stride=args.pose_eval_stride,
                    )
                else:
                    gt_traj = None

                if gt_traj is not None:
                    ate, rpe_trans, rpe_rot = eval_metrics(
                        pred_traj,
                        gt_traj,
                        seq=seq,
                        filename=f"{save_dir}/{seq}_eval_metric.txt",
                    )
                    plot_trajectory(
                        pred_traj, gt_traj, title=seq, filename=f"{save_dir}/{seq}.png"
                    )
                else:
                    ate, rpe_trans, rpe_rot = 0, 0, 0
                    bug = True

                ate_list.append(ate)
                rpe_trans_list.append(rpe_trans)
                rpe_rot_list.append(rpe_rot)

                # Write to error log after each sequence
                with open(error_log_path, "a") as f:
                    f.write(
                        f"{args.eval_dataset}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
                    )
                    f.write(f"{ate:.5f}\n")
                    f.write(f"{rpe_trans:.5f}\n")
                    f.write(f"{rpe_rot:.5f}\n")
                
                # Clear CUDA cache after each sequence to prevent memory accumulation
                torch.cuda.empty_cache()

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception

    # Clear all CUDA cache before synchronization to prevent OOM
    torch.cuda.empty_cache()
    distributed_state.wait_for_everyone()

    results = process_directory(save_dir)
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

    # Write the averages to the error log (only on the main process)
    if distributed_state.is_main_process:
        with open(f"{save_dir}/_error_log.txt", "a") as f:
            # Copy the error log from each process to the main error log
            for i in range(distributed_state.num_processes):
                if not os.path.exists(f"{save_dir}/_error_log_{i}.txt"):
                    break
                with open(f"{save_dir}/_error_log_{i}.txt", "r") as f_sub:
                    f.write(f_sub.read())
            f.write(
                f"Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n"
            )
    return avg_ate, avg_rpe_trans, avg_rpe_rot


def resize_or_crop(images: torch.Tensor, size=(224, 224), crop: bool = False):
    B, C, H, W = images.shape
    if isinstance(size, int):
        target_h, target_w = size, size
    else:
        target_h, target_w = size
    if crop:
        if target_h > H or target_w > W:
            raise ValueError(f"Crop size {size} cannot be larger than input size {(H, W)}")
        i = (H - target_h) // 2
        j = (W - target_w) // 2
        images_out = images[:, :, i:i+target_h, j:j+target_w]
    else:
        images_out = F.interpolate(images, size=size, mode="bilinear", align_corners=False)
    return images_out

from page.models.vggt import VGGT
from page.utils.load_fn import load_and_preprocess_images
from page.utils.pose_enc import pose_encoding_to_extri_intri
from page.utils.geometry import unproject_depth_map_to_point_map

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    add_path_to_dust3r(args.weights)

    args.no_crop = True

    def prepare_input(img_paths, size, crop=True,):
        images = load_and_preprocess_images(img_paths)
        images = resize_or_crop(images, size=size, crop=crop)
        return images

    def prepare_output(outputs: dict):
        """
        处理 vggt_inference_slide 的输出字典，返回和 eval_pose_estimation 对齐的结构。
        Args:
            outputs: dict，包含
                - "images": (N, 3, H, W)
                - "extrinsic": (N, 4, 4)
                - "intrinsic": (N, ...)
                - "depth_map": (N, 1, H, W)
                - "depth_conf": (N, 1, H, W)
        Returns:
            colors: list of (H,W,3) numpy or tensor
            pts3ds_self: torch.Tensor (N, H, W, 3)
            pts3ds_other: None (保持接口一致)
            conf_self: list of torch.Tensor (每帧的 depth_conf)
            conf_other: None (保持接口一致)
            cam_dict: dict { "focal": ..., "pp": ... }
            pr_poses: extrinsics (N, 4, 4)
        """
        images     = outputs["images"]        # (N, 3, H, W)
        extrinsic  = torch.inverse(outputs["extrinsic"])     # (N, 4, 4)
        intrinsic  = outputs["intrinsic"]     # (N, 3, 3)
        depth_map  = outputs["depth_map"]     # (N, 1, H, W)
        depth_conf = outputs["depth_conf"]    # (N, 1, H, W)
        N, _, H, W = images.shape
        # --- unproject depth map to 3D points ---
        pts3ds_self = []
        for i in range(N):
            pts = unproject_depth_map_to_point_map(
                depth_map[i].unsqueeze(0),  # (H,W)
                extrinsic[i].unsqueeze(0),              # (4,4)
                intrinsic[i].unsqueeze(0),             # (3,3)
            )  # (H,W,3)
            pts = torch.from_numpy(pts)
            pts3ds_self.append(pts.unsqueeze(0))
        pts3ds_self = torch.cat(pts3ds_self, dim=0)  # (N,H,W,3)
        # --- colors from images ---
        colors = [ (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8) for img in images ]
        # --- principle point + focal (简单取中心) ---
        pp = torch.tensor([W//2, H//2], device=images.device).float().repeat(N,1)
        focal = intrinsic[:,0,0]  # fx 简单取 (N,)
        cam_dict = {
            "focal": focal.cpu().numpy(),
            "pp": pp.cpu().numpy()}
        conf_self = depth_conf.cpu()
        return (colors,
            pts3ds_self,
            None,       # pts3ds_other 不再需要
            conf_self,
            None,       # conf_other 不再需要
            cam_dict, extrinsic.cpu())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    origin = args.weights
    model = VGGT(mask_hold_start=args.num_mask, mask_hold_end=args.num_mask)
    checkpoint = torch.load(origin, map_location=device)
    try:
        model.load_state_dict(checkpoint['model'], strict=False)
    except Exception as e:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation(args, model, save_dir=args.output_dir)
    print(f"ATE: {ate_mean:.5f}, RPE trans: {rpe_trans_mean:.5f}, RPE rot: {rpe_rot_mean:.5f}")