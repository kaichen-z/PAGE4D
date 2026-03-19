import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MODEL_DIR = _PROJECT_ROOT / "model"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import torch.nn.functional as F
import pdb
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.monodepth.metadata import dataset_metadata
MAX_FRAMES = 160
MAX = True
POINT = False
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, help="path to the model weights", default=""
    )
    parser.add_argument("--num_mask", type=int, default=0, help="number of mask")
    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument("--output_dir", type=str, default="", help="value for outdir")
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )
    parser.add_argument(
        "--full_seq", type=bool, default=False, help="whether to use all seqs"
    )
    parser.add_argument("--seq_list", default=None)

    parser.add_argument(
        "--eval_dataset", type=str, default="nyu", choices=list(dataset_metadata.keys())
    )
    return parser


def eval_mono_depth_estimation(args, model, device):
    metadata = dataset_metadata.get(args.eval_dataset)
    if metadata is None:
        raise ValueError(f"Unknown dataset: {args.eval_dataset}")

    img_path = metadata.get("img_path")
    if "img_path_func" in metadata:
        img_path = metadata["img_path_func"](args)
    
    process_func = metadata.get("process_func")
    if process_func is None:
        raise ValueError(
            f"No processing function defined for dataset: {args.eval_dataset}"
        )
    for filelist, save_dir in process_func(args, img_path):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        eval_mono_depth(args, model, device, filelist, save_dir=save_dir)

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

def prepare_input( img_paths, size, crop=True,):
    images = load_and_preprocess_images(img_paths)
    images = resize_or_crop(images, size=size, crop=crop)
    return images

def eval_mono_depth(args, model, device, filelist, save_dir=None):
    model.eval()
    load_img_size = 518
    for file in tqdm(filelist):
        # construct the "image pair" for the single image
        file = [file]
        views = prepare_input(file, load_img_size, crop=not args.no_crop)
        outputs = vggt_inference_single(views, model, device)
                # 取 (H,W)
        depth_map = outputs["depth_map"][0, :, :, 0].float().cpu()            # (H,W)
        #pts3ds_self = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
        #depth_map = pts3ds_self[0][..., -1].mean(dim=0)

        if save_dir is not None:
            # save the depth map to the save_dir as npy
            np.save(
                f"{save_dir}/{file[0].split('/')[-1].replace('.png','depth.npy')}",
                depth_map.cpu().numpy(),
            )
            # also save the png
            depth_map = (depth_map - depth_map.min()) / (
                depth_map.max() - depth_map.min()
            )
            depth_map = (depth_map * 255).cpu().numpy().astype(np.uint8)
            cv2.imwrite(
                f"{save_dir}/{file[0].split('/')[-1].replace('.png','depth.png')}",
                depth_map,
            )


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
    
from page.models.vggt import VGGT
from page.utils.load_fn import load_and_preprocess_images
from page.utils.pose_enc import pose_encoding_to_extri_intri

def vggt_inference_single(images: torch.Tensor, model: torch.nn.Module, device="cuda"):
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
    if POINT:
        depth_map = point_depth_map.view(depth_map.size())
        depth_conf = point_conf.view(depth_conf.size())
    return {
        "images": images,              # (S, 3, H, W)
        "extrinsic": extrinsic[0],     # (S, 4, 4)
        "intrinsic": intrinsic[0],     # (S, ...)
        "depth_map": depth_map[0],     # (S, 1, H, W) 或 (S, H, W)
        "depth_conf": depth_conf[0],   # (S, 1, H, W) 或 (S, H, W)
    }

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.eval_dataset == "sintel":
        args.full_seq = True
    else:
        args.full_seq = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    origin = args.weights
    model = VGGT(mask_hold_start=args.num_mask, mask_hold_end=args.num_mask)
    checkpoint = torch.load(origin, map_location=device)
    try:
        model.load_state_dict(checkpoint['model'], strict=False)
    except Exception as e:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    eval_mono_depth_estimation(args, model, args.device)
