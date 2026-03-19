# In this one we test the performance on dyancmi 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
from pathlib import Path
# Add PAGE4D/model to path so 'page' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # PAGE4D root
_MODEL_DIR = _PROJECT_ROOT / "model"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))
import pdb
import torch
import matplotlib
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
import torch
from page.models.vggt import VGGT
from page.utils.load_fn import load_and_preprocess_images
import numpy as np
from PIL import Image
from scipy.stats import skew

def otsu_threshold_batch(img: torch.Tensor) -> torch.Tensor:
    """
    Batched Otsu's thresholding without Python loops.
    Args:
        img: (B, H, W) tensor, any range.
    Returns:
        mask: (B, H, W) bool tensor
    """
    B, H, W = img.shape
    img = img.detach()
    # 1) Normalize each image to [0, 255]
    img_min = img.view(B, -1).min(dim=1)[0].view(B, 1, 1)
    img_max = img.view(B, -1).max(dim=1)[0].view(B, 1, 1)
    img_norm = (img - img_min) / (img_max - img_min + 1e-8)
    img_uint8 = (img_norm * 255).to(torch.uint8)  # (B, H, W)
    # 2) Flatten
    flat = img_uint8.view(B, -1)  # (B, N)
    N = flat.size(1)
    # 3) Compute batched histograms using bincount + offset
    offsets = torch.arange(B, device=img.device) * 256  # batch offset
    flat_offsets = flat + offsets[:, None]  # (B, N)
    hist_all = torch.bincount(flat_offsets.view(-1), minlength=B * 256)
    hist = hist_all.view(B, 256).float()  # (B, 256)
    # 4) Probabilities
    prob = hist / N  # (B, 256)
    # 5) Cumulative sums
    omega = torch.cumsum(prob, dim=1)  # (B, 256)
    mu = torch.cumsum(prob * torch.arange(256, device=img.device), dim=1)  # (B, 256)
    mu_t = mu[:, -1].unsqueeze(1)  # (B, 1)
    # 6) Between-class variance
    numerator = (mu_t * omega - mu) ** 2
    denominator = omega * (1 - omega)
    variance = torch.zeros_like(numerator)
    valid = denominator > 0
    variance[valid] = numerator[valid] / denominator[valid]
    # 7) Best thresholds for each batch
    best_thresh = variance.argmax(dim=1)  # (B,)
    # 8) Apply thresholds (vectorized comparison)
    thresholds = best_thresh.view(B, 1, 1)
    mask = img_uint8 > thresholds  # (B, H, W)
    return mask

def confidence_to_rgb(confidence_map: torch.Tensor, cmap_name='Reds') -> torch.Tensor:
    """
    Converts a 2D confidence map (H, W) to an RGB image (H, W, 3).
    Args:
        confidence_map: torch.Tensor of shape (H, W), values in [0, 1] or [min, max]
        cmap_name: str, e.g. 'Reds', 'plasma', 'hot', 'viridis', etc.
    Returns: torch.Tensor of shape (H, W, 3) with RGB values in [0, 1]
    """
    # Normalize confidence to [0, 1]
    conf = confidence_map.clone()
    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
    # Convert to numpy and apply colormap
    conf_np = conf.cpu().numpy()
    colormap = matplotlib.colormaps.get_cmap(cmap_name)
    colored = colormap(conf_np)  # (H, W, 4), includes alpha
    # Remove alpha and convert to torch
    rgb = torch.from_numpy(colored[:, :, :3]).float()
    return rgb

def compute_statistical_features(tensor: torch.Tensor):
    L, B, N, P, C = tensor.shape
    # Mean and Std
    mean = tensor.mean(dim=0)  # (L, B, N, C)
    std = tensor.std(dim=0)    # (L, B, N, C)
    # Skewness: use numpy for now
    x_np = tensor.detach().cpu().numpy()
    skewness = torch.tensor(skew(x_np, axis=0, bias=False), device=tensor.device, dtype=tensor.dtype)
    # Energy: sum of squares
    energy = (tensor ** 2).sum(dim=0)  # (L, B, N, C)
    # Entropy: softmax over patches
    probs = F.softmax(tensor, dim=0)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=0)  # (L, B, N, C)
    # Stack all features
    features = torch.stack([mean, std, skewness, energy, entropy], dim=-1)  # (L, B, N, C, 5)
    return features

def normalize_feature(feature):
    return (feature - feature.min()) / (feature.max() - feature.min())

def main(directory, save_name, image_names, save_model_path, save_result = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    channel = 1024
    # Load and preprocess example images (replace with your own image paths)
    images = load_and_preprocess_images(image_names).to(device)
    input_H, input_W = images.size(2), images.size(3)
    patch_H, patch_W = input_H//model.aggregator.patch_size, input_W//model.aggregator.patch_size
    patch_size = patch_H * patch_W
    gloabl_list = []
    frame_list = []

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        for num in range(len(aggregated_tokens_list)):
            frame_list.append(aggregated_tokens_list[num][...,:channel])
            gloabl_list.append(aggregated_tokens_list[num][...,channel:])
        frame_list = torch.stack(frame_list, dim=0)
        gloabl_list = torch.stack(gloabl_list, dim=0) # (Layers, Batch, Num, Num_path_h*Num_path_w, channel)
        frame_features = compute_statistical_features(frame_list) 
        Batch, Num_img, _, Channel, Num_Stats = frame_features.size()
        frame_features_square = frame_features[:, :, -patch_size:, :, :].view(Batch, Num_img, patch_H, patch_W, Channel, Num_Stats)

        os.makedirs(f"{directory}/{save_name:05d}", exist_ok=True)
        if save_result:
            for i in range(images.size(1)):
                feature_mean_fr = frame_features_square[0,i,:,:,:,0].mean(dim=2).cpu()
                H_S, W_S = feature_mean_fr.shape[:2]; TIME = 12

                number_layer = gloabl_list.size(0)
                frame_list_square = frame_list[:, :, :, -patch_size:, :].view(number_layer, Batch, Num_img, patch_H, patch_W, Channel).cpu()
                gloabl_list_square = gloabl_list[:, :, :, -patch_size:, :].view(number_layer, Batch, Num_img, patch_H, patch_W, Channel).cpu()

                for layer in tqdm(range(number_layer)):
                    frame_features_square_layer = frame_list_square[layer][0, i].mean(dim=-1)
                    gloabl_features_square_layer = gloabl_list_square[layer][0, i].mean(dim=-1)

                    Dynamic_mask_img = normalize_feature(gloabl_features_square_layer)
                    Dynamic_mask_img = confidence_to_rgb(Dynamic_mask_img, cmap_name='YlOrRd')
                    Dynamic_mask_img = Image.fromarray((Dynamic_mask_img.clamp(0, 1) * 255).byte().numpy()).resize((W_S*TIME, H_S*TIME), resample=Image.BILINEAR)  
                    Dynamic_mask_img.save(f"{directory}/{save_name:05d}/{save_name+i:05d}_global_feature_{layer}.png")

                    Dynamic_mask_img = normalize_feature(frame_features_square_layer)
                    Dynamic_mask_img = confidence_to_rgb(Dynamic_mask_img, cmap_name='YlOrRd')
                    Dynamic_mask_img = Image.fromarray((Dynamic_mask_img.clamp(0, 1) * 255).byte().numpy()).resize((W_S*TIME, H_S*TIME), resample=Image.BILINEAR)  
                    Dynamic_mask_img.save(f"{directory}/{save_name:05d}/{save_name+i:05d}_frame_feature_{layer}.png")
                break

if __name__ == "__main__":
    # -----------------------
    directory = "evaluation/"
    frame_name = 200
    initial_num = 200
    gap = 16
    image_names = [f"/workspace/code/12_4d/datasets/template/odyssey/dancing/rgbs/rgb_{initial_num:05d}.jpg",\
                    f"/workspace/code/12_4d/datasets/template/odyssey/dancing/rgbs/rgb_{initial_num+gap:05d}.jpg", \
                        f"/workspace/code/12_4d/datasets/template/odyssey/dancing/rgbs/rgb_{initial_num+gap*2:05d}.jpg", \
                        f"/workspace/code/12_4d/datasets/template/odyssey/dancing/rgbs/rgb_{initial_num+gap*3:05d}.jpg"]  
    save_model_path = "checkpoint.pt"
    main(directory, frame_name, image_names, save_model_path)
