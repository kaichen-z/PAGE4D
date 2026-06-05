<div align="center">
<h1>Page-4d: VGGT-4D perception via Disentangled pose and geometry estimation</h1>

Two variants with similar performance:<br>
1.Training-only masking (= VGGT Structure).<br>
2.Inference-time masking (VGGT Structure with Mask).

<a href="https://openreview.net/pdf?id=Nfmzp5PBzr" target="_blank" rel="noopener noreferrer"> 
<img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF"></a>
<a href="https://arxiv.org/pdf/2510.17568"><img src="https://img.shields.io/badge/arXiv-2510.17568-b31b1b" alt="arXiv"></a>
<a href="https://page4d.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

**[Media Lab, MIT](https://www.media.mit.edu/)**; 
**[Harvard Medical School](https://hms.harvard.edu/)**

[Kaichen Zhou](https://kaichen-z.github.io/), [Yuhan Wang](https://yuhanwang14.github.io/), [Grace Chen](https://gracee-chen.github.io/), [Xinhai Chang](https://chang-xinhai.github.io/), [Gaspard Beaudouin](https://www.google.com), [Fangneng Zhan](https://fnzhan.com/), [Paul Pu Liang†](https://pliang279.github.io/), [Mengyu Wang†](https://wang.hms.harvard.edu/team/dr-wang/)

**(†: Jointly Supervised)**
</div>

```bibtex
@article{zhou2025page,
  title={PAGE-4D: Disentangled Pose and Geometry Estimation for VGGT-4D Perception},
  author={Zhou, Kaichen and Wang, Yuhan and Chen, Grace and Chang, Xinhai and Beaudouin, Gaspard and Zhan, Fangneng and Liang, Paul Pu and Wang, Mengyu},
  journal={arXiv preprint arXiv:2510.17568},
  year={2025}
}
```

## Overview

PAGE-4D (ICLR 2026) extends the Visual Geometry Grounded Transformer (VGGT, CVPR 2025) to dynamic scenes. It is a feed-forward neural network that directly infers key 4D scene attributes, including camera poses, depth maps, and dense point maps, while explicitly modeling dynamic elements such as moving humans and deformable objects—all without requiring post-processing or optimization.

## Quick Start

First, clone this repository to your local machine, and install the dependencies (torch, torchvision, numpy, Pillow, and huggingface_hub). 

```bash
git clone https://github.com/kaichen-z/PAGE4D.git
pip install -r requirements.txt
```

Now, try the model with just a few lines of code:

```python
import torch
from page.models.vggt import VGGT
from page.utils.load_fn import load_and_preprocess_images
from page.utils.pose_enc import pose_encoding_to_extri_intri
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
model = VGGT()
checkpoint = torch.load(Directory, map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]  
images = load_and_preprocess_images(image_names).to(device)
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)
```
## Training

Training uses `launch_gra.py` with gradient checkpointing for memory efficiency.

### Quick start

```bash
cd training_bash
bash final_train.sh
```

The script runs training with automatic retries on failure and logs to `logs/training_final.log`.

### Direct run

```bash
cd training
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29508 launch_gra.py --config training_final
```

For multi-GPU training, set `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` accordingly.

### Configuration

Edit `training/config/training_final.yaml` to customize:

- **Datasets**: Training and validation datasets under `data.train.dataset.dataset_configs` and `data.val.dataset.dataset_configs`. Update `dataset_location` paths for your environment.
- **Resume**: Set `checkpoint.resume_checkpoint_path` to resume from a checkpoint.
- **Experiment**: `exp_name` controls checkpoint and log directory names.
- **Debug limits**: `limit_train_batches` and `limit_val_batches` cap batches per epoch; set to `null` for full training.

Update `TRAINING_CMD` and `LOG_DIR` in `final_train.sh` if your project path differs from the default.

## Feature Map Visualization
We provide a detailed visualization strategy used in Figure 2 of our paper. The script `eval/visualization.py` extracts and visualizes the model's internal feature maps to illustrate how PAGE-4D disentangles frame-local and global cross-view information.

### Usage
```bash
cd eval
python visualization.py
```

Configure at the bottom of `visualization.py`:
- `directory`: Output folder for saved visualizations.
- `image_names`: List of image paths (multi-view inputs).
- `initial_num`, `gap`: Frame indices for video sequences (e.g., `rgb_{initial_num:05d}.jpg`, `rgb_{initial_num+gap:05d}.jpg`).

### Output
For each input image and each transformer layer, the script saves:
- `{name}_frame_feature_{layer}.png`: Frame-local feature heatmap.
- `{name}_global_feature_{layer}.png`: Global cross-view feature heatmap.


## Data-Preparation

### Prepare scripts

Dataset-specific preparation scripts live in `training/data/datasets/prepare/`. They sample frames and produce standardized directory layouts for the dataloaders.

**TUM RGB-D** (`tum_pre.py`):
- Input: Raw TUM format with `rgb.txt`, `groundtruth.txt`, `depth.txt`.
- Process: Associates RGB, depth, and pose by timestamp; samples 90 frames at stride 3.
- Output per sequence: `rgb_90/`, `depth_90/`, `groundtruth_90.txt`.

```bash
# Run from project root; update dataset_location in script if needed
python -m data.datasets.prepare.tum_pre
```

**Bonn RGB-D** (`bonn_pre.py`):
- Input: `rgbd_bonn_dataset/*/rgb/*.png`, `depth/*.png`, `groundtruth.txt`.
- Process: Samples frames 30–140 (110 frames) for sequences `balloon2`, `crowd2`, `crowd3`, `person_tracking2`, `synchronous`.
- Output per sequence: `rgb_110/`, `depth_110/`, `groundtruth_110.txt`.

```bash
python -m data.datasets.prepare.bonn_pre
```

Update the `dirs` path at the top of each script to your dataset location.

### Validate the dataloader

`dataset_validation.py` checks that the dataloader works with your config and optionally saves visualizations (point clouds, depth maps, tracks).

```bash
cd training
python -m data.dataset_validation --config debug
```

Enable your dataset in `training/data/datasets/config/debug.yaml` (or the config you pass) by uncommenting the corresponding dataset entry. The script loads the dataset via Hydra, iterates the loader, and can save:
- `.ply` point clouds (world and camera coordinates),
- Side-by-side track visualizations,
- Depth maps,
- Track-overlay videos.

Set `save_address` in the script to the desired output directory.

## Evaluation

We provide evaluation pipelines for **monocular depth**, **video depth**, and **relative pose (camera trajectory)** on dynamic scenarios. Each pipeline can be run via its `run_page.sh` script. Edit `model_weights`, `datasets`, and paths in the script (and in `eval/eval/*/metadata.py`) for your environment before running.

### 1. Monocular Depth (`eval/eval/monodepth/`)

Evaluates single-image depth estimation. Uncomment the `launch_page.py` block in `run_page.sh` to run inference first (saves depth `.npy`); otherwise the script runs `eval_metrics.py` on existing predictions (Abs Rel, Sq Rel, RMSE, δ thresholds).

```bash
# Edit model_weights, datasets in run_page.sh first
bash eval/eval/monodepth/run_page.sh
```

**Datasets**: sintel, bonn, dyncheck (edit the `datasets` array in the script). See `metadata.py` for more options.

### 2. Video Depth (`eval/eval/video_depth/`)

Evaluates depth on video sequences with sliding-window inference. Uses multi-GPU via `accelerate`.

```bash
# Edit model_weights, datasets in run_page.sh first
bash eval/eval/video_depth/run_page.sh
```

**Datasets**: sintel, bonn, dyncheck. **Metrics**: Abs Rel, Sq Rel, RMSE, Log RMSE, δ < 1.25, etc. To compute metrics after inference, uncomment and run the `eval_depth.py` block in the script.

### 3. Relative Pose (`eval/eval/relpose/`)

Evaluates camera trajectory (pose) estimation using evo. Outputs ATE and RPE (translation, rotation).

```bash
# Edit model_weights, datasets in run_page.sh first
bash eval/eval/relpose/run_page.sh
```

**Datasets**: sintel, tum. **Outputs**: `pred_traj.txt`, `pred_focal.txt`, `pred_intrinsics.txt`, trajectory plots, `*_eval_metric.txt` with ATE/RPE.

## Detailed Usage

You can also optionally choose which attributes (branches) to predict, as shown below. This achieves the same result as the example above. This example uses a batch size of 1 (processing a single scene), but it naturally works for multiple scenes.

```python
from page.utils.pose_enc import pose_encoding_to_extri_intri
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
```

</details>


## Checkpoint

**Spatial mask during training.** Fine-tuning uses a learnable spatial mask in the aggregator (`SpatialMaskHead_IMP` in `model/page/layers/block.py`). Its strength is scheduled with `mask_alpha(step, mask_hold_start, mask_hold_end)`: the mask is fully on for early optimizer steps, then its influence is reduced smoothly (cosine decay) until it is off. Set `mask_hold_start` / `mask_hold_end` in your training config (e.g. `training_final.yaml` under `model`).

**At inference:**

```python
#Non-mask version:
mask_hold_start = 0
mask_hold_end = 0
```
```python
#Mask-enabled version:
mask_hold_start > 0
mask_hold_end > 0
```

Two variants with similar performance:  
1.Training-only masking (= VGGT Structure).  
2.Inference-time masking (VGGT Structure with Mask).

**Download Weights (non mask version - use the mask only during the early stage of training) (Suggested).** Pretrained weights are released as `checkpoint_nomask.pt` on Hugging Face ([dataset page](https://huggingface.co/datasets/zhouk777/PAGE4D/tree/main)). Download the file and point the Quick Start `Directory` (or eval `model_weights`) to its path:

```bash
huggingface-cli download zhouk777/PAGE4D checkpoint_nomask.pt --repo-type dataset --local-dir .
```

**Download Weights (mask version - always keep mask during training).** Pretrained weights are released as `checkpoint_mask.pt` on Hugging Face ([dataset page](https://huggingface.co/datasets/zhouk777/PAGE4D/tree/main)). Download the file and point the Quick Start `Directory` (or eval `model_weights`) to its path:

```bash
huggingface-cli download zhouk777/PAGE4D checkpoint_mask.pt --repo-type dataset --local-dir .
```

## Interactive Demo

Our interactive code follows a similar design to VGGT (CVPR 2025). Please refer to their original [repository](https://github.com/facebookresearch/vggt/tree/main) for more details.

## Acknowledgements

Thanks to these great repositories: [VGGT](https://github.com/facebookresearch/vggt/tree/main), [CUT3R](https://github.com/CUT3R/CUT3R) and many other inspiring works in the community.
