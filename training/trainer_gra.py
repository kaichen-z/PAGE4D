# trainer_gra: Trainer with gradient checkpointing for memory saving.
# Uses torch.utils.checkpoint to recompute activations during backward instead of storing them.
import torch
import torch.nn as nn
import logging

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _PROJECT_ROOT / "model"
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "training", _MODEL_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
import contextlib
import gc
import pdb
import sys
import json
import logging
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers

# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
VAL_FIRST = False

def set_gpu_memory_limit_gb(memory_gb, device=None):
    """
    Set exact GPU memory limit in GB instead of using fraction.
    
    Args:
        memory_gb (float): Desired memory limit in GB
        device (torch.device or int, optional): GPU device. If None, uses current device.
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, cannot set memory limit")
        return False
        
    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device = device.index if device.index is not None else 0
        
    # Get total GPU memory in bytes
    total_memory_bytes = torch.cuda.get_device_properties(device).total_memory
    total_memory_gb = total_memory_bytes / 1e9
    
    # Calculate fraction needed for desired GB amount
    if memory_gb > total_memory_gb:
        logging.warning(f"Requested {memory_gb}GB exceeds total GPU memory {total_memory_gb:.2f}GB")
        memory_fraction = 1.0
    else:
        memory_fraction = memory_gb / total_memory_gb
        
    # Set the memory fraction
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=device)
    
    logging.info(f"Set GPU memory limit to {memory_gb}GB ({memory_fraction:.3f} of {total_memory_gb:.2f}GB total) on device {device}")
    return True


class MemoryManager:
    """Enhanced memory management utilities for stable memory usage."""
    
    def __init__(self, device, adaptive_cleanup=True, memory_threshold_gb=38):
        self.device = device
        self.adaptive_cleanup = adaptive_cleanup
        self.memory_threshold_gb = memory_threshold_gb
        self.cleanup_counter = 0
        self.peak_memory_usage = 0
        self.last_cleanup_step = 0
        
    def set_exact_memory_limit_gb(self, memory_gb):
        """Set exact GPU memory limit in GB instead of using fraction."""
        if not torch.cuda.is_available():
            logging.warning("CUDA not available, cannot set memory limit")
            return False
            
        # Get total GPU memory in bytes
        total_memory_bytes = torch.cuda.get_device_properties(self.device).total_memory
        total_memory_gb = total_memory_bytes / 1e9
        
        # Calculate fraction needed for desired GB amount
        if memory_gb > total_memory_gb:
            logging.warning(f"Requested {memory_gb}GB exceeds total GPU memory {total_memory_gb:.2f}GB")
            memory_fraction = 1.0
        else:
            memory_fraction = memory_gb / total_memory_gb
            
        # Set the memory fraction
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=self.device)
        
        logging.info(f"Set GPU memory limit to {memory_gb}GB ({memory_fraction:.3f} of {total_memory_gb:.2f}GB total)")
        return True
        
    def get_memory_usage_gb(self):
        """Get current memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device) / 1e9
        return 0
    
    def should_cleanup(self, step, force_cleanup_interval=8):  # Reduce from 16 to 8
        """Determines if memory cleanup should be performed."""
        if not self.adaptive_cleanup:
            return step % force_cleanup_interval == 0
        current_memory = self.get_memory_usage_gb()
        # reserved_memory = self.get_memory_reserved_gb()
        # Cleanup if memory usage exceeds threshold or every 100 steps as fallback
        memory_pressure = current_memory > self.memory_threshold_gb
        # print(f"Memory pressure: {memory_pressure}, current_memory: {current_memory}, reserved_memory: {reserved_memory}")
        interval_cleanup = (step - self.last_cleanup_step) >= force_cleanup_interval
        print(f"Memory pressure: {memory_pressure}, current_memory: {current_memory}, interval_cleanup: {interval_cleanup}, pid: {os.getpid()}")
        return memory_pressure or interval_cleanup
    
    def cleanup_memory(self, step, aggressive=False):
        """Perform memory cleanup with logging."""
        if not torch.cuda.is_available():
            return
        before_cleanup = self.get_memory_usage_gb()
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        if aggressive:
            # More aggressive cleanup
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
        after_cleanup = self.get_memory_usage_gb()
        freed_memory = before_cleanup - after_cleanup
        logging.info(f"Step {step}: Freed {freed_memory:.2f}GB memory "
                    f"(Before: {before_cleanup:.2f}GB, After: {after_cleanup:.2f}GB)")
        self.last_cleanup_step = step
        self.cleanup_counter += 1

class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """
    EPSILON = 1e-8
    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        self.highest_prev  = 0
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # --------------------------- NEW BLOCK ---------------------------
        # Create run-specific log directory AFTER distributed setup
        # Only rank 0 creates the directory, then broadcasts to all ranks
        logging.log_dir = self._make_unique_run_dir_distributed(logging.log_dir)
        # If the tensorboard path was templated with ${logging.log_dir} in YAML
        # it is already resolved; refresh it so it follows the new folder.
        if hasattr(logging, "tensorboard_writer") and \
           hasattr(logging.tensorboard_writer, "path"):
            logging.tensorboard_writer.path = os.path.join(
                logging.log_dir, "tensorboard"
            )
        # -----------------------------------------------------------------

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Initialize memory manager for better memory handling
        self.memory_manager = MemoryManager(
            device=self.device,
            adaptive_cleanup=True,
            memory_threshold_gb=40  # Lower from 34 to 30
        )
        
        # self.memory_manager.set_exact_memory_limit_gb(35)

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            try:
                self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
            except:
                print("Error loading checkpoint, continuing with new training run.")
        else:   
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        
        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    # --------------------------- NEW METHOD ---------------------------
    def _make_unique_run_dir_distributed(self, base_log_dir: str) -> str:
        """
        Turn `logs` → `logs/train_01_logs`, `logs/train_02_logs`, …
        The counter is computed by scanning existing sub-directories.
        Only rank 0 creates the directory, then broadcasts to all ranks.
        """
        if self.rank == 0:
            # Only rank 0 creates the directory
            base = Path(base_log_dir)
            base.mkdir(parents=True, exist_ok=True)

            # Identify folders like train_XX_logs
            existing = [
                p for p in base.iterdir()
                if p.is_dir() and p.name.startswith("train_") and p.name.endswith("_logs")
            ]
            if existing:
                nums = [
                    int(p.name.split("_")[1])    # train_XX_logs → XX
                    for p in existing
                    if p.name.split("_")[1].isdigit()
                ]
                next_idx = max(nums) + 1
            else:
                next_idx = 1

            run_dir = base / f"train_{next_idx:02d}_logs"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_dir_str = str(run_dir)
        else:
            # Other ranks wait for the directory path
            run_dir_str = ""

        # Broadcast the directory path from rank 0 to all other ranks
        run_dir_list = [run_dir_str]
        dist.broadcast_object_list(run_dir_list, src=0)
        
        return run_dir_list[0]
    # ------------------------------------------------------------------

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict)
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint:
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            optimizer_states = checkpoint["optimizer"]
            if isinstance(optimizer_states, list):
                # Handle multiple optimizers (saved as list)
                for i, optim in enumerate(self.optims):
                    if i < len(optimizer_states):
                        optim.optimizer.load_state_dict(optimizer_states[i])
            else:
                # Handle single optimizer (most common case)
                self.optims[0].optimizer.load_state_dict(optimizer_states)

        # Load training progress
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False) 
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            self.model = freeze_modules(
                self.model,
                patterns=["aggregator"],
                recursive=False)
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def _find_highest_checkpoint_number(self, checkpoint_folder: str, epoch: int=0) -> int:
        """
        Scans the checkpoint folder to find the highest numbered checkpoint.
        Args:
            checkpoint_folder: Path to the checkpoint directory
        Returns:
            Highest checkpoint number found, or 0 if none exist
        """
        if not os.path.exists(checkpoint_folder): return 0
        checkpoint_numbers = []
        for filename in os.listdir(checkpoint_folder):
            if filename.startswith("checkpoint_") and not filename.endswith(".json"):
                # Extract number from checkpoint_X format
                try:
                    number_str = filename.replace("checkpoint_", "").replace(".pt", "")
                    checkpoint_numbers.append(int(number_str))
                except ValueError:
                    continue  # Skip files that don't have valid numbers
        return max(checkpoint_numbers) if checkpoint_numbers else 0

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch+previous}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if int(epoch) == 0:
                self.highest_prev = self._find_highest_checkpoint_number(checkpoint_folder, epoch)
                print(f"highest_prev: {self.highest_prev}")
            if (self.checkpoint_conf.save_freq > 0 and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)):
                # Find highest existing checkpoint number and add current epoch
                new_checkpoint_num = self.highest_prev + int(epoch)
                checkpoint_names.append(f"checkpoint_{new_checkpoint_num}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],}
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )

    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.highest_prev = self._find_highest_checkpoint_number(self.checkpoint_conf.save_dir)
            if self.highest_prev == 0 and VAL_FIRST:
                print('----------This is the first time to run the training----------')
                self.run_val()
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs with validation at the beginning of each cycle."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            # Run validation at the specified frequency BEFORE training epoch
            # This creates the pattern: val + train + val + train + ...
            if self.epoch % self.val_epoch_freq == 0 and self.epoch > 0:
            # if self.epoch % self.val_epoch_freq == 0:
                self.run_val()
            
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            self.train_epoch(dataloader)
            
            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            #gc.collect()
            #torch.cuda.empty_cache()
            #torch.cuda.reset_peak_memory_stats()
            self.memory_manager.cleanup_memory(self.epoch, aggressive=True)
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        self.memory_manager.cleanup_memory(self.epoch, aggressive=True)

        self.val_epoch(dataloader)
        
        del dataloader
        #gc.collect()
        #torch.cuda.empty_cache()
        #torch.cuda.reset_peak_memory_stats()
        self.memory_manager.cleanup_memory(self.epoch, aggressive=True)


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break
            if self.memory_manager.should_cleanup(data_iter):
                self.memory_manager.cleanup_memory(data_iter, aggressive=True)
            else:
                self.memory_manager.cleanup_memory(data_iter, aggressive=False)
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16
            
            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict = self._step(
                        batch, self.model, phase, loss_meters
                    )

            # Manual extraction of objective loss for validation (same as training)
            if "objective" in val_loss_dict:
                val_objective_loss = val_loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = batch["images"].shape[0]
                loss_meters[loss_key].update(val_objective_loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)
        return True

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        batch_size_meter = AverageMeter("BS", self.device, ":.0f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                batch_size_meter,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            try:
                start_time_this_iter = time.time()  # Start timing the iteration
                if data_iter > limit_train_batches:
                    break
                # Signal that this rank is starting the iteration
                iteration_status = torch.tensor([0.0], device=self.device)  # 0 = success, 1 = error
                # Adaptive memory cleanup based on usage patterns
                if self.memory_manager.should_cleanup(data_iter):
                    self.memory_manager.cleanup_memory(data_iter, aggressive=True)
                else:
                    self.memory_manager.cleanup_memory(data_iter, aggressive=False)
                # measure data loading time
                data_time.update(time.time() - end)
                data_times.append(data_time.val)
                with torch.cuda.amp.autocast(enabled=False):
                    batch = self._process_batch(batch)
                batch = copy_data_to_device(batch, self.device, non_blocking=True)
                batch_size_meter.update(batch["images"].shape[0])
                accum_steps = self.accum_steps
                if accum_steps==1:
                    chunked_batches = [batch]
                else:
                    chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)
                self._run_steps_on_batch_chunks(
                    chunked_batches, phase, loss_meters
                )
                # compute gradient and do SGD step
                assert data_iter <= limit_train_batches  # allow for off by one errors
                exact_epoch = self.epoch + float(data_iter) / limit_train_batches
                self.where = float(exact_epoch) / self.max_epochs
                assert self.where <= 1 + self.EPSILON
                if self.where < 1.0:
                    for optim in self.optims:
                        optim.step_schedulers(self.where)
                else:
                    logging.warning(
                        f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                    )
                        
                # Log schedulers
                if self.steps[phase] % self.logging_conf.log_freq == 0:
                    for i, optim in enumerate(self.optims):
                        for j, param_group in enumerate(optim.optimizer.param_groups):
                            for option in optim.schedulers[j]:
                                optim_prefix = (
                                    f"{i}_"
                                    if len(self.optims) > 1
                                    else (
                                        "" + f"{j}_"
                                        if len(optim.optimizer.param_groups) > 1
                                        else ""
                                    )
                                )
                                self.tb_writer.log(
                                    os.path.join("Optim", f"{optim_prefix}", option),
                                    param_group[option],
                                    self.steps[phase],
                                )
                    self.tb_writer.log(
                        os.path.join("Optim", "where"),
                        self.where,
                        self.steps[phase],
                    )

                # Always unscale gradients before checking inf/nan and optimizer step
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                # Clipping gradients and detecting diverging gradients
                if self.gradient_clipper is not None:
                    grad_norm_dict = self.gradient_clipper(model=self.model)

                    for key, grad_norm in grad_norm_dict.items():
                        loss_meters[f"Grad/{key}"].update(grad_norm)

                # Check for inf/nan gradients before optimizer step
                has_inf_nan = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_inf_nan = True
                            logging.warning(f"Found inf/nan gradients in {name}")
                            param.grad.data.zero_()

                # Optimizer step - only if no inf/nan gradients
                if not has_inf_nan:
                    for optim in self.optims:   
                        self.scaler.step(optim.optimizer)
                    self.scaler.update()
                else:
                    # Skip this step if we found inf/nan gradients
                    logging.warning("Skipping optimizer step due to inf/nan gradients")
                    # Note: scaler.unscale_() was already called above, so we can safely call update()
                    self.scaler.update()

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)
                
                if time.time() - start_time_this_iter > 72.0*3:
                    raise TimeoutError(f"Iteration {data_iter} exceeded 72.0*3s")

                # Mark this iteration as successful
                iteration_status[0] = 0.0
                    
            except Exception as e:
                # Mark this iteration as failed
                iteration_status[0] = 1.0
                
                print(f'====Encounter Error (PID: {os.getpid()})====', file=sys.stderr)
                print(f'Exception Type: {type(e).__name__}', file=sys.stderr)
                print(f'Exception Message: {str(e)}', file=sys.stderr)
                # Print detailed CUDA memory info if it's a CUDA error
                if torch.cuda.is_available():
                    try:
                        for gpu_id in range(torch.cuda.device_count()):
                            memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
                            memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
                            max_memory_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1e9
                            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                            print(f'GPU {gpu_id}: Allocated={memory_allocated:.2f}GB, Reserved={memory_reserved:.2f}GB, Max={max_memory_allocated:.2f}GB, Total={total_memory:.2f}GB', file=sys.stderr)
                    except Exception as cuda_err:
                        print(f'Error getting CUDA memory info: {cuda_err}', file=sys.stderr)
                self.memory_manager.cleanup_memory(data_iter, aggressive=True)
                print("we clean one time here.")
                # Recreate the scaler to reset any corrupted state
                print("Recreating GradScaler due to error", file=sys.stderr)
                self.scaler = torch.cuda.amp.GradScaler(enabled=self.scaler.is_enabled())
        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16      
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)

        # Check for inf/nan gradients before optimizer step
        has_inf_nan = False
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_inf_nan = True
                    logging.warning(f"Found inf/nan gradients in {name}")
                    param.grad.data.zero_()

        # Only call scaler.step() if no inf/nan gradients were found
        if not has_inf_nan:
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()
        else:
            # Skip this step if we found inf/nan gradients
            logging.warning("Skipping optimizer step due to inf/nan gradients")
            self.scaler.update()  # Still update the scaler


    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks",
            "tracks", "track_vis_mask", "track_positive_mask",  # Add track data keys
        ]        
        string_keys = ["seq_name"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        
        return batch

    def _process_batch(self, batch: Mapping):      
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)

        # Normalize camera extrinsics and points. The function returns new tensors.
        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
            )

        # Replace the original values in the batch with the normalized ones.
        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths

        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.
        Returns:
            A dictionary containing the computed losses.
        """
        # Extract query points from tracks if available for track head processing
        temporal_features = batch.get("temporal_features", None)
        if "tracks" in batch:
            # Use first frame coordinates as query points for tracking
            # batch["tracks"] has shape [B, S, N, 2], we want [B, N, 2] for first frame
            # Note: tracks are already padded to track_num by datasets to ensure consistent dimensions
            query_points = batch["tracks"][:, 0, :, :]  # First frame coordinates for all batch items
            if self.rank == 0 and self.steps[phase] % 100 == 0:  # Log occasionally
                logging.info(f"Track data available - query_points shape: {query_points.shape}, tracks shape: {batch['tracks'].shape}")
            y_hat = model(images=batch["images"], query_points=query_points, temporal_features=temporal_features)
        else:
            # No track data available, run model without query points
            if self.rank == 0 and self.steps[phase] % 100 == 0:  # Log occasionally
                logging.info("No track data in batch - running model without query points")
            y_hat = model(images=batch["images"], temporal_features=temporal_features)
        
        # Loss computation
        loss_dict = self.loss(y_hat, batch)
        
        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]
        
        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )


class TrainerGradCheckpoint(Trainer):
    """Extends local Trainer; uses aggregator's block-level gradient checkpointing for memory saving.
    Outer checkpoint removed to avoid CheckpointError (tensor count mismatch during recomputation).
    """

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        temporal_features = batch.get("temporal_features", None)

        # Direct forward; aggregator internally has block-level checkpoint for frame/global blocks.
        if "tracks" in batch:
            y_hat = model(images=batch["images"], query_points=batch["tracks"][:, 0, :, :], temporal_features=temporal_features)
        else:
            y_hat = model(images=batch["images"], temporal_features=temporal_features)

        loss_dict = self.loss(y_hat, batch)
        log_data = {**y_hat, **loss_dict, **batch}
        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])
        self.steps[phase] += 1
        return loss_dict


def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data


Trainer = TrainerGradCheckpoint
