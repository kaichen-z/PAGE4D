"""
Launch script with gradient checkpointing. Uses trainer_gra for memory saving.
Run with: torchrun ... launch_gra.py --config final_9_gra
"""
import os
import argparse
import torch
import gc
from hydra import initialize, compose
from trainer_gra import Trainer


def cleanup_gpu_memory():
    if torch.cuda.is_available():
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
        gc.collect()
        print("GPU memory cleaned up for all devices")


def main():
    parser = argparse.ArgumentParser(description="Train with gradient checkpointing")
    parser.add_argument("--config", type=str, default="final_9_gra")
    parser.add_argument("--memory_fraction", type=float, default=0.5)
    args = parser.parse_args()
    cleanup_gpu_memory()
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)
    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
