# %%
import glob
import os
import pdb
import shutil
"""
From Monst3R:
https://github.com/Junyi42/monst3r/tree/574cc77ad278bad582f470e5382624e01f8769a7
"""
dirs = glob.glob("/workspace/data/kaichen/data/test/bonn/rgbd_bonn_dataset/*/")
dirs = sorted(dirs)
test_seq = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]
# python3 -m data.datasets.bonn_pre 
# extract frames
for dir in dirs:
    print(os.path.basename(dir))
    if any(seq in dir for seq in test_seq):
        print(dir)
        frames = glob.glob(dir + 'rgb/*.png')
        frames = sorted(frames)
        # sample 110 frames at the stride of 2
        frames = frames[30:140]
        # cut frames after 110
        new_dir = dir + 'rgb_110/'
        for frame in frames:
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy(frame, new_dir)
            # print(f'cp {frame} {new_dir}')
        depth_frames = glob.glob(dir + 'depth/*.png')
        depth_frames = sorted(depth_frames)
        # sample 110 frames at the stride of 2
        depth_frames = depth_frames[30:140]
        # cut frames after 110
        new_dir = dir + 'depth_110/'
        for frame in depth_frames:
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy(frame, new_dir)
            # print(f'cp {frame} {new_dir}')
import numpy as np
for dir in dirs:
    if any(seq in dir for seq in test_seq):
        gt_path = "groundtruth.txt"
        gt = np.loadtxt(dir + gt_path)
        gt_110 = gt[30:140]
        np.savetxt(dir + 'groundtruth_110.txt', gt_110)