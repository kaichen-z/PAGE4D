import glob
import os
import shutil
import numpy as np
import re

"""
From Monst3R:
https://github.com/Junyi42/monst3r/tree/574cc77ad278bad582f470e5382624e01f8769a7
"""
from bisect import bisect_left
# python3 -m data.datasets.prepare.tum_pre

def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    Input:
    filename -- File name
    Output:
    dict -- dictionary of (stamp,data) tuples
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp, data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    Input:
    first_list -- first dictionary of (stamp, data) tuples
    second_list -- second dictionary of (stamp, data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation
    Output:
    matches -- list of matched tuples ((stamp1, data1), (stamp2, data2))
    """
    # Convert keys to sets for efficient removal
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    matches.sort()
    return matches

# >>> 新增：在已排序键列表里，找与 target 最接近且不超过阈值的时间戳
def nearest_within(sorted_keys, target, max_diff):
    i = bisect_left(sorted_keys, target)
    cand = []
    if i < len(sorted_keys): cand.append(sorted_keys[i])
    if i > 0: cand.append(sorted_keys[i-1])
    if not cand: return None
    best = min(cand, key=lambda k: abs(k - target))
    return best if abs(best - target) <= max_diff else None

dirs = glob.glob("/workspace/data/kaichen/data/test/tum/*/")
dirs = sorted(dirs)
# extract frames
for dir in dirs:
    frames = []
    depth_frames_sel = []
    gt = []
    first_file = dir + 'rgb.txt'
    second_file = dir + 'groundtruth.txt'
    depth_file  = dir + 'depth.txt'   # >>> 新增：TUM depth 索引（若文件名不同，改这里）
    first_list = read_file_list(first_file)
    second_list = read_file_list(second_file)
    depth_list  = read_file_list(depth_file)    # DEP: {ts: [path, ...]}
    matches = associate(first_list, second_list, 0.0, 0.02)
    # for a,b in matches[:10]:
    #     print("%f %s %f %s"%(a," ".join(first_list[a]),b," ".join(second_list[b])))
    depth_keys_sorted = sorted(depth_list.keys())
    max_diff_depth = 0.04   # 可按需要调
    for a,b in matches:
        frames.append(dir + first_list[a][0])
        gt.append([b]+second_list[b])
        nd = nearest_within(depth_keys_sorted, a, max_diff_depth)
        if nd is None:
            # 没找到合适的 depth：与其破坏对齐，建议同时丢弃这一对（RGB/GT）
            # 为最小改动，这里简单地“跳过 depth”，也可以选择 pop 最近一条 frames/gt
            depth_frames_sel.append(None)
        else:
            depth_frames_sel.append(dir + depth_list[nd][0])
    # sample 90 frames at the stride of 3
    frames = frames[::3][:90]
    gt_90 = gt[::3][:90]
    depth_frames_sel = depth_frames_sel[::3][:90]
    print(len(frames), len(gt_90), len(depth_frames_sel), '===========frames, gt_90, depth_frames_sel===========')
    # cut frames after 90
    new_dir = dir + 'rgb_90/'
    depth_out = dir + 'depth_90/'
    # ---------------------------- RGB ---------------------------- 
    if os.path.exists(new_dir):
        for f in glob.glob(os.path.join(new_dir, "*.png")):
            os.remove(f)
    os.makedirs(new_dir, exist_ok=True)
    for frame in frames:
        shutil.copy(frame, new_dir)
    # ---------------------------- GT ---------------------------- 
    if os.path.exists(dir + 'groundtruth_90.txt'):
        os.remove(dir + 'groundtruth_90.txt')
    with open(dir + 'groundtruth_90.txt', 'w') as f:
        for pose in gt_90:
            f.write(f"{' '.join(map(str, pose))}\n")
    # ---------------------------- Depth ---------------------------- 
    if os.path.exists(depth_out):
        for f in glob.glob(os.path.join(depth_out, "*.png")):
            os.remove(f)
    os.makedirs(depth_out, exist_ok=True)
    for num_frame, frame in enumerate(depth_frames_sel):
        if frame is None:
            continue
        filename = os.path.basename(frame)
        name, ext = os.path.splitext(filename)
        new_path = os.path.join(depth_out, f"{name}{ext}")
        date, number = name.split('.')
        while os.path.exists(new_path):
            number += 1
            new_path = os.path.join(depth_out, f"{date:10d}.{number:06d}{ext}")
        shutil.copy(frame, new_path)
        # print(len(glob.glob(os.path.join(depth_out, "*.png"))), '===========len(glob.glob(os.path.join(depth_out, "*.png")))===========', num_frame)
    print(len(glob.glob(os.path.join(depth_out, "*.png"))), '===========len(glob.glob(os.path.join(depth_out, "*.png")))===========', num_frame)
