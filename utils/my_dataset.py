# -*- coding: utf-8 -*-
"""
PyTorch Dataset 实现

用于加载龙卷风检测数据集
"""

import json
import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# 设置随机数种子，保证可复现性
random.seed(0)

# ImageNet 数据集的均值和标准差（用于归一化）
MEAN = torch.tensor([[[[0.485]]], [[[0.456]]], [[[0.406]]]] , dtype=torch.float32)
STD = torch.tensor([[[[0.229]]], [[[0.224]]], [[[0.225]]]] , dtype=torch.float32)

# 数据集根目录
DATA_PATH = '/mnt/4T/chenhj/00_datasets/my_dataset/new_dataset'


def normalize(data):
    """
    数据归一化

    使用 ImageNet 的均值和标准差进行归一化

    Args:
        data: 输入数据，形状为 (T, H, W, C)

    Returns:
        归一化后的数据，形状为 (C, T, H, W)
    """
    data = np.transpose(data, (3, 0, 1, 2))  # (T, H, W, C) -> (C, T, H, W)
    data = torch.tensor(data, dtype=torch.float32)
    data = (data / 255.0 - MEAN) / STD
    return data


def load_mp4(path):
    """
    加载 MP4 视频文件

    Args:
        path: 视频文件路径

    Returns:
        视频帧数组，形状为 (帧数, H, W, C)
    """
    import time
    start_time = time.time()

    cap = cv2.VideoCapture(path)
    video_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for i in range(int(video_frames_num)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            frames.append(frame)
        else:
            break

    cap.release()
    frames = np.array(frames)

    print(f'加载耗时: {time.time() - start_time:.2f}s')
    return frames


class MyDataset(Dataset):
    """
    龙卷风检测数据集

    从 JSON 文件加载数据集划分，加载预处理好的视频帧数据

    Args:
        split: 数据集划分，可选 'train', 'val', 'test'
    """

    def __init__(self, split='train'):
        self.split = split

        # 加载数据集划分文件
        split_file = os.path.join(DATA_PATH, 'present_dataset_split.json')
        with open(split_file, 'r') as f:
            self.data_list = json.load(f)[split]

        print(f"加载 {split} 数据集，共 {len(self.data_list)} 个样本")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        获取单个样本

        Returns:
            data: 预处理后的视频数据，形状 (C, T, H, W)
            label: 标签，0 表示无龙卷风，1 表示有龙卷风
            path: 数据文件路径
        """
        item = self.data_list[index]
        mp4_path, label = item

        # 加载预处理好的 .npy 文件
        npy_path = os.path.join(DATA_PATH, 'extraction', mp4_path + '.npy')
        data = np.load(npy_path)

        # 归一化处理
        data = normalize(data)

        return data, label, mp4_path


def get_dataloaders():
    """
    创建数据加载器

    Returns:
        train_loader, valid_loader, test_loader
    """
    train_dataset = MyDataset('train')
    valid_dataset = MyDataset('valid')
    test_dataset = MyDataset('test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    return train_loader, valid_loader, test_loader


def dataset_split():
    """
    数据集划分工具

    根据类别将数据划分为训练集、验证集和测试集
    生成 txt 文件记录划分结果
    """
    # 负样本类别（不含龙卷风的场景）
    vivd_neg_paths = [
        'on_fire', 'train_accident', 'van_accident', 'dirty_contamined',
        'rockslide_rockfall', 'motorcycle_accident', 'earthquake', 'oil_spill',
        'mudslide_mudflow', 'traffic_jam', 'ship_accident', 'snowslide_avalanche',
        'truck_accident', 'bicycle_accident', 'flooded', 'wildfire', 'derecho',
        'fire_whirl', 'drought', 'damaged', 'dust_sand_storm', 'burned',
        'airplane_accident', 'blocked', 'landslide', 'thunderstorm', 'snow_covered',
        'dust_devil', 'nuclear_explosion', 'storm_surge', 'heavy_rainfall',
        'hailstorm', 'with_smoke', 'bus_accident', 'collapsed', 'tropical_cyclone',
        'sinkhole', 'car_accident', 'under_construction', 'volcanic_eruption',
        'ice_storm', 'fog'
    ]

    # 抖音负样本类别
    douyin_neg_paths = ['ai', 'after_calamity', 'wind', 'rain', 'hail', 'dust_whirl']

    # 正样本类别（含龙卷风的场景）
    douyin_pos_path = os.path.join(DATA_PATH, 'tornado_our')
    vivd_pos_path = os.path.join(DATA_PATH, 'tornado')

    # ===== 划分负样本数据 =====
    train_list, valid_list, test_list = [], [], []

    # VIVD 负样本划分（9:0.5:0.5 比例）
    for neg_path in vivd_neg_paths:
        neg_full_path = os.path.join(DATA_PATH, neg_path)

        if not os.path.exists(neg_full_path):
            continue

        for file_name in os.listdir(neg_full_path):
            rand = random.randint(0, 10)
            if rand < 9:
                train_list.append(f'{neg_path}/{file_name}')
            elif rand <= 5:
                valid_list.append(f'{neg_path}/{file_name}')
            else:
                test_list.append(f'{neg_path}/{file_name}')

    print(f"VIVD 负样本划分:")
    print(f"  训练集: {len(train_list)}, 验证集: {len(valid_list)}, 测试集: {len(test_list)}")

    # 抖音负样本划分
    for neg_path in douyin_neg_paths:
        neg_full_path = os.path.join(DATA_PATH, neg_path)

        if not os.path.exists(neg_full_path):
            continue

        names, splits = [], []
        for file_name in os.listdir(neg_full_path):
            if file_name[:-7] not in names:
                names.append(file_name)

                rand = random.randint(0, 10)
                if rand < 8:
                    splits.append('train')
                elif rand < 5:
                    splits.append('valid')
                else:
                    splits.append('test')

        for name, split in zip(names, splits):
            if split == 'train':
                train_list.append(f'{neg_path}/{name}')
            elif split == 'valid':
                valid_list.append(f'{neg_path}/{name}')
            else:
                test_list.append(f'{neg_path}/{name}')

    print(f"抖音负样本划分:")
    print(f"  训练集: {len(train_list)}, 验证集: {len(valid_list)}, 测试集: {len(test_list)}")

    # 保存负样本划分
    with open(os.path.join(DATA_PATH, 'train.txt'), 'w') as f:
        for name in train_list:
            f.write(f'{name} 0\n')

    with open(os.path.join(DATA_PATH, 'valid.txt'), 'w') as f:
        for name in valid_list:
            f.write(f'{name} 0\n')

    with open(os.path.join(DATA_PATH, 'test.txt'), 'w') as f:
        for name in test_list:
            f.write(f'{name} 0\n')

    # ===== 划分正样本数据 =====
    train_list, valid_list, test_list = [], [], []

    # VIVD 正样本划分（9:0.5:0.5 比例）
    for file_name in os.listdir(vivd_pos_path):
        rand = random.randint(0, 10)
        if rand < 9:
            train_list.append(f'tornado/{file_name}')
        elif rand < 4:
            valid_list.append(f'tornado/{file_name}')
        else:
            test_list.append(f'tornado/{file_name}')

    # 抖音正样本按时间划分
    for file_name in os.listdir(douyin_pos_path):
        utc = file_name.split('_')[0]
        if utc <= '20220701':
            train_list.append(f'tornado_our/{file_name}')
        elif utc < '20220805':
            valid_list.append(f'tornado_our/{file_name}')
        else:
            test_list.append(f'tornado_our/{file_name}')

    print(f"正样本划分:")
    print(f"  训练集: {len(train_list)}, 验证集: {len(valid_list)}, 测试集: {len(test_list)}")

    # 保存正样本划分
    with open(os.path.join(DATA_PATH, 'train.txt'), 'a') as f:
        for name in train_list:
            f.write(f'{name} 1\n')

    with open(os.path.join(DATA_PATH, 'valid.txt'), 'a') as f:
        for name in valid_list:
            f.write(f'{name} 1\n')

    with open(os.path.join(DATA_PATH, 'test.txt'), 'a') as f:
        for name in test_list:
            f.write(f'{name} 1\n')


if __name__ == '__main__':
    # 示例：划分数据集
    dataset_split()

    # 示例：获取数据加载器
    # train_loader, valid_loader, test_loader = get_dataloaders()
