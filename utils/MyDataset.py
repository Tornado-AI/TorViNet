# -*-coding: Utf-8 -*-
# Author: CHJ
# Time: 2024/7/17 下午3:42


"""
自定义数据集，用于加载数据、数据预处理等
"""
import json
import cv2

import numpy as np
from torch.utils.data import Dataset,DataLoader
from config import config
import os
import torch
import random

# 设置随机数种子
random.seed(0)

# imagenet数据集的均值和方差
mean = torch.tensor([[[[0.485]]], [[[0.456]]], [[[0.406]]]],dtype=torch.float32)
std = torch.tensor([[[[0.229]]], [[[0.224]]], [[[0.225]]]],dtype=torch.float32)
data_path = '/mnt/4T/chenhj/00_datasets/my_dataset/new_dataset'

def normalize(data):

    data = np.transpose(data, (3, 0, 1, 2)) # (3, 64, 224, 224)
    data = torch.tensor(data, dtype=torch.float32) # (64, 224, 224, 3)

    # print(data.shape)
    # 归一化, 由于数据集是RGB格式，所以均值和方差是固定的
    data = (data / 255.0 - mean) / std

    return data

def load_mp4(path):
    import time
    start_time =  time.time()
    v = cv2.VideoCapture(path)
    video_frames_num = v.get(cv2.CAP_PROP_FRAME_COUNT)

    frames = []
    for i in range(0, int(video_frames_num)):
        v.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = v.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            frames.append(frame)
        else:
            break
    v.release()

    frames = np.array(frames)

    print(f'load_mp4 time: {time.time()-start_time}')

    return frames


class MyDataset(Dataset):
    def __init__(self, split='train'):

        self.split = split
        with open(os.path.join(data_path, f'present_dataset_split.json'), 'r') as f:
            self.data_list = json.load(f)[split]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]

        mp4_path, label = item

        # if self.split == 'train' or self.split == 'val':
        #     mp4_path = os.path.join(data_path, 'extracted_64', mp4_path+'.npy')
        #     data = np.load(mp4_path)
        # else:
        #     mp4_path = os.path.join(data_path, 'video', mp4_path+'.mp4')
        #     data = load_mp4(mp4_path)

        mp4_path = os.path.join(data_path, 'extraction', mp4_path + '.npy')
        data = np.load(mp4_path)
        data = normalize(data)

        return data, label, mp4_path



def get_dataloaders():
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



    # 负样本路径
    vivd_neg_paths = ['on_fire', 'train_accident', 'van_accident', 'dirty_contamined', 'rockslide_rockfall',
                      'motorcycle_accident', 'earthquake', 'oil_spill', 'mudslide_mudflow', 'traffic_jam',
                      'ship_accident', 'snowslide_avalanche', 'truck_accident', 'bicycle_accident', 'flooded',
                      'wildfire', 'derecho', 'fire_whirl', 'drought',
                      'damaged', 'dust_sand_storm', 'burned', 'airplane_accident',
                      'blocked', 'landslide', 'thunderstorm', 'snow_covered',
                      'dust_devil', 'nuclear_explosion', 'storm_surge', 'heavy_rainfall', 'hailstorm',
                      'with_smoke', 'bus_accident', 'collapsed', 'tropical_cyclone',
                      'sinkhole', 'car_accident', 'under_construction',
                      'volcanic_eruption', 'ice_storm', 'fog']

    douyin_neg_paths = ['ai', 'after_calamity', 'wind', 'rain', 'hail', 'dust_whirl']

    # 正样本路径
    douyin_pos_path = os.path.join(data_path, 'tornado_our')
    vivd_pos_path = os.path.join(data_path, 'tornado')

    # # 整理负样本数据
    train_list, valid_list, test_list = [], [], []
    for neg_path in vivd_neg_paths:
        for file_name in os.listdir(os.path.join(data_path, neg_path)):
            if random.randint(0, 10) < 9:
                train_list.append(f'{neg_path}/{file_name}')
            elif random.randint(0, 10) <= 5:
                valid_list.append(f'{neg_path}/{file_name}')
            else:
                test_list.append(f'{neg_path}/{file_name}')

    print(f'train_neg_count: {len(train_list)}')
    print(f'valid_neg_count: {len(valid_list)}')
    print(f'test_neg_count: {len(test_list)}')

    for neg_path in douyin_neg_paths:
        names, splits = [], []
        for file_name in os.listdir(os.path.join(data_path, neg_path)):
            if file_name[:-7] not in names:
                names.append(file_name)
                if random.randint(0, 10) < 8:
                    splits.append('train')
                elif random.randint(0, 10) < 5:
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

    print(f'train_neg_count: {len(train_list)}')
    print(f'valid_neg_count: {len(valid_list)}')
    print(f'test_neg_count: {len(test_list)}')

    with open(os.path.join(data_path, 'train.txt'), 'w') as f:
        for name in train_list:
            f.write(f'{name} 0\n')
    with open(os.path.join(data_path, 'valid.txt'), 'w') as f:
        for name in valid_list:
            f.write(f'{name} 0\n')
    with open(os.path.join(data_path, 'test.txt'), 'w') as f:
        for name in test_list:
            f.write(f'{name} 0\n')

    # 整理正样本数据
    train_list, valid_list, test_list = [], [], []
    for file_name in os.listdir(vivd_pos_path):
        if random.randint(0, 10) < 9:
            train_list.append(f'tornado/{file_name}')
        elif random.randint(0, 10) < 4:
            valid_list.append(f'tornado/{file_name}')
        else:
            test_list.append(f'tornado/{file_name}')

    for file_name in os.listdir(douyin_pos_path):
        utc = file_name.split('_')[0]
        if utc <= '20220701':
            train_list.append(f'tornado_our/{file_name}')
        elif utc < '20220805':
            valid_list.append(f'tornado_our/{file_name}')
        else:
            test_list.append(f'tornado_our/{file_name}')
    print('-----------------------------------------')
    print(f'train_pos_count: {len(train_list)}')
    print(f'valid_pos_count: {len(valid_list)}')
    print(f'test_pos_count: {len(test_list)}')

    with open(os.path.join(data_path, 'train.txt'), 'a') as f:
        for name in train_list:
            f.write(f'{name} 1\n')
    with open(os.path.join(data_path, 'valid.txt'), 'a') as f:
        for name in valid_list:
            f.write(f'{name} 1\n')
    with open(os.path.join(data_path, 'test.txt'), 'a') as f:
        for name in test_list:
            f.write(f'{name} 1\n')

if __name__ == '__main__':

    dataset_split()
    # # train_loader, valid_loader, test_loader = get_dataloaders()

    # for video_path in os.listdir(os.path.join('D:\datasets\my_dataset\extracted_video', 'tornado')):
    #     data = np.load(os.path.join('D:\datasets\my_dataset\extracted_video', 'tornado', video_path))['arr_0'] # (24, 512, 512, 3)
    #
    #     print(np.mean(data))


    # for video_path in os.listdir('D:\datasets\my_dataset\extracted_video'):
    #     for video_name in os.listdir(os.path.join('D:\datasets\my_dataset\extracted_video', video_path)):
    #         if video_name[-4:] == '.npy':
    #             continue
    #         data = np.load(os.path.join('D:\datasets\my_dataset\extracted_video', video_path, video_name))['arr_0']
    #         os.remove(os.path.join('D:\datasets\my_dataset\extracted_video', video_path, video_name))
    #         np.save(os.path.join('D:\datasets\my_dataset\extracted_video', video_path, video_name[:-4]), data)
    #         print(video_name[:-4])