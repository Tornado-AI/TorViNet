# -*- coding: utf-8 -*-
"""
数据预处理工具

包含:
    - 视频转帧提取
    - 数据集构建
    - 示例可视化
"""

import math
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


# 默认路径配置
VIDEO_DIR = '/mnt/4T/chenhj/datasets/my_dataset/new_dataset/video'
SAVE_DIR = '/mnt/4T/chenhj/datasets/my_dataset/new_dataset/extraction'


def video_to_frames(category):
    """
    将视频转换为帧序列

    处理逻辑:
        - 每 5 秒提取 64 帧
        - 视频不足 5 秒时，均匀采样 64 帧
        - 视频超过 5 秒时，分段提取，最后一段可能不足 5 秒

    Args:
        category: 视频类别文件夹名称
    """
    video_path = os.path.join(VIDEO_DIR, category)
    save_path = os.path.join(SAVE_DIR, category)

    video_list = os.listdir(video_path)

    for video_name in video_list:
        video_path_file = os.path.join(video_path, video_name)
        video_data = cv2.VideoCapture(video_path_file)

        # 获取视频信息
        video_frame_count = int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = video_data.get(cv2.CAP_PROP_FPS)

        # 计算视频时长和采样次数
        video_duration = math.floor(video_frame_count / video_fps)
        sample_count = video_duration // 5 + 1

        # 检查是否已处理
        save_video_path = os.path.join(save_path, video_name.split('.')[0])

        if sample_count == 1:
            # ===== 视频不足 5 秒，均匀采样 64 帧 =====
            if os.path.exists(save_video_path + '_0.npy'):
                continue

            frame_list = []
            for i in range(64):
                frame_index = math.floor(video_frame_count * (i / 64))
                video_data.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = video_data.read()

                if ret:
                    frame = cv2.resize(frame, (224, 224))
                    frame_list.append(frame)

            # 保存为 npy 文件
            os.makedirs(save_path, exist_ok=True)
            np.save(save_video_path + '_0.npy', frame_list)
            print(f'已处理: {video_name}, 帧数: 0')

        else:
            # ===== 视频超过 5 秒，分段提取 =====
            for i in range(sample_count):
                if os.path.exists(save_video_path + f'_{i}.npy'):
                    continue

                frame_list = []

                if i < (sample_count - 2):
                    # 中间段落：每段 5 秒
                    for j in range(64):
                        frame_index = math.floor(5 * video_fps * (i + j / 64))
                        video_data.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ret, frame = video_data.read()

                        if ret:
                            frame = cv2.resize(frame, (224, 224))
                            frame_list.append(frame)

                elif i == (sample_count - 2):
                    # 倒数第二段：提取剩余部分
                    for j in range(64):
                        frame_index = math.floor(
                            5 * video_fps * i + (j / 64) * (video_frame_count - 5 * video_fps * i)
                        )
                        video_data.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ret, frame = video_data.read()

                        if ret:
                            frame = cv2.resize(frame, (224, 224))
                            frame_list.append(frame)
                else:
                    break

                # 保存
                os.makedirs(save_path, exist_ok=True)
                np.save(save_video_path + f'_{i}.npy', frame_list)
                print(f'已处理: {video_name}, 段数: {i}')

        video_data.release()


def show_example(show_frame_num=16):
    """
    展示提取后的帧示例

    用于验证数据预处理的效果

    Args:
        show_frame_num: 展示的帧数量
    """
    # 加载示例数据
    example = np.load(
        '/mnt/4T/chenhj/datasets/my_dataset/new_dataset/extraction/'
        'tornado_my/20210601_03_12_split_01_0.npy'
    )

    fig = plt.figure(figsize=(50, 50))

    # ImageNet 数据集的均值和标准差
    mean = torch.tensor([[[[0.485]]], [[[0.456]]], [[[0.406]]]], dtype=torch.float32)
    std = torch.tensor([[[[0.229]]], [[[0.224]]], [[[0.225]]]], dtype=torch.float32)

    # 预处理
    data_input = torch.asarray(example, dtype=torch.float32)
    data_input = data_input.permute(3, 0, 1, 2).unsqueeze(0)
    data_input = (data_input / 255.0 - mean) / std

    print(f"数据形状: {data_input.shape}")
    print(f"数据范围: [{data_input.min():.3f}, {data_input.max():.3f}]")


def plot_input_element(video_path):
    """
    可视化模型的输入元素

    将视频帧保存为单独的图像文件

    Args:
        video_path: 视频路径（相对于 extraction 目录）
    """
    video_data = np.load(
        f'/mnt/4T/chenhj/datasets/my_dataset/new_dataset/extraction/'
        f'tornado_my/{video_path}_0.npy'
    )

    for i in range(64):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        ax.imshow(cv2.cvtColor(video_data[i], cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.savefig(f'input_element_{i:02d}.png', dpi=360)
        print(f"已保存帧: {i}")


def generate_green_gradient(n=10, hex_format=True):
    """
    生成绿色渐变色数组

    用于可视化

    Args:
        n: 颜色数量
        hex_format: 是否返回十六进制格式

    Returns:
        颜色值列表

    Example:
        >>> generate_green_gradient(5)
        ['#00441b', '#1b693f', '#368f63', '#51b487', '#6cdcab']
    """
    import matplotlib.colors as mcolors

    if n < 1:
        raise ValueError("颜色数量 n 必须大于 0")

    cmap = plt.get_cmap('Greens_r')
    colors = cmap(np.linspace(0, 1, n))

    if hex_format:
        return [mcolors.to_hex(c) for c in colors]
    else:
        return [tuple(c[:3]) for c in colors]


def plot_wavelet_transform():
    """
    展示小波变换的效果

    对视频帧进行 Haar 小波分解并可视化
    """
    # 加载示例数据
    data = np.load(
        '/mnt/4T/chenhj/datasets/my_dataset/new_dataset/extraction/'
        'tornado_my/20220731_00_00_split_00_0.npy'
    )
    data = data[60]  # 取第 60 帧

    # 转换为灰度图
    data = data.mean(axis=2)

    # Haar 小波变换
    import pywt
    LL, (LH, HL, HH) = pywt.dwt2(data, 'haar', mode='periodization')

    cmaps = ['viridis', 'turbo', 'turbo', 'turbo']
    titles = ['LL (低频)', 'LH (水平高频)', 'HL (垂直高频)', 'HH (对角高频)']

    for i, (d, cmap, title) in enumerate(zip([LL, LH, HL, HH], cmaps, titles)):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes((0., 0., 1, 1))
        ax.imshow(d, aspect='auto', cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        plt.savefig(f'wavelet_{i}.png', dpi=360)
        print(f"已保存: wavelet_{i}.png")


if __name__ == '__main__':
    # 示例：处理一个类别的视频
    # video_to_frames('tornado_my')

    # 示例：展示数据
    show_example()
