# -*- coding: utf-8 -*-
# -------------------------------

# @软件：PyCharm
# @PyCharm：
# @Python：python=3.11
# @项目：Tornado_video_detection

# -------------------------------

# @文件：data_prosessing.py
# @时间：2025/3/20 15:34
# @作者：chenhj
# @邮箱：2426742974@qq.com

# -------------------------------

"""
This script is used to process data.

-------------------------------------------
Fist, we need to transform a sample video into a series of frames,
every 5 seconds, we will get 64 frames.
If the video is less than 5 seconds, 64 frames will be obtained uniformly.
And if the video is more than 5 seconds, the part of the video will produce an extra 64 frames.

-------------------------------------------
Second, we need to demonstrate a few examples of the extracted frames.
1、we will show the example of diffuse frames.
2、we will show the example of architectural sheltering.
3、we will show the example of the long-distance shooting.

-------------------------------------------
Third, we need to plot the model's input element.
Input is 64 frames.

"""
import math
import os
import cv2

import numpy as np



video_dir = '/mnt/4T/chenhj/datasets/my_dataset/new_dataset/video'
video_categories = os.listdir(video_dir)
save_dir = '/mnt/4T/chenhj/datasets/my_dataset/new_dataset/extraction'

def video_to_frames(category):
    """
    Transform a sample video into a series of frames.
    :param category: video category
    :return: None
    """
    video_path = os.path.join(video_dir, category)
    save_path = os.path.join(save_dir, category)

    video_list = os.listdir(video_path)
    for video_name in video_list:
        video_data = cv2.VideoCapture(os.path.join(video_path, video_name))

        # Get video information
        video_frame_count = int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = video_data.get(cv2.CAP_PROP_FPS)

        # Get video duration and calculate the number of frames to be extracted
        video_duration = math.floor(video_frame_count / video_fps)
        sample_count = video_duration // 5 + 1

        # Extract frames
        # If the video is less than 5 seconds, 64 frames will be obtained uniformly.
        if sample_count == 1:
            frame_list = []


            save_video_path = os.path.join(save_path, video_name.split('.')[0])
            if os.path.exists(save_video_path + f'_0.npy'):
                continue

            for i in range(64):
                frame_index = math.floor(video_frame_count * (i / 64))
                video_data.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = video_data.read()
                frame = cv2.resize(frame, (224, 224))
                frame_list.append(frame)

            # Save frames as npy files
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            np.save(save_video_path + f'_0.npy', frame_list)
            print(f'Save the video: {video_name} frame: 0')

        # If the video is more than 5 seconds, the part of the video will produce an extra 64 frames.
        else:
            for i in range(sample_count):
                frame_list = []

                save_video_path = os.path.join(save_path, video_name.split('.')[0])
                if os.path.exists(save_video_path + f'_{i}.npy'):
                    continue

                if i < (sample_count - 2):
                    for j in range(64):
                        frame_index = math.floor(5 * video_fps * (i + j / 64))
                        video_data.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ret, frame = video_data.read()
                        frame = cv2.resize(frame, (224, 224))
                        frame_list.append(frame)
                elif i == (sample_count - 2):
                    for j in range(64):
                        frame_index = math.floor(5 * video_fps * i + (j / 64) * (video_frame_count - 5 * video_fps * i))
                        video_data.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ret, frame = video_data.read()
                        frame = cv2.resize(frame, (224, 224))
                        frame_list.append(frame)
                else:
                    break

                # Save frames as npy files
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                np.save(save_video_path + f'_{i}.npy', frame_list)
                print(f'Save the video: {video_name} frame: {i}')

def show_example(show_frame_num=16):
    """
    Show the example of the extracted frames.
    :param show_frame_num: the number of frames to be shown
    :return: None
    """

    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from models.TSTVM_01 import TSTVM

    # # Load the example of the extracted frames
    # example1 = np.load('/mnt/4T/chenhj/datasets/my_dataset/new_dataset/extraction/tornado_my/20210601_03_23_split_04_0.npy')
    # example3 = np.load('/mnt/4T/chenhj/datasets/my_dataset/new_dataset/extraction/tornado_my/20210609_00_03_split_01_0.npy')
    # example2 = np.load('/mnt/4T/chenhj/datasets/my_dataset/new_dataset/extraction/tornado_my/20210702_00_00_split_00_0.npy')
    #
    # fig, axs = plt.subplots(6, show_frame_num//2, figsize=(50, 40))
    #
    # # Set the row spacing and column spacing
    # plt.subplots_adjust(wspace=0.05, hspace=0.005)
    #
    # for i in range(show_frame_num):
    #
    #     axs[i//8, i%8].imshow(cv2.cvtColor(example1[i*(64//show_frame_num)], cv2.COLOR_BGR2RGB))
    #     axs[2 + i//8, i%8].imshow(cv2.cvtColor(example2[i*(64//show_frame_num)], cv2.COLOR_BGR2RGB))
    #     axs[4 + i//8, i%8].imshow(cv2.cvtColor(example3[i*(64//show_frame_num)], cv2.COLOR_BGR2RGB))
    #
    #     axs[i//8, i%8].axis('off')
    #     axs[2 + i//8, i%8].axis('off')
    #     axs[4 + i//8, i%8].axis('off')
    #
    # plt.savefig('example.png', dpi=360)

    example = np.load('/mnt/4T/chenhj/datasets/my_dataset/new_dataset/extraction/tornado_my/20210601_03_12_split_01_0.npy')
    fig = plt.figure(figsize=(50, 50))

    # for i in range(8):
    #     for j in range(8):
    #         ax = fig.add_axes((j*0.125, (7-i)*0.125, 0.125, 0.125))
    #         ax.imshow(cv2.cvtColor(example[i * 8 + j], cv2.COLOR_BGR2RGB), aspect='auto')
    #         ax.axis('off')
    #
    # plt.savefig('example01.png', dpi=360)

    # imagenet数据集的均值和方差
    mean = torch.tensor([[[[0.485]]], [[[0.456]]], [[[0.406]]]], dtype=torch.float32)
    std = torch.tensor([[[[0.229]]], [[[0.224]]], [[[0.225]]]], dtype=torch.float32)

    data_input = torch.asarray(example, dtype=torch.float32)
    data_input = data_input.permute(3, 0, 1, 2).unsqueeze(0)
    data_input = (data_input / 255.0 - mean) / std

    model = TSTVM()
    for i in range(10):
        model.load_state_dict(torch.load(f'/mnt/4T/chenhj/checkpoints/TornadoCV/tvm_01/tvm_01_{i:02d}.pth'))

        out = torch.sigmoid(model(data_input))
        # print(f'{i},{model(data_input)}')

        print(f'{i},{out}')

def plot_input_element(video_path):
    """
    Plot the model's input element.
    Input is 64 frames.
    :param video_path: the path of the video
    :return: None
    """
    import matplotlib.pyplot as plt

    video_data = np.load(f'/mnt/4T/chenhj/datasets/my_dataset/new_dataset/extraction/tornado_my/{video_path}_0.npy')

    for i in range(64):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        ax.imshow(cv2.cvtColor(video_data[i], cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.savefig(f'input_element_{i:02d}.png', dpi=360)
        print("Save the input element: ", i)


def generate_green_gradient(n=10, hex_format=True):
    """
    生成从深绿到浅绿的渐变色数组

    参数：
    n : int > 0，颜色数量（默认10）
    hex_format : bool，是否返回十六进制格式（True），否则返回RGB元组（False）

    返回：
    list - 颜色值列表，格式根据hex_format参数决定

    示例：
    generate_green_gradient(5)
    ['#00441b', '#1b693f', '#368f63', '#51b487', '#6cdcab']
    """

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if n < 1:
        raise ValueError("颜色数量n必须大于0")

    # 使用matplotlib的Greens_r色图（_r表示反向，从深到浅）
    cmap = plt.get_cmap('Greens_r')

    # 生成等间距的颜色采样点
    colors = cmap(np.linspace(0, 1, n))

    # 转换为指定格式
    if hex_format:
        return [mcolors.to_hex(c) for c in colors]
    else:
        return [tuple(c[:3]) for c in colors]  # 返回RGB元组，忽略alpha通道

def plot_illustration_element():
    """"
    """
    import matplotlib.pyplot as plt

    # # sigmoid function
    # x = np.linspace(-10, 10, 100)
    # y = 1 / (1 + np.exp(-x))
    #
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.plot(x, y, color='#01420e', linewidth=5)


    # # sorted
    # x = np.arange(1, 65)
    # y = (np.square(x + 8) + 100) / 5500
    #
    # colors = generate_green_gradient(64)
    #
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # for i in range(64):
    #     ax.barh(x[i], y[i], color=colors[63-i], linewidth=5)

    # haar wavelet function
    x = np.linspace(-10, 10, 100)
    y = np.zeros_like(x)
    y[(x > -10) & (x < 0)] = 1
    y[(x >= 0) & (x < 10)] = -1

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(x, y, color='#a66d17', linewidth=5)
    ax.plot([-15, 15], [0, 0], color='black', linewidth=1)

    ax.axis('off')

    plt.savefig('wevalet.png', dpi=360)


def plot_embedding_element():
    """"
    """

    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    import pywt

    data = np.load('/mnt/4T/chenhj/datasets/my_dataset/new_dataset/extraction/tornado_my/20220731_00_00_split_00_0.npy')
    data = data[60]
    # fig = plt.figure(figsize=(5, 5))
    # for i in range(8):
    #     ax = fig.add_axes((0.13*i, 0., 0.1, 1.))
    #     ax.imshow(cv2.cvtColor(data[:, i*28:i*28+28, :], cv2.COLOR_BGR2RGB), aspect='auto')
    #     ax.axis('off')
    #
    # plt.subplots_adjust(
    #     left=0,
    #     right=1,
    #     bottom=0,
    #     top=1,
    #     wspace=0,  # 子图间水平间距
    #     hspace=0  # 子图间垂直间距
    # )
    #
    # plt.savefig('embedding_element1.png', dpi=360)
    #
    # fig = plt.figure(figsize=(5, 5))
    # for i in range(8):
    #     ax = fig.add_axes((0., 0.13 * (7 - i), 1, 0.1))
    #     ax.imshow(cv2.cvtColor(data[i * 28:i * 28 + 28, :, :], cv2.COLOR_BGR2RGB), aspect='auto')
    #     ax.axis('off')
    #
    # plt.subplots_adjust(
    #     left=0,
    #     right=1,
    #     bottom=0,
    #     top=1,
    #     wspace=0,  # 子图间水平间距
    #     hspace=0  # 子图间垂直间距
    # )
    #
    # plt.savefig('embedding_element2.png', dpi=360)
    #
    # fig = plt.figure(figsize=(5, 5))
    # for i in range(8):
    #     for j in range(8):
    #         ax = fig.add_axes((0.13 * j, 0.13 * (7 - i), 0.1, 0.1))
    #         ax.imshow(cv2.cvtColor(data[i * 28:i * 28 + 28, j * 28:j * 28 + 28, :], cv2.COLOR_BGR2RGB), aspect='auto')
    #         ax.axis('off')
    #
    # plt.subplots_adjust(
    #     left=0,
    #     right=1,
    #     bottom=0,
    #     top=1,
    #     wspace=0,  # 子图间水平间距
    #     hspace=0  # 子图间垂直间距
    # )
    #
    # plt.savefig('embedding_element3.png', dpi=360)

    # wavelet transform
    data = data.mean(axis=2)
    LL, (LH, HL, HH) = pywt.dwt2(data, 'haar', mode='periodization')
    cmaps = ['viridis', 'turbo', 'turbo', 'turbo']
    for i, d in enumerate([LL, LH, HL, HH]):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes((0., 0., 1, 1))
        ax.imshow(d, aspect='auto', cmap=cmaps[i])
        ax.axis('off')
        plt.savefig(f'wavelet_{i}.png', dpi=360)
        # plt.show()
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_axes((0., 0., 1, 1))
    # ax.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB), aspect='auto')
    # ax.axis('off')
    # plt.savefig('wavelet_data.png', dpi=360)

    pass




def main():
    show_example()

if __name__ == '__main__':
    main()