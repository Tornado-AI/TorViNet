# TorViNet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-Expert%20Systems%20With%20Applications-blue)](https://www.sciencedirect.com/science/article/pii/S0957417426000072)

**TorViNet: A Spatiotemporal Deep Learning Network for Tornado Detection in User-Captured Social Media Videos**

> Hongjin Chen, Kanghui Zhou, Zhonghua Zheng, Lei Han, Yongguang Zheng  
> *Expert Systems With Applications*, 2026  
> DOI: [10.1016/j.eswa.2026.000007](https://www.sciencedirect.com/science/article/pii/S0957417426000072)

## 📖 简介

龙卷风是破坏力极强且生命周期极短的强对流天气，传统气象雷达往往只能探测到龙卷风特征，却无法确认其是否真正触地。社交媒体视频提供了直接的视觉证据，是实时灾害监测的重要补充来源。

TorViNet 是一个面向专家系统的 AI 驱动时空识别框架，直接从用户拍摄的社交媒体视频中检测龙卷风。针对真实场景中的三大挑战，TorViNet 集成了三个核心模块：

- **DFSM**（Dynamic Frame-level Selection Module）— 动态帧级选择，过滤时间上无信息的冗余帧
- **SFMHA**（Spatial-Frequency Multi-Head Attention）— 空间频率注意力，增强细粒度涡旋结构特征
- **LC-MLP**（Local Contrast MLP）— 对比感知细化，抑制背景干扰

在超过 10,000 条经过验证的龙卷风/非龙卷风视频数据集上，TorViNet 达到了 **91% 准确率** 和 **0.89 F1 分数**，超越了主流视频分类基线模型。

## ✨ 核心特性

- 🎯 **动态帧选择** — 自动过滤冗余帧，聚焦关键时刻
- 🌊 **空间频率注意力** — 结合空间域与频率域特征，增强涡旋结构感知
- 🔍 **对比感知细化** — 抑制背景噪声，突出龙卷风目标
- 📱 **面向社交媒体** — 专为嘈杂、不稳定、远距离拍摄场景设计
- ⚡ **高性能** — 91% 准确率，0.89 F1，优于主流视频分类模型

## 🏗️ 模型架构

```
输入视频 (B, 3, T, 224, 224)
        ↓
  DFSM 动态帧选择
  (过滤冗余帧，保留关键帧)
        ↓
  SFMHA 空间频率注意力
  (增强涡旋结构特征)
        ↓
  LC-MLP 对比感知细化
  (抑制背景，突出目标)
        ↓
  分类输出 (龙卷风 / 非龙卷风)
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+（推荐）

### 安装

```bash
git clone https://github.com/Tornado-AI/TorViNet.git
cd TorViNet
pip install -r requirements.txt
```

### 训练

```bash
python trainer.py
```

修改 `config.py` 中的路径配置：

```python
DATASETS_PATH = '/path/to/your/datasets/'
CHECKPOINT_PATH = '/path/to/save/checkpoints/'
```

## 📁 数据集准备

```
datasets/
└── my_dataset/
    └── new_dataset/
        ├── video/                        # 原始视频
        ├── extraction/                   # 预处理后的帧（.npy）
        ├── dataset_split.json            # 原始划分文件
        └── present_dataset_split.json    # 实际有效数据划分
```

**视频预处理：**

```python
from utils.data_processing import video_to_frames
video_to_frames('tornado_my')
```

**生成数据集划分：**

```python
from utils.dataset import build_json_file
build_json_file()
```

## 📂 项目结构

```
TorViNet/
├── config.py              # 路径与训练参数配置
├── trainer.py             # 训练脚本
├── requirements.txt       # 依赖列表
├── models/
│   ├── __init__.py
│   └── torvinet.py        # TorViNet 模型定义
├── utils/
│   ├── __init__.py
│   ├── my_dataset.py      # PyTorch Dataset
│   ├── dataset.py         # 数据集划分工具
│   └── data_processing.py # 视频预处理工具
└── docs/
    └── architecture.md    # 模型架构详细说明
```

## 📊 性能

在超过 10,000 条视频的数据集上：

| 指标 | 数值 |
|------|------|
| Accuracy | 91% |
| F1-score | 0.89 |

## 📝 引用

如果本工作对您有帮助，请引用：

```bibtex
@article{chen2026torvinet,
  title     = {TorViNet: A spatiotemporal deep learning network for tornado detection in user-captured social media videos},
  author    = {Chen, Hongjin and Zhou, Kanghui and Zheng, Zhonghua and Han, Lei and Zheng, Yongguang},
  journal   = {Expert Systems With Applications},
  volume    = {308},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.eswa.2026.000007},
  url       = {https://www.sciencedirect.com/science/article/pii/S0957417426000072}
}
```

## 📄 许可证

[MIT License](LICENSE)

## 🔗 相关工作

- [TorDet](https://github.com/Tornado-AI/TorDet) — 基于雷达的龙卷风检测
- [DeepTornado-Benchmark](https://github.com/Tornado-AI/DeepTornado-Benchmark) — 龙卷风雷达特征数据集
