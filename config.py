# -*- coding: utf-8 -*-
"""
TorViNet 项目配置文件

论文: TorViNet: A Spatiotemporal Deep Learning Network for Tornado Detection
      in User-Captured Social Media Videos
期刊: Expert Systems With Applications, 2026
"""

import os
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# ==================== 路径配置 ====================
# 可以通过环境变量覆盖默认路径
DATASETS_PATH = os.getenv('TORVINET_DATASETS_PATH', '/mnt/4T/chenhj/datasets/')
CHECKPOINT_PATH = os.getenv('TORVINET_CHECKPOINT_PATH', '/mnt/4T/chenhj/checkpoints/TornadoCV/')

# 数据集具体路径
DATASET_ROOT = os.path.join(DATASETS_PATH, 'my_dataset', 'new_dataset')
VIDEO_PATH = os.path.join(DATASET_ROOT, 'video')
EXTRACTION_PATH = os.path.join(DATASET_ROOT, 'extraction')

# 数据集划分文件
DATASET_SPLIT_FILE = os.path.join(DATASET_ROOT, 'dataset_split.json')
PRESENT_DATASET_SPLIT_FILE = os.path.join(DATASET_ROOT, 'present_dataset_split.json')

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    'input_channels': 3,        # 输入通道数 (RGB)
    'num_classes': 1,           # 输出类别数 (二分类)
    'input_frames': 64,         # 输入帧数
    'input_height': 224,        # 输入高度
    'input_width': 224,         # 输入宽度
    'key_frames': 8,            # 关键帧数量 (KFS 模块)
}

# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    'epochs': 50,               # 训练轮数
    'batch_size': 32,           # 批次大小
    'init_lr': 1e-4,            # 初始学习率
    'min_lr': 1e-6,             # 最小学习率
    'weight_decay': 1e-4,       # 权重衰减
    'pos_weight': 2.5,          # 正样本权重 (处理类别不平衡)
}

# ==================== 数据配置 ====================
DATA_CONFIG = {
    'num_workers': 4,           # 数据加载线程数
    'pin_memory': True,         # 是否使用锁页内存
    'train_shuffle': True,      # 训练时是否打乱数据
}

# ==================== 设备配置 ====================
DEVICE_CONFIG = {
    'cuda_device': '0',         # GPU 设备号
    'use_amp': False,           # 是否使用自动混合精度
}

# 组合配置 (保持向后兼容)
config = {
    'datasets_path': DATASETS_PATH,
    'checkpoint_path': CHECKPOINT_PATH,
    'dataset_root': DATASET_ROOT,
    'video_path': VIDEO_PATH,
    'extraction_path': EXTRACTION_PATH,
    'dataset_split_file': DATASET_SPLIT_FILE,
    'present_dataset_split_file': PRESENT_DATASET_SPLIT_FILE,
}


def get_config():
    """
    获取完整配置
    
    Returns:
        dict: 包含所有配置的字典
    """
    return {
        'paths': {
            'datasets_path': DATASETS_PATH,
            'checkpoint_path': CHECKPOINT_PATH,
            'dataset_root': DATASET_ROOT,
            'video_path': VIDEO_PATH,
            'extraction_path': EXTRACTION_PATH,
        },
        'model': MODEL_CONFIG,
        'train': TRAIN_CONFIG,
        'data': DATA_CONFIG,
        'device': DEVICE_CONFIG,
    }


def print_config():
    """打印当前配置"""
    print("=" * 50)
    print("TorViNet 配置信息")
    print("=" * 50)
    print(f"数据集路径: {DATASET_ROOT}")
    print(f"检查点路径: {CHECKPOINT_PATH}")
    print(f"输入尺寸: ({MODEL_CONFIG['input_channels']}, "
          f"{MODEL_CONFIG['input_frames']}, "
          f"{MODEL_CONFIG['input_height']}, "
          f"{MODEL_CONFIG['input_width']})")
    print(f"训练轮数: {TRAIN_CONFIG['epochs']}")
    print(f"批次大小: {TRAIN_CONFIG['batch_size']}")
    print(f"初始学习率: {TRAIN_CONFIG['init_lr']}")
    print("=" * 50)


if __name__ == '__main__':
    print_config()
