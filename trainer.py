# -*- coding: utf-8 -*-
"""
TorViNet 训练脚本

论文: TorViNet: A Spatiotemporal Deep Learning Network for Tornado Detection
      in User-Captured Social Media Videos
期刊: Expert Systems With Applications, 2026
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.torvinet import TorViNet
from utils.my_dataset import MyDataset
from config import config, print_config

# 设置 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_data_loaders(batch_size):
    """
    创建数据加载器

    Args:
        batch_size: 批次大小

    Returns:
        train_loader, valid_loader, test_loader
    """
    train_data = MyDataset(split='train')
    valid_data = MyDataset(split='val')
    test_data = MyDataset(split='test')

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    return train_loader, valid_loader, test_loader


class Trainer:
    """
    TorViNet 模型训练器

    支持:
        - 预训练模型加载
        - 学习率调度
        - 训练/验证损失记录
        - 模型检查点保存
    """

    def __init__(self, epochs, init_lr, batch_size, model_name,
                 is_pretrain=False, pretrain_epoch=99,
                 in_channel=24, classes_num=2):
        """
        初始化训练器

        Args:
            epochs: 训练轮数
            init_lr: 初始学习率
            batch_size: 批次大小
            model_name: 模型名称
            is_pretrain: 是否使用预训练模型
            pretrain_epoch: 预训练模型轮数
            in_channel: 输入通道数
            classes_num: 类别数
        """
        print(f'训练参数: epochs={epochs}, init_lr={init_lr}, batch_size={batch_size}, '
              f'model_name={model_name}, is_pretrain={is_pretrain}')
        print(f'设备: {device}')

        self.epochs = epochs
        self.init_lr = init_lr
        self.train_loader, self.valid_loader, self.test_loader = create_data_loaders(batch_size)

        self.model_name = model_name
        self.in_channel = in_channel
        self.classes_num = classes_num
        self.is_pretrain = is_pretrain
        self.pretrain_epoch = pretrain_epoch

        self.train_loss_all = []
        self.valid_loss_all = []

        self.model = self._create_model()
        self.train_criterion, self.valid_criterion, self.optimizer, self.scheduler = self._setup_optimizer()

    def _create_model(self):
        """创建并加载模型"""
        if self.model_name == 'tvm':
            model = TorViNet()
        else:
            raise ValueError(f'未知的模型名称: {self.model_name}')

        model.to(device)

        # 加载预训练权重
        if self.is_pretrain:
            checkpoint_path = os.path.join(
                config['checkpoint_path'],
                self.model_name,
                f'{self.model_name}_{self.pretrain_epoch}.pth'
            )
            model.load_state_dict(torch.load(checkpoint_path))

            # 加载训练损失历史
            loss_file = os.path.join(
                config['checkpoint_path'],
                self.model_name,
                f'{self.model_name}_loss.txt'
            )
            with open(loss_file, 'r') as f:
                lines = f.readlines()
                for i in range(1, self.pretrain_epoch + 1):
                    self.train_loss_all.append(float(lines[i]))
                for i in range(len(lines) // 2 + 1, len(lines) // 2 + self.pretrain_epoch + 1):
                    self.valid_loss_all.append(float(lines[i]))

            print(f'已加载预训练模型: epoch {self.pretrain_epoch}')

        return model

    def _setup_optimizer(self):
        """
        设置损失函数和优化器

        使用 BCEWithLogitsLoss 处理类别不平衡问题
        pos_weight: 正负样本比例约为 3100:1015 ≈ 3.05
        """
        # 损失函数 - 使用 pos_weight 处理类别不平衡
        train_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]).to(device))
        valid_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]).to(device))

        # 优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=self.init_lr)

        # 学习率调度 - 余弦退火
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )

        # 如果使用预训练模型，跳过预训练轮数的学习率调整
        if self.is_pretrain:
            for _ in range(self.pretrain_epoch):
                scheduler.step()

        return train_criterion, valid_criterion, optimizer, scheduler

    def train(self):
        """执行训练循环"""
        loss_all = []
        start_epoch = self.pretrain_epoch if self.is_pretrain else 0

        for epoch in range(start_epoch, self.epochs):
            # ===== 训练阶段 =====
            self.model.train()
            train_loss = 0

            for inputs, labels, _ in tqdm(self.train_loader, desc=f'Epoch {epoch}'):
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs = outputs.squeeze(axis=1)

                loss = self.train_criterion(outputs, labels.float())
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                loss_all.append(loss.item())

                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'  batch_loss: {loss.item():.4f}, lr: {current_lr:.2e}')

                # 释放显存
                del inputs, labels, outputs, loss

            avg_train_loss = train_loss / len(self.train_loader)
            self.train_loss_all.append(avg_train_loss)
            self.scheduler.step()

            print(f'Epoch {epoch} - train_loss: {avg_train_loss:.4f}')

            # ===== 验证阶段 =====
            self.model.eval()
            valid_loss = 0

            with torch.no_grad():
                for inputs, labels, _ in tqdm(self.valid_loader, desc='Validating'):
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self.model(inputs)
                    outputs = outputs.squeeze(axis=1)

                    loss = self.valid_criterion(outputs, labels.float())
                    valid_loss += loss.item()

                    del inputs, labels, outputs, loss

            avg_valid_loss = valid_loss / len(self.valid_loader)
            self.valid_loss_all.append(avg_valid_loss)

            print(f'Epoch {epoch} - valid_loss: {avg_valid_loss:.4f}')

            # ===== 保存检查点 =====
            self._save_checkpoint(epoch, loss_all)

    def _save_checkpoint(self, epoch, loss_all):
        """保存模型检查点和损失曲线"""
        checkpoint_dir = os.path.join(config['checkpoint_path'], self.model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 保存模型权重
        model_path = os.path.join(
            checkpoint_dir,
            f'{self.model_name}_{str(epoch).rjust(2, "0")}.pth'
        )
        torch.save(self.model.state_dict(), model_path)

        # 保存训练损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(loss_all)
        plt.title('Training Loss (per batch)')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(checkpoint_dir, f'{self.model_name}_train_loss.png'))
        plt.close()

        # 保存 epoch 级损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_all, label='Train Loss')
        plt.plot(self.valid_loss_all, label='Valid Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(checkpoint_dir, f'{self.model_name}_loss.png'))
        plt.close()

        # 保存损失数值
        loss_path = os.path.join(checkpoint_dir, f'{self.model_name}_loss.txt')
        with open(loss_path, 'w') as f:
            f.write('train_loss\n')
            for loss in self.train_loss_all:
                f.write(f'{loss}\n')
            f.write('valid_loss\n')
            for loss in self.valid_loss_all:
                f.write(f'{loss}\n')


if __name__ == '__main__':
    print_config()

    # 训练配置
    train_list = [
        ['tvm_00', 32],  # [模型名称, 批次大小]
    ]

    for model_name, batch_size in train_list:
        trainer = Trainer(
            epochs=50,
            init_lr=0.0001,
            batch_size=batch_size,
            model_name=model_name,
            is_pretrain=False,
            pretrain_epoch=45,
            in_channel=3,
            classes_num=1
        )
        trainer.train()
