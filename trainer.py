# -*-coding: Utf-8 -*-
# Author: CHJ
# Time: 2024/7/24 下午8:31


"""
训练脚本
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.TSTVM_00 import TorViNet
from utils.MyDataset import MyDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def data_loader(batch_size):

    # 初始化数据集
    train_data = MyDataset(split='train')
    valid_data = MyDataset(split='val')
    test_data = MyDataset(split='test')

    # 数据加载器
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

# 计算评价指标
def metric(outputs, labels):
    return torch.sum(torch.argmax(outputs, dim=1) == labels).item() / len(labels)

class Trainer:
    def __init__(self, epochs, init_lr, batch_size,
                 model_name, is_pretrain=False, pretrain_epoch=99,
                 in_channel=24, classes_num=2):

        print(f'epochs:{epochs}, init_lr:{init_lr}, batch_size:{batch_size}, model_name:{model_name}, '
              f'is_pretrain:{is_pretrain}, in_channel:{in_channel}, classes_num:{classes_num}')

        self.epochs = epochs
        self.init_lr = init_lr

        self.train_loader, self.valid_loader, self.test_loader = data_loader(batch_size)

        self.model_name = model_name
        self.in_channel = in_channel
        self.classes_num = classes_num
        self.is_pretrain = is_pretrain
        self.pretrain_epoch = pretrain_epoch
        self.train_loss_all = []
        self.valid_loss_all = []
        self.model = self.create_model()

        self.train_criterion, self.valid_criterion, self.optimizer, self.scheduler = self.criterion_optimizer()

    def create_model(self):

        # 创建模型
        if self.model_name == 'tvm':
            model = TorViNet()
        else:
            raise ValueError('模型名称错误')

        model.to(device)

        # 加载预训练模型
        if self.is_pretrain:
            model.load_state_dict(torch.load(f'{config["checkpoint_path"]}/'
                                             f'{self.model_name}/{self.model_name}_{self.pretrain_epoch}.pth'))
            with open(f'{config["checkpoint_path"]}/{self.model_name}/{self.model_name}_loss.txt', 'r') as f:
                lines = f.readlines()
                for i in range(1, self.pretrain_epoch+1):
                    self.train_loss_all.append(float(lines[i]))
                for i in range(len(lines)//2+1, len(lines)//2+self.pretrain_epoch+1):
                    self.valid_loss_all.append(float(lines[i]))
            print(f'加载{self.model_name}预训练模型')

        return model

    # 损失函数、优化器
    def criterion_optimizer(self):
        # 损失函数
        # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device))
        # 二分类，正样本数量为1015，负样本数量为3100，pos_weight=3100/1015
        train_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]).to(device))
        valid_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]).to(device))
        # 优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=self.init_lr)
        # 学习率调整
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        if self.is_pretrain:
            for _ in range(self.pretrain_epoch):
                scheduler.step()

        return train_criterion, valid_criterion, optimizer, scheduler

    def train(self):

        loss_all = []
        start_epoch = self.pretrain_epoch if self.is_pretrain else 0
        for epoch in range(start_epoch, self.epochs):

            self.model.train()
            train_loss = 0
            for inputs, labels, _ in tqdm(self.train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs = outputs.squeeze(axis=1)
                loss = self.train_criterion(outputs, labels.float())
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                loss_all.append(loss.item())
                print(f'epoch:{epoch}, train_loss:{loss.item()}, lr:{self.optimizer.param_groups[0]["lr"]}')

                # 释放显存
                del inputs, labels, outputs, loss
            self.train_loss_all.append(train_loss / len(self.train_loader))
            self.scheduler.step()
            print(f'epoch:{epoch}, train_loss:{train_loss / len(self.train_loader)}')

            # 验证集
            self.model.eval()
            valid_loss = 0
            with torch.no_grad():
                for inputs, labels, _ in tqdm(self.valid_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)
                    outputs = outputs.squeeze(axis=1)
                    loss = self.valid_criterion(outputs, labels.float())
                    valid_loss += loss.item()
                    # 释放显存
                    del inputs, labels, outputs, loss
                self.valid_loss_all.append(valid_loss / len(self.valid_loader))
                print(f'epoch:{epoch}, valid_loss:{valid_loss / len(self.valid_loader)}')

            # 保存模型
            if not os.path.exists(f'{config["checkpoint_path"]}/{self.model_name}'):
                os.makedirs(f'{config["checkpoint_path"]}/{self.model_name}')
            torch.save(self.model.state_dict(),
                       f'{config["checkpoint_path"]}/'
                       f'{self.model_name}/{self.model_name}_{str(epoch).rjust(2, "0")}.pth')

            plt.plot(loss_all)
            plt.savefig(f'{config["checkpoint_path"]}/{self.model_name}/{self.model_name}_train_loss.png')
            plt.close()

            # 画loss曲线
            plt.plot(self.train_loss_all, label='train_loss')
            plt.plot(self.valid_loss_all, label='valid_loss')
            plt.legend()
            plt.savefig(f'{config["checkpoint_path"]}/{self.model_name}/{self.model_name}_loss.png')
            plt.close()

            with open(f'{config["checkpoint_path"]}/{self.model_name}/{self.model_name}_loss.txt', 'w') as f:
                f.write('train_loss\n')
                for loss in self.train_loss_all:
                    f.write(f'{loss}\n')
                f.write('valid_loss\n')
                for loss in self.valid_loss_all:
                    f.write(f'{loss}\n')


if __name__ == '__main__':

    train_lsit = [['tvm_00', 32]]

    for model_name, batch_size in train_lsit:
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



