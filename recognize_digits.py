# -*- coding: utf-8 -*-

"""
@software: PyCharm
@file: recognize_digits.py
@time: 2024/5/24 10:00
@author: B. AI
""" 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from muon import MuonWithAuxAdam
import torch.distributed as dist
import os
from torch.optim.lr_scheduler import OneCycleLR

# 1. 定义一个更强大的CNN模型 (替代 LeNet-5)
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        # 卷积层部分
        # 初始输入: 1x28x28
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # 28x28 -> 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 第二个卷积块
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # 32x28x28 -> 64x28x28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x28x28 -> 64x14x14
            nn.Dropout(0.25)
        )
        
        # 全连接层部分
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 14 * 14, 128), # 展平后为 12544
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 14 * 14) # 展平操作
        x = self.fc_layers(x)
        return x

# 2. 准备数据
def prepare_data(batch_size=64):
    # 为训练集定义包含数据增强的预处理
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), # 旋转, 平移, 缩放
        transforms.ToTensor(), # 将图片转换为Tensor
        transforms.Normalize((0.1307,), (0.3081,)) # 标准化
    ])
    
    # 为测试集定义不包含数据增强的预处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 下载/加载训练集和测试集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform) # 使用带增强的transform
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform) # 使用不带增强的transform
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 3. 训练函数
def train_model(model, train_loader, optimizer, criterion, device, epoch, scheduler):
    model.train() # 设置为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 清空梯度
        output = model(data) # 前向传播
        loss = criterion(output, target) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新权重
        scheduler.step() # <- 在每个批次后更新学习率
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 4. 测试函数
def test_model(model, test_loader, criterion, device):
    model.eval() # 设置为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 在评估时不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # 累加批次损失
            pred = output.argmax(dim=1, keepdim=True) # 获取最大概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return accuracy


# 5. 主函数
def main():
    # --- 修复Muon分布式环境错误 ---
    # 模拟一个单进程的分布式环境以满足Muon优化器的要求
    os.environ['MASTER_ADDR'] = '127.0.0.1' # 使用 127.0.0.1 避免 'localhost' 解析问题
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

    # 设置设备 (使用GPU如果可用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 超参数
    batch_size = 64
    epochs = 30 # 保持30个周期进行充分训练
    
    # 准备数据
    train_loader, test_loader = prepare_data(batch_size)
    
    # 初始化模型、损失函数和优化器
    model = ImprovedNet().to(device)
    criterion = nn.CrossEntropyLoss()

    # --- 最终修复：将 Muon 仅应用于全连接层 ---
    # 这是一个更安全、更稳定的策略，以避开 Muon 在卷积层上可能存在的 bug
    
    muon_params = []
    adam_params = []

    # 使用 named_parameters() 来根据层名和参数维度进行分组
    for name, p in model.named_parameters():
        if 'fc_layers' in name and p.ndim == 2:
            # 如果参数在 'fc_layers' 中且是权重矩阵 (2D), 则使用 Muon
            print(f"Applying Muon to: {name}")
            muon_params.append(p)
        else:
            # 其他所有参数 (所有卷积层, 所有偏置项, 所有BN层) 使用 AdamW
            adam_params.append(p)

    # 2. 为两组参数分别设置超参数
    param_groups = [
        {'params': muon_params, 'use_muon': True, 'lr': 0.02},
        {'params': adam_params, 'use_muon': False, 'lr': 3e-4, 'betas': (0.9, 0.95)}
    ]

    # 3. 初始化 MuonWithAuxAdam 优化器
    optimizer = MuonWithAuxAdam(param_groups)
    
    # 4. 添加 OneCycleLR 学习率调度器
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[0.02, 3e-4], # 对应 Muon 和 AdamW 两组的最高学习率
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )

    # 训练和测试
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train_model(model, train_loader, optimizer, criterion, device, epoch, scheduler)
        accuracy = test_model(model, test_loader, criterion, device)
        best_accuracy = max(accuracy, best_accuracy)
        
    print("Improved model training finished.")
    print(f"Highest accuracy achieved: {best_accuracy:.2f}%")


if __name__ == '__main__':
    main() 