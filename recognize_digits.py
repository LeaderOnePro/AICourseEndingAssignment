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
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(), # 将图片转换为Tensor
        transforms.Normalize((0.1307,), (0.3081,)) # 标准化
    ])
    
    # 下载/加载训练集和测试集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 3. 训练函数
def train_model(model, train_loader, optimizer, criterion, device, epoch):
    model.train() # 设置为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 清空梯度
        output = model(data) # 前向传播
        loss = criterion(output, target) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新权重
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
    # 设置设备 (使用GPU如果可用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 超参数
    batch_size = 64
    epochs = 15 # 增加训练周期以适应更复杂的模型
    learning_rate = 0.001 # 为Adam优化器设置一个合适的学习率
    
    # 准备数据
    train_loader, test_loader = prepare_data(batch_size)
    
    # 初始化模型、损失函数和优化器
    model = ImprovedNet().to(device) # <- 使用我们改进后的模型
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # <- 使用Adam优化器
    
    # 训练和测试
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train_model(model, train_loader, optimizer, criterion, device, epoch)
        accuracy = test_model(model, test_loader, criterion, device)
        best_accuracy = max(accuracy, best_accuracy)
        
    print("Improved model training finished.")
    print(f"Highest accuracy achieved: {best_accuracy:.2f}%")


if __name__ == '__main__':
    main() 