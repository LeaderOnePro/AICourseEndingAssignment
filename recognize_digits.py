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

# 1. 定义 LeNet-5 网络模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层部分
        self.conv_layers = nn.Sequential(
            # 输入: 1x28x28
            # 第一个卷积层: 1个输入通道, 6个输出通道, 5x5的卷积核
            # (28-5)/1 + 1 = 24. 输出: 6x24x24
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2), # Padding=2使得输出尺寸仍为28x28
            nn.ReLU(),
            # 第一个池化层: 2x2的最大池化
            # 28 / 2 = 14. 输出: 6x14x14
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积层: 6个输入通道, 16个输出通道, 5x5的卷积核
            # (14-5)/1 + 1 = 10. 输出: 16x10x10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            # 第二个池化层: 2x2的最大池化
            # 10 / 2 = 5. 输出: 16x5x5
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 全连接层部分
        self.fc_layers = nn.Sequential(
            # 将16x5x5的张量展平为 16*5*5 = 400
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            # 输出层: 84个输入, 10个输出 (对应0-9十个数字)
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 16 * 5 * 5) # 展平操作
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
    epochs = 10 # 我们先训练10个周期看看效果
    learning_rate = 0.01
    
    # 准备数据
    train_loader, test_loader = prepare_data(batch_size)
    
    # 初始化模型、损失函数和优化器
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # 训练和测试
    for epoch in range(1, epochs + 1):
        train_model(model, train_loader, optimizer, criterion, device, epoch)
        test_model(model, test_loader, criterion, device)
        
    print("Baseline model training finished.")


if __name__ == '__main__':
    main() 