# -*- coding: utf-8 -*-
"""
@software: PyCharm
@file: recognize_digits.py
@time: 2024/5/24 10:00
@author: B. AI
@description: 简化版手写数字识别 - 不依赖muon包
""" 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

# 1. 改进的CNN模型
class ImprovedNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedNet, self).__init__()
        
        # 卷积层部分
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.25),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2. 标签平滑损失
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes=10, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=1), dim=1))

# 3. 数据准备
def prepare_data(batch_size=128):
    # 训练时数据增强
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.12, 0.12), scale=(0.85, 1.15), shear=8),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.25), ratio=(0.3, 3.3))
    ])
    
    # 测试时标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

# 4. 训练函数
def train_epoch(model, train_loader, optimizer, criterion, device, epoch, scheduler, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 200 == 0:
            print(f'Epoch {epoch}: [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}, Acc: {100. * correct / total:.2f}%')

# 5. 测试函数
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                output = model(data)
            
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss/len(test_loader):.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.4f}%)\n')
    return accuracy

# 6. 主函数
def main():
    # 分布式设置
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo', rank=0, world_size=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 超参数
    batch_size = 128
    epochs = 30
    
    # 准备数据
    train_loader, test_loader = prepare_data(batch_size)
    
    # 初始化模型
    model = ImprovedNet().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = LabelSmoothingLoss(smoothing=0.1)
    
    # 使用AdamW优化器，为不同层设置不同学习率
    classifier_params = []
    feature_params = []
    
    for name, p in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(p)
        else:
            feature_params.append(p)
    
    optimizer = optim.AdamW([
        {'params': feature_params, 'lr': 5e-4, 'weight_decay': 1e-4},
        {'params': classifier_params, 'lr': 1e-3, 'weight_decay': 1e-4}
    ], betas=(0.9, 0.999))
    
    # 学习率调度器
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[5e-4, 1e-3],
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # 训练循环
    best_accuracy = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        train_epoch(model, train_loader, optimizer, criterion, device, epoch, scheduler, scaler)
        
        # 测试
        if epoch % 3 == 0 or epoch > epochs - 5:
            accuracy = test_model(model, test_loader, criterion, device)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'best_simple_model.pth')
                print(f"New best accuracy: {best_accuracy:.4f}%")
    
    print(f"\n=== Final Results ===")
    print(f"Best accuracy achieved: {best_accuracy:.4f}%")
    print(f"Target 99.5% achieved: {'YES' if best_accuracy >= 99.5 else 'NO'}")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'final_simple_model.pth')
    print("Model saved!")

if __name__ == '__main__':
    main()
