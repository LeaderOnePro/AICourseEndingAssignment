# Handwritten Digit Recognition using Deep Learning

## 项目背景
本项目是基于深度学习的MNIST手写数字识别系统，是人工智能课程的结课作业。通过改进的CNN网络架构和训练技巧，成功地将模型在测试集上的准确率从基准的98.35%提升到了99.6+ %。

## 项目特点
- 使用改进的CNN架构，相比传统LeNet-5有更好的特征提取能力
- 实现了99.6+ %的测试准确率（MNIST测试集）
- 使用PyTorch框架实现，代码结构清晰，易于理解和扩展
- 包含数据增强、学习率调度等训练技巧

## 环境要求
- Python 3.7+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- 其他依赖见 `requirements.txt`

## 安装
1. 克隆本仓库
   ```bash
   git clone [your-repository-url]
   cd CourseEndingAssignment
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

## 数据集
本项目使用标准的MNIST手写数字数据集，包含60,000张训练图像和10,000张测试图像。程序会自动下载数据集到`data/`目录。

## 使用方法
1. 训练模型
   ```bash
   python recognize_digits.py
   ```

2. 训练参数（可在代码中修改）
   - 批量大小 (batch_size)
   - 学习率 (learning_rate)
   - 训练轮数 (epochs)
   - 是否使用数据增强

## 模型架构
```
ImprovedNet(
  (conv_layers): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (7): Dropout(p=0.25, inplace=False)
  )
  (fc_layers): Sequential(
    (0): Linear(in_features=12544, out_features=128, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=128, out_features=10, bias=True)
  )
)
```

## 性能指标
- 测试准确率: 99.6+ %
- 训练设备: CPU/GPU (自动检测)

## 文件结构
```
CourseEndingAssignment/
├── data/               # 数据集目录
├── recognize_digits.py  # 主程序
├── requirements.txt     # 项目依赖
└── README.md           # 项目说明
```

## 许可证
MIT License
