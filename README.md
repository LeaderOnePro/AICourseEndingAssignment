# 高性能 MNIST 手写数字识别项目

本项目旨在通过 PyTorch 实现一个高性能的深度卷积神经网络 (CNN)，用于解决经典的 MNIST 手写数字识别任务。通过综合运用多种现代深度学习优化技术，本项目成功将模型在标准测试集上的准确率提升至 **99.65%**，超越了 99.5% 的设计目标。

## 主要特性与技术栈

-   **高性能模型**: 最终测试准确率达到 **99.65%**。
-   **现代网络架构**: 构建了一个包含三个卷积块、批量归一化 (BatchNorm)、Dropout 和自适应平均池化层的深度 CNN。
-   **高级数据增强**: 采用随机仿射变换 (Random Affine) 与随机擦除 (Random Erasing) 相结合的复合数据增强策略，提升模型鲁棒性。
-   **高级训练策略**:
    -   **标签平滑 (Label Smoothing)**: 代替传统交叉熵损失，提升模型泛化能力。
    -   **AdamW 优化器**: 采用性能更优的 AdamW 优化器。
    -   **分层学习率 (Discriminative LR)**: 为网络不同部分设置不同学习率，加速有效收敛。
    -   **OneCycleLR 调度器**: 使用先进的学习率调度策略，帮助模型跳出局部最优。
    -   **自动混合精度 (AMP)**: 利用 `torch.cuda.amp` 加速训练，降低显存占用。

## 文件结构

```
.
├── recognize_digits.py     # 主训练脚本
├── requirements.txt        # 项目基础依赖
├── best_simple_model.pth   # 训练过程中最佳模型权重
├── final_simple_model.pth  # 最终模型权重
├── result.png              # 最终手写数字识别结果图
├── 结课作业-多选项2.md      # 作业要求说明
├── 结课报告.md             # Markdown 格式的结课报告
├── 结课报告.tex            # LaTeX 格式的结课报告
├── data/                     # MNIST 数据集目录
└── README.md               # 本说明文件
```

---

## 环境要求与安装指南

本项目基于 Python 3.8+ 和 PyTorch。请根据您的硬件情况（仅使用 CPU 或使用 NVIDIA GPU）选择相应的安装指南。

### 1. 克隆项目

```bash
git clone https://github.com/LeaderOnePro/AICourseEndingAssignment.git
cd AICourseEndingAssignment
```

### 2. 安装基础依赖

首先，安装除 PyTorch 之外的通用依赖包：

```bash
pip install -r requirements.txt
```

### 3. 安装 PyTorch

**PyTorch 的安装与您的硬件（CPU/GPU）直接相关，请选择以下一种方式进行安装。**

#### **选项 A：仅使用 CPU**

如果您的电脑没有支持 CUDA 的 NVIDIA 显卡，或者您只想使用 CPU 进行训练，请执行以下命令：

```bash
# 访问 PyTorch 官网 (https://pytorch.org/get-started/locally/) 获取适合您系统的最新命令
# 以下是一个示例命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
> **注意**: 使用 CPU 训练会非常慢，完成30个周期的训练可能需要数小时。

#### **选项 B：使用 GPU (推荐)**

为了复现项目的高性能并进行高效训练，强烈建议使用支持 CUDA 的 NVIDIA GPU。

**第一步：检查您的 NVIDIA 驱动和 CUDA 版本**

在您的终端中运行 `nvidia-smi` 命令，查看所支持的最高 CUDA 版本。

**第二步：安装与 CUDA 兼容的 PyTorch 版本**

访问 [PyTorch 官网](https://pytorch.org/get-started/locally/)，根据您的操作系统和 `nvidia-smi` 显示的 CUDA 版本，选择对应的 PyTorch 安装命令。

例如，如果您的 CUDA 版本是 11.8，安装命令如下：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

如果您的 CUDA 版本是 12.1 或更高，安装命令如下：
```bash
pip install torch torchvision torchaudio
```

---

## 如何运行

确保所有依赖项已正确安装后，直接运行主脚本即可开始训练：

```bash
python recognize_digits.py
```

脚本会自动检测可用的设备（GPU 或 CPU），下载 MNIST 数据集（如果尚未下载），并开始训练过程。训练日志会实时打印在控制台，包含每个周期的损失、准确率等信息。训练过程中准确率最高的模型将被保存为 `best_simple_model.pth`，最终模型将被保存为 `final_simple_model.pth`。

## 项目特点
- 使用改进的CNN架构，相比传统LeNet-5有更好的特征提取能力
- 实现了99.6+ %的测试准确率（MNIST测试集）
- 使用PyTorch框架实现，代码结构清晰，易于理解和扩展
- 包含数据增强、学习率调度等训练技巧

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

最终模型 `ImprovedNet` 是一个包含三个卷积块的深度卷积神经网络，具体结构如下：

```
ImprovedNet(
  (conv_layers): Sequential(
    # Block 1
    (0): Conv2d(1, 32, kernel_size=3, padding=1), ReLU, BatchNorm2d
    (1): Conv2d(32, 32, kernel_size=3, padding=1), ReLU, BatchNorm2d
    (2): MaxPool2d(kernel_size=2), Dropout(0.25)
    
    # Block 2
    (3): Conv2d(32, 64, kernel_size=3, padding=1), ReLU, BatchNorm2d
    (4): Conv2d(64, 64, kernel_size=3, padding=1), ReLU, BatchNorm2d
    (5): MaxPool2d(kernel_size=2), Dropout(0.25)

    # Block 3
    (6): Conv2d(64, 128, kernel_size=3, padding=1), ReLU, BatchNorm2d
    (7): Conv2d(128, 128, kernel_size=3, padding=1), ReLU, BatchNorm2d
    (8): AdaptiveAvgPool2d(1), Dropout(0.25)
  )
  (classifier): Sequential(
    (0): Linear(128 -> 256), ReLU, Dropout(0.5)
    (1): Linear(256 -> 128), ReLU, Dropout(0.3)
    (2): Linear(128 -> 10)
  )
)
```

## 性能指标
- 测试准确率: **99.65%**
- 训练设备: CPU/GPU (自动检测)
