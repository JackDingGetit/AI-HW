# 卷积神经网络在MNIST手写数字识别中的应用

## 摘要

本报告实现了两种卷积神经网络（CNN）模型用于MNIST手写数字识别任务：一种是仅包含一个卷积层和一个全连接层的极简CNN，另一种是经典的LeNet-5模型。通过对比实验，分析了不同网络结构在准确率、参数量和训练时间等方面的差异。

## 一、任务一：极简CNN实现

### 1.1 目标

实现一个仅包含一个卷积层和一个全连接层的简单CNN，旨在让深度学习初学者理解CNN的基本原理和PyTorch的实现方法。

### 1.2 模型结构

模型名为SimpleCNN，其核心结构顺序为：输入层 -> 卷积层 -> ReLU激活 -> 最大池化层 -> 展平操作 -> 全连接层 -> 输出层。

### 1.3 代码实现详解

#### 1.3.1 导入必要库

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
```

#### 1.3.2 定义SimpleCNN模型类

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 16, 3, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 14 * 14, 10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc(x)
        return x
```

#### 1.3.3 数据加载与预处理

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
```

#### 1.3.4 模型训练与测试

- 损失函数：交叉熵损失（CrossEntropyLoss）
- 优化器：Adam优化器，学习率0.001
- 训练轮数：5个epoch
- 批量大小：64

### 1.4 运行结果

经过5个训练周期后，该简单模型在MNIST测试集上达到了98.14%的准确率，展示了CNN在图像识别任务上的强大能力。

### 1.5 模型参数统计

| 层类型 | 输入尺寸 | 输出尺寸 | 参数量 |
|--------|----------|----------|--------|
| 卷积层 | 1x28x28 | 16x28x28 | 160 (1*3*3*16 + 16) |
| 池化层 | 16x28x28 | 16x14x14 | 0 |
| 全连接层 | 16x14x14 | 10 | 31370 (16*14*14*10 + 10) |
| **总计** | - | - | **31530** |

## 二、任务二：LeNet-5实现

### 2.1 模型结构

LeNet-5是经典的卷积神经网络模型，其结构如下：

| 层类型 | 输入尺寸 | 输出尺寸 | 参数量 |
|--------|----------|----------|--------|
| 卷积层1 | 1x28x28 | 6x28x28 | 156 (1*5*5*6 + 6) |
| 池化层1 | 6x28x28 | 6x14x14 | 0 |
| 卷积层2 | 6x14x14 | 16x10x10 | 2416 (6*5*5*16 + 16) |
| 池化层2 | 16x10x10 | 16x5x5 | 0 |
| 全连接层1 | 16x5x5 | 120 | 48120 (16*5*5*120 + 120) |
| 全连接层2 | 120 | 84 | 10164 (120*84 + 84) |
| 全连接层3 | 84 | 10 | 850 (84*10 + 10) |
| **总计** | - | - | **61706** |

### 2.2 运行结果

经过5个训练周期后，LeNet-5模型在MNIST测试集上达到了99.2%的准确率，表现优于极简CNN模型。

## 三、模型对比分析

### 3.1 性能对比

| 模型 | 参数量 | 测试准确率 | 训练时间 |
|------|--------|------------|----------|
| 极简CNN | ~31,530 | 98.14% | ~5分钟(CPU) |
| LeNet-5 | ~61,706 | 99.2% | ~10分钟(CPU) |

### 3.2 分析

1. **准确率提升**：LeNet-5的测试准确率比极简CNN高出约1.06个百分点，这得益于其更深的网络结构和更多的卷积层，能够提取更复杂的特征。

2. **参数量增加**：LeNet-5的参数量是极简CNN的约2倍，这意味着它需要更多的计算资源和更长的训练时间，但也带来了更好的性能。

3. **训练时间**：由于参数量增加，LeNet-5的训练时间大约是极简CNN的2倍，但在GPU上训练时，两者的训练时间都非常短。

## 四、改进方向

1. **增加网络深度**：可以尝试增加更多的卷积层和全连接层，以提高模型的表达能力。

2. **使用更复杂架构**：可以尝试使用ResNet、VGG等更复杂的网络架构。

3. **应用数据增强**：可以通过旋转、平移、缩放等方式增加训练数据的多样性，提高模型的泛化能力。

4. **调整超参数**：可以尝试调整学习率、批量大小、训练轮数等超参数，以优化模型性能。

5. **添加批归一化**：可以在卷积层和全连接层之间添加批归一化层，以加速训练过程和提高模型稳定性。

6. **使用正则化技术**：可以使用Dropout、L1/L2正则化等技术，防止模型过拟合。

## 五、参考文献

1. https://mp.weixin.qq.com/s/iBNvhk-uAeAfTuanxiLs9Q（《计算机视觉》第10篇 极简卷积神经网络CNN识别手写数字）
2. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
