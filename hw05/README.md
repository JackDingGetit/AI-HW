# AI-HW hw05: 卷积神经网络在MNIST上的应用

## 目录结构

```
hw05/
├── work1/                 # 任务一：极简CNN实现
│   ├── code1.py           # 极简CNN训练与测试代码
│   ├── mnist_samples.png  # MNIST样本可视化
│   ├── predictions.png    # 预测结果可视化
│   ├── simple_cnn_mnist.pth  # 训练好的极简CNN模型
│   └── training_loss.png  # 训练损失曲线
├── work2/                 # 任务二：LeNet-5实现
│   ├── code2.py           # LeNet-5训练与测试代码
│   ├── comparison_analysis.md  # 模型对比分析
│   ├── lenet5_mnist.pth   # 训练好的LeNet-5模型
│   ├── predictions.png    # 预测结果可视化
│   └── training_loss.png  # 训练损失曲线
├── requirements.txt       # 依赖环境配置
└── README.md              # 项目说明文档
```

## 依赖与环境

### Python版本
- Python 3.8+ (推荐使用Python 3.9或3.10)

### 安装依赖

```bash
pip install -r requirements.txt
```

## 一键训练与评估

### 任务一：极简CNN

```bash
cd work1
python code1.py
```

### 任务二：LeNet-5

```bash
cd work2
python code2.py
```

## 数据说明

- 代码会自动下载MNIST数据集到`./data`目录
- 数据集大小约为11MB，包含60,000张训练图像和10,000张测试图像
- 如果自动下载失败，可以手动下载：
  1. 访问MNIST官方网站：http://yann.lecun.com/exdb/mnist/
  2. 下载四个文件：train-images-idx3-ubyte.gz、train-labels-idx1-ubyte.gz、t10k-images-idx3-ubyte.gz、t10k-labels-idx1-ubyte.gz
  3. 解压后放在`./data/MNIST/raw`目录下

## 输出结果

运行代码后会生成以下文件：

1. `*.pth`：训练好的模型参数文件
2. `training_loss.png`：训练损失曲线
3. `predictions.png`：模型预测结果可视化
4. `mnist_samples.png`：MNIST数据集样本可视化（仅任务一）

## 模型对比

| 模型 | 参数量 | 测试准确率 | 训练时间 |
|------|--------|------------|----------|
| 极简CNN | ~31,530 | ~98.5% | ~5分钟(CPU) |
| LeNet-5 | ~61,706 | ~99.2% | ~10分钟(CPU) |

## 注意事项

1. 如果有GPU可用，代码会自动使用GPU加速训练
2. 训练时间取决于硬件配置，GPU训练会比CPU快很多
3. 可以通过修改代码中的`epochs`参数调整训练轮数
4. 可以通过修改代码中的`batch_size`参数调整批量大小
