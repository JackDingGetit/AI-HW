# 胸部X光片肺炎检测项目

## 项目简介

本项目使用深度学习技术构建了胸部X光片分类模型，支持两种分类任务：
1. **二分类**：检测正常肺部 vs 肺炎肺部
2. **三分类**：检测正常肺部 vs 病毒性肺炎 vs 细菌性肺炎

通过卷积神经网络(CNN)对胸部X光片进行分析，实现对肺部健康状态的自动分类。

## 目录结构

```
hw07/
├── chest_xray/                    # 数据集目录
│   ├── train/                    # 训练集
│   │   ├── NORMAL/              # 正常肺部X光片
│   │   └── PNEUMONIA/           # 肺炎肺部X光片（包含virus/bacteria标记）
│   ├── test/                     # 测试集
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── val/                      # 验证集
│       ├── NORMAL/
│       └── PNEUMONIA/
├── train.py                      # 二分类训练脚本
├── train_3class.py               # 三分类训练脚本
├── build_model.py               # 模型构建脚本
├── preprocess_data.py           # 数据预处理脚本
├── evaluate_model.py            # 二分类模型评估脚本
├── evaluate_3class.py           # 三分类模型评估脚本
├── load_3class_data.py          # 三分类数据加载脚本
├── plot_curves.py               # 绘制训练曲线
├── count_images.py              # 统计图片数量
├── trained_model.keras          # 训练好的二分类模型
├── trained_model_3class.keras   # 训练好的三分类模型
├── training_history.pkl         # 二分类训练历史
├── training_history_3class.pkl  # 三分类训练历史
├── figures/                     # 图像输出目录
│   ├── training_curves.png     # 二分类训练曲线图
│   ├── confusion_matrix.png     # 二分类混淆矩阵图
│   ├── training_curves_3class.png  # 三分类训练曲线图
│   └── confusion_matrix_3class.png # 三分类混淆矩阵图
├── archive.zip                  # 原始数据集压缩文件
└── README.md                    # 项目说明文档
```

## 环境要求

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- PIL (Pillow)

## 数据集

本项目使用的是胸部X光片数据集（Kaggle Chest X-Ray Images），包含正常肺部和肺炎肺部的X光片。

### 二分类数据集统计

| 数据集 | NORMAL | PNEUMONIA | 总计 |
|--------|--------|-----------|------|
| 训练集 | 1341 | 3875 | 5216 |
| 测试集 | 234 | 390 | 624 |
| 验证集 | 8 | 8 | 16 |
| **总计** | **1583** | **4273** | **5856** |

### 三分类数据集统计

| 数据集 | Viral | Bacterial | Normal | 总计 |
|--------|-------|-----------|--------|------|
| 训练集 | 1345 | 2530 | 1341 | 5216 |
| 测试集 | 148 | 242 | 234 | 624 |
| 验证集 | 5 | 7 | 4 | 16 |
| **总计** | **1498** | **2779** | **1579** | **5856** |

图像尺寸统一处理为 150x150 像素。

## 模型架构

### 二分类模型

4 层卷积块的 CNN 模型：

1. **卷积块 1**：32 个 3x3 卷积核，ReLU 激活，最大池化
2. **卷积块 2**：64 个 3x3 卷积核，ReLU 激活，最大池化
3. **卷积块 3**：128 个 3x3 卷积核，ReLU 激活，最大池化
4. **卷积块 4**：128 个 3x3 卷积核，ReLU 激活，最大池化
5. **全连接层**：512 神经元 + Dropout(0.5)
6. **输出层**：1 神经元，Sigmoid 激活（二分类）

### 三分类模型

基于二分类模型的扩展：

1. **卷积块 1**：32 个 3x3 卷积核，ReLU 激活 + BatchNormalization，最大池化
2. **卷积块 2**：64 个 3x3 卷积核，ReLU 激活 + BatchNormalization，最大池化
3. **卷积块 3**：128 个 3x3 卷积核，ReLU 激活 + BatchNormalization，最大池化
4. **卷积块 4**：128 个 3x3 卷积核，ReLU 激活 + BatchNormalization，最大池化
5. **全连接层**：512 神经元 + Dropout(0.5)
6. **输出层**：3 神经元，Softmax 激活（三分类）

## 训练配置

### 二分类训练
- **优化器**：Adam
- **损失函数**：Binary Crossentropy
- **数据增强**：
  - 旋转 (±20°)
  - 宽度/高度偏移 (±20%)
  - 剪切变换 (±20%)
  - 缩放 (±20%)
  - 水平翻转
- **验证集比例**：20%

### 三分类训练
- **优化器**：Adam
- **损失函数**：Categorical Crossentropy
- **数据增强**：与二分类相同
- **验证集比例**：20%
- **额外功能**：EarlyStopping + ReduceLROnPlateau

## 使用方法

### 1. 训练二分类模型

```bash
cd f:\Python\AI-HW\hw07
python train.py
```

训练完成后会生成：
- `trained_model.keras` - 训练好的二分类模型
- `training_history.pkl` - 二分类训练历史
- `figures/training_curves.png` - 二分类训练曲线

### 2. 训练三分类模型

```bash
python train_3class.py
```

训练完成后会生成：
- `trained_model_3class.keras` - 训练好的三分类模型
- `training_history_3class.pkl` - 三分类训练历史
- `figures/training_curves_3class.png` - 三分类训练曲线

### 3. 评估二分类模型

```bash
python evaluate_model.py
```

输出内容：
- 测试集 Loss 和 Accuracy
- 分类报告 (Precision, Recall, F1-Score)
- 混淆矩阵图 (`figures/confusion_matrix.png`)

### 4. 评估三分类模型

```bash
python evaluate_3class.py
```

输出内容：
- 测试集 Loss 和 Accuracy
- 分类报告 (Precision, Recall, F1-Score)
- 混淆矩阵图 (`figures/confusion_matrix_3class.png`)
- 归一化混淆矩阵图 (`figures/confusion_matrix_3class_normalized.png`)

### 5. 绘制训练曲线

```bash
python plot_curves.py
```

从训练历史生成 Loss 和 Accuracy 曲线图。

### 6. 统计图片数量

```bash
python count_images.py
```

统计数据集中各类别的图片数量。

### 7. 仅构建模型

```bash
python build_model.py
```

仅构建模型架构，不进行训练。

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `trained_model.keras` | 训练好的二分类 Keras 模型 |
| `trained_model_3class.keras` | 训练好的三分类 Keras 模型 |
| `training_history.pkl` | 二分类训练历史 (Python pickle 格式) |
| `training_history_3class.pkl` | 三分类训练历史 (Python pickle 格式) |
| `figures/training_curves.png` | 二分类训练/验证 Loss 和 Accuracy 曲线 |
| `figures/confusion_matrix.png` | 二分类混淆矩阵热力图 |
| `figures/training_curves_3class.png` | 三分类训练/验证 Loss 和 Accuracy 曲线 |
| `figures/confusion_matrix_3class.png` | 三分类混淆矩阵热力图 |
| `figures/confusion_matrix_3class_normalized.png` | 三分类归一化混淆矩阵热力图 |

## 结果对比

| 指标 | 二分类 (N vs P) | 三分类 (V/B/N) |
|------|-----------------|----------------|
| 准确率 (Accuracy) | **81%** | 57% |
| Viral 召回率 | N/A | 25.68% |
| Bacterial 召回率 | N/A | **96.69%** |
| Normal 召回率 | 51% | 35.90% |
| 加权平均 Precision | 84% | 65% |
| 加权平均 Recall | 81% | 57% |
| 加权平均 F1-Score | 79% | 54% |

## 注意事项

1. 运行训练脚本前请确保数据集完整解压在 `chest_xray/` 目录下
2. GPU 训练会自动启用（如已安装 CUDA/cuDNN）
3. 所有输出文件默认保存在 `hw07` 目录下
4. 三分类模型通过文件名中的 `virus`/`bacteria` 关键字区分病毒性和细菌性肺炎