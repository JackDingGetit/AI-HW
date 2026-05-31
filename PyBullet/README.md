# PyBullet 机械臂抓取仿真

## 项目简介

本项目使用 PyBullet 物理仿真引擎演示 Franka Panda 机械臂的抓取操作。程序模拟了完整的抓取流程：从初始位置移动到物块上方、下降抓取、提升并移动到目标位置、最后释放物块。

## 功能特性

- Franka Panda 7自由度机械臂仿真
- 夹爪开合控制
- 逆运动学姿态控制
- 物体约束绑定实现稳定抓取
- 完整的抓取-移动-放置流程演示

## 文件结构

```
PyBullet/
├── README.md      # 项目说明文档（本文件）
├── 运行说明.txt   # 详细运行步骤说明
└── catch.py       # 机械臂抓取仿真主程序
```

## 快速开始

```bash
# 安装依赖
pip install pybullet pybullet-data

# 运行程序
python catch.py
```

详细运行步骤请参阅 [运行说明.txt](./运行说明.txt)

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.7 或更高版本 |
| PyBullet | 最新版本 |
| 操作系统 | Windows / Linux / macOS |
| 显示器 | 需要（GUI模式） |

## 核心函数说明

| 函数名 | 功能 |
|--------|------|
| `move_to_pose()` | 移动机械臂到指定位置和姿态 |
| `control_gripper()` | 控制夹爪开合 |
| `grasp_object()` | 创建约束绑定物块到夹爪 |
| `release_object()` | 解除约束释放物块 |

## 许可证

MIT License