# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于 Ultralytics YOLO v8.4.50 的修改版本，专注于无人机航拍车辆检测任务。

## 环境配置

```bash
# 激活 conda 环境
conda activate yolo11

# 安装开发模式
pip install -e .

# 安装导出依赖
pip install -e ".[export]"
```

## 常用命令

### 训练

```bash
# 单卡训练
yolo detect train model=yolo11n.pt data=UAVDT.yaml imgsz=640 epochs=100 batch=16

# 使用自定义配置训练
python auto_train_all.py  # 运行消融实验

# CLI 训练示例
yolo train model=ultralytics/cfg/models/11/yolo11.yaml data=coco8.yaml epochs=100 imgsz=640
```

### 推理/预测

```bash
# 图片推理
yolo detect predict model=yolo11n.pt source=image.jpg

# 视频推理
yolo detect predict model=yolo11n.pt source=video.mp4

# 批量推理
yolo detect predict model=yolo11n.pt source=folder/
```

### 验证

```bash
yolo detect val model=yolo11n.pt data=UAVDT.yaml
```

### 导出

```bash
# ONNX 导出
yolo export model=yolo11n.pt format=onnx

# TensorRT 导出
yolo export model=yolo11n.pt format=engine

# OpenVINO 导出
yolo export model=yolo11n.pt format=openvino
```

### 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_python.py

# 运行带覆盖率的测试
pytest tests/ --cov=ultralytics --cov-report=html
```

## 代码架构

### 核心目录结构

```
ultralytics/
├── cfg/           # 配置文件（模型YAML、训练参数）
│   └── models/    # 模型架构定义
│       ├── 11/    # YOLO11 系列
│       ├── 26/    # YOLO26 系列
│       ├── v8/    # YOLOv8 系列
│       └── rt-detr/  # RT-DETR 系列
├── data/          # 数据加载和预处理
├── engine/        # 核心引擎（训练器、预测器、验证器、导出器）
├── models/        # 模型实现
│   ├── yolo/      # YOLO 系列模型
│   │   ├── detect/    # 目标检测
│   │   ├── segment/   # 实例分割
│   │   ├── classify/  # 图像分类
│   │   ├── pose/      # 姿态估计
│   │   └── obb/       # 旋转目标检测
│   ├── sam/       # SAM 分割模型
│   └── rtdetr/    # RT-DETR 检测模型
├── nn/            # 神经网络模块
│   ├── modules/   # 基础模块（Conv、C3k2、SPPF等）
│   └── tasks.py   # 模型构建和解析
├── solutions/     # 应用解决方案
├── trackers/      # 目标跟踪
└── utils/         # 工具函数
```

### 关键组件

1. **模型定义**：`ultralytics/cfg/models/` 下的 YAML 文件定义网络架构
2. **训练器**：`ultralytics/engine/trainer.py` - BaseTrainer 基类
3. **检测训练器**：`ultralytics/models/yolo/detect/train.py` - DetectionTrainer
4. **神经网络模块**：`ultralytics/nn/modules/` - Conv、C3k2、SPPF、Detect 等
5. **损失函数**：`ultralytics/utils/loss.py` - v8DetectionLoss 等
6. **数据增强**：`ultralytics/data/augment.py`

### 自定义实验

项目包含消融实验脚本 `auto_train_all.py`

实验配置示例：
```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11s.yaml").load("yolo11s.pt")
model.train(data="UAVDT.yaml", epochs=150, imgsz=640, batch=32)
```

## 数据集配置

数据集使用 YAML 配置文件，示例（EVD4UAV.yaml）：
```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 3
names:
  0: car
  1: bus
  2: truck
```

## 模型缩放

YOLO11 支持多种缩放级别：
- `n` (nano): 最轻量
- `s` (small): 轻量
- `m` (medium): 中等
- `l` (large): 大型
- `x` (xlarge): 超大型

使用方式：`model=yolo11n.yaml` 或 `model=yolo11s.yaml`

## 开发注意事项

- 代码修改后使用 `pip install -e .` 重新安装
- 模型权重文件（*.pt）不纳入版本控制
- 训练日志和结果保存在 `runs/` 目录
- 使用 `yolo checks` 检查环境依赖
