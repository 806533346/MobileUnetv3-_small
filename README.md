# MobileV3Unet++

基于 MobileNetV3 + UNet++ 架构的语义分割项目，使用 PyTorch 实现。

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
双击 environment/create_env.bat

# 2. 激活环境
conda activate seg

# 3. 准备数据集（放到 inputs/ 下，详见"数据集准备"）

# 4. 训练
python train.py

# 5. 测试
python val.py

# 6. 导出 ONNX
python export_onnx.py
```

## 环境配置

环境配置文件位于 `environment/` 目录下：

| 文件 | 用途 |
|------|------|
| `environment/create_env.bat` | 创建 conda 环境 `seg` 并一键安装所有依赖 |
| `environment/install_deps.bat` | 在已有环境 `Py` 中安装依赖（不创建新环境） |
| `environment/environment_requirement.txt` | 依赖包列表 |

已有 conda 环境也可以直接：

```bash
pip install -r environment/environment_requirement.txt
```

### 依赖说明

| 包 | 用途 |
|------|------|
| PyTorch >= 1.7 | 深度学习框架 |
| albumentations | 数据增强 |
| torchsummary, fvcore | 模型分析 |
| onnx, onnxruntime | ONNX 导出与推理 |
| opencv-python, numpy, pandas, matplotlib, tqdm, pyyaml | 数据处理与可视化 |
| scikit-learn | 数据集划分 |

## 项目结构

```
├── README.md
├── environment/
│   ├── environment_requirement.txt
│   ├── create_env.bat
│   └── install_deps.bat
└── V3/
    ├── archs.py                   # 模型定义
    ├── train.py                   # 训练脚本
    ├── val.py                     # 验证/测试脚本
    ├── export_onnx.py             # ONNX 导出脚本
    ├── dataset.py                 # 数据集加载器
    ├── losses.py                  # 损失函数
    ├── metrics.py                 # 评估指标
    ├── utils.py                   # 工具函数
    ├── preprocess_dsb2018.py      # DSB2018 数据集预处理
    ├── inputs/                    # 数据集
    ├── models/                    # 训练输出
    │   └── <name>/
    │       ├── model.pth          # 最佳权重
    │       ├── model.onnx         # ONNX 模型
    │       ├── config.yml         # 训练配置
    │       └── log.csv            # 训练日志
    ├── outputs/                   # 分割结果
    └── deployment/               # ESP32 边缘部署
        └── esp32/                → 详见下方 ESP32 边缘部署 章节
```

## 模型架构

### MobileUNetV3

经典的 U-Net 编解码结构，编码器使用 MobileNetV3 倒置残差块（`Block`），解码器使用标准 `VGGBlock`，通过跳跃连接和上采样逐步恢复空间分辨率。

### MobileNestedUNetv3 (推荐)

UNet++ 风格的嵌套 U-Net 结构，在跳跃路径上引入密集的嵌套卷积块，支持 **深度监督（deep supervision）**——可在多个解码层级同时输出分割结果，加速梯度传播并提升模型收敛。

两者均使用：
- **Hardswish** / **ReLU** 激活函数
- **Squeeze-and-Excitation (SE)** 注意力模块
- 深度可分离卷积（MobileNetV3 核心）

## 数据集准备

> 数据集需自行下载，`inputs/` 已在 `.gitignore` 中排除，不会被提交到仓库。

将数据集按以下结构放入 `inputs/<dataset_name>/`：

```
inputs/<dataset_name>/
├── images/
│   ├── sample1.jpg
│   └── ...
└── masks/
    ├── 0/
    │   ├── sample1.jpg
    │   └── ...
    └── 1/                    # 多类别时继续添加
        ├── sample1.jpg
        └── ...
```

images 和 masks 文件名需一致，mask 按类别分子文件夹存放。

## 训练

```bash
python train.py \
    --dataset Kvasir_SEG2026 \
    --arch MobileNestedUNetv3 \
    --num_classes 1 \
    --input_channels 3 \
    --input_h 320 \
    --input_w 384 \
    --epochs 150 \
    --batch_size 8
```

也可直接运行，全部使用默认参数：

```bash
python train.py
```

模型名自动生成为 `{dataset}_{arch}_woDS`（深度监督为 `wDS`）。训练输出保存在 `models/<name>/` 下：

| 文件 | 说明 |
|------|------|
| `model.pth` | 最佳模型权重（按 val_iou 选择） |
| `config.yml` | 训练配置 |
| `log.csv` | 训练日志（含 loss、IoU、Dice、Recall、Precision、F1、F2 等） |

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集名称（对应 inputs/ 下的文件夹） | `Kvasir_SEG2026` |
| `--arch` | 模型架构：`MobileNestedUNetv3` 或 `MobileUNetV3` | `MobileNestedUNetv3` |
| `--deep_supervision` | 是否启用深度监督 | `false` |
| `--num_classes` | 类别数 | `1` |
| `--input_channels` | 输入通道数 | `3` |
| `--input_h / --input_w` | 输入尺寸 | `320 / 384` |
| `--epochs` | 训练轮数 | `150` |
| `--batch_size` | 批次大小 | `8` |
| `--loss` | 损失函数：`BCEDiceLoss` / `BCEWithLogitsLoss` / `LovaszHingeLoss` | `BCEDiceLoss` |
| `--optimizer` | 优化器：`Adam` / `SGD` / `RMSProp` | `Adam` |
| `--lr` | 初始学习率 | `1e-3` |
| `--scheduler` | 调度器：`CosineAnnealingLR` / `ReduceLROnPlateau` / `MultiStepLR` | `CosineAnnealingLR` |
| `--early_stopping` | 早停轮数（-1 关闭） | `20` |

## 验证 / 测试

```bash
python val.py --name Kvasir_SEG2026_MobileNestedUNetv3_woDS
```

从 `models/<name>/config.yml` 恢复训练配置，加载最佳权重，在测试集（20% 数据）上计算指标，并将分割结果保存至 `outputs/<name>/`。

## ONNX 导出

```bash
python export_onnx.py --name Kvasir_SEG2026_MobileNestedUNetv3_woDS
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--opset` | ONNX opset 版本 | `11` |
| `--dynamic_batch` | 导出动态 batch size | `false` |

## ESP32 边缘部署

完整的端到端部署管线：PyTorch QAT → ESP-PPQ INT8 量化 → ESP-DL flatbuffer → ESP32-S3 硬件推理。

```bash
# 1. 搭建部署环境
cd V3/deployment/esp32
双击 setup_env.bat           # 安装 PC 端 Python 依赖
conda activate esp32

# 2. 量化导出
python convert_to_espdl.py   # QAT 模型 → ESP-DL INT8 flatbuffer

# 3. 烧录固件
双击 esp32_inference/mobilenet/_build_now.bat   # 编译固件
双击 esp32_inference/mobilenet/_flash_now.bat   # 烧录到 ESP32

# 4. 硬件推理测试
python _run_test.py          # 200 张图像 ESP32 推理 + IoU 评估
```

### 部署目录结构

```
V3/deployment/esp32/
├── convert_to_espdl.py              # PyTorch → ESP-DL INT8 量化导出
├── _run_test.py                     # ESP32 硬件推理测试 (921600 baud UART)
├── _validate_full_testset.py        # PC 端 PPQ INT8 仿真验证
├── _fp32_baseline.py                # FP32 模型基线测试
├── _pc_int8_vs_gt.py                # PC INT8 vs GT 对比
├── generate_test_data.py            # 测试数据生成
├── setup_env.bat                    # PC 端环境一键搭建
├── test_data/                       # 200 张 Kvasir-SEG 测试集
├── esp32_inference/
│   ├── model/                       # ESP-DL flatbuffer 模型 (model.espdl)
│   └── mobilenet/
│       ├── main/main.cpp            # ESP32 固件源码 (C++/FreeRTOS)
│       ├── _build_now.bat           # ESP-IDF 编译脚本
│       ├── _flash_now.bat           # 烧录脚本
│       ├── partitions.csv           # Flash 分区表
│       └── pack_storage_bin.py      # 存储分区打包
├── IMPLEMENTATION_LOG.md            # 实施日志与问题记录
└── README.md                        # 详细部署文档
```

> 详细说明见 `V3/deployment/esp32/README.md`

## 评估指标

| 指标 | 说明 |
|------|------|
| IoU (Jaccard) | 交并比 |
| Dice Coefficient | Dice 相似系数 |
| Recall | 召回率 / 灵敏度 |
| Specificity | 特异度 |
| Precision | 精确率 |
| F1 Score | F1 分数 |
| F2 Score | F2 分数（召回率权重更高） |

## 训练结果参考

| 模型 | 数据集 | val_iou | test_iou |
|------|--------|---------|----------|
| MobileNestedUNetv3 woDS | Kvasir_SEG2026 | 0.7191 | 0.6786 |
