# ESP32-S3 部署: MobileNestedUNetv3 息肉分割

INT8 量化 + UART 流式推理 (921600 baud)。200 张 Kvasir-SEG 测试集 IoU 0.6115。

## 概述

- **模型**: MobileNestedUNetv3 (~3.9M 参数, QAT INT8)
- **芯片**: ESP32-S3 (240MHz, 8MB PSRAM, 16MB Flash)
- **框架**: ESP-IDF v6.0.2 + ESP-DL
- **输入**: 256×256×3 NHWC INT8 (exponent=-14, scale=2^14=16384)
- **输出**: 256×256 二值掩码, 打包为 8192 bytes (8 像素/byte, LSB first)
- **精度**: ESP32 IoU 0.6115 vs GT (FP32 基线 0.6892 / PC INT8 0.6583)
- **速度**: ~6.2s/张 (含 UART 传输), 纯推理 ~3.8s

## 架构

```
PC (_run_test.py)                          ESP32-S3 (main.cpp)
      │                                             │
      │  115200 baud, "READY\n", 'G'                │  握手
      │←───────────────────────────────────────────→│
      │                                             │
      │  921600 baud                                │  切换波特率
      │                                             │
      │  0xFEED0010 + 196608B INT8 NHWC ──────────→│  发图 → Core 0 RX 三缓冲
      │                                             │  Core 1 推理 (~3.8s)
      │  ←──── 0xFEED0030 + 4B time + 8192B packed │  回传打包掩码
      │                                             │
      │  ... 循环 200 次 ...                         │
      │                                             │
      │  0xFEEDFFFF ───────────────────────────────→│  停止
```

双核三缓冲流水线: Core 0 持续 UART 接收, Core 1 推理, 互不阻塞。

## 目录结构

```
deployment/esp32/
├── README.md                           # 本文件
├── IMPLEMENTATION_LOG.md               # 实施日志与问题记录
├── setup_env.bat                       # PC 端 conda 环境一键搭建
│
├── convert_to_espdl.py                 # PyTorch QAT → ESP-DL INT8 量化导出
├── _run_test.py                        # ESP32 硬件推理测试 (921600 baud)
├── _validate_full_testset.py           # PC 端 PPQ INT8 200 张仿真验证
├── _fp32_baseline.py                   # FP32 模型 IoU 基线
├── _pc_int8_vs_gt.py                   # PC INT8 vs GT 对比 (同阈值=-2)
├── generate_test_data.py               # 测试数据生成
│
├── test_data/                          # 200 张 Kvasir-SEG 测试集
│   ├── images/                         # 原图
│   └── masks/0/                        # 息肉 GT 标注
│
└── esp32_inference/
    ├── model/                          # ESP-DL 量化模型
    │   ├── model.espdl                 # FlatBuffer (~386KB)
    │   ├── model.onnx                  # ONNX (FP32, 1MB)
    │   ├── model.json                  # PPQ 量化配置
    │   └── model.info                  # 模型结构文本
    │
    └── mobilenet/                      # ESP-IDF 固件项目
        ├── main/main.cpp               # UART 流式推理固件 (C++/FreeRTOS)
        ├── main/test_config.h          # 输入/输出常量
        ├── _build_now.bat              # ESP-IDF 编译脚本
        ├── _flash_now.bat              # 烧录脚本
        ├── deploy.ps1 / deploy.bat     # 一键部署
        ├── pack_storage_bin.py         # 存储分区打包
        ├── partitions.csv              # Flash 分区表
        └── sdkconfig.defaults          # ESP32-S3 N16R8
```

## UART 流式协议

### 魔数

| 名称 | 值 | 方向 | 含义 |
|------|-----|------|------|
| `MAGIC_IMAGE` | `0xFEED0010` | PC → ESP32 | 196608B INT8 数据紧跟 |
| `MAGIC_RESULT` | `0xFEED0030` | ESP32 → PC | 推理结果紧跟 (packed 格式) |
| `MAGIC_STOP` | `0xFEEDFFFF` | PC → ESP32 | 结束流 |

### 握手

```
1. ESP32 上电, 发送 "READY\n" (115200 baud)
2. PC 收到后发送 'G'
3. 双方等 200ms, 切换到 921600 baud
4. ESP32 关闭日志, 进入二进制模式
```

### 数据包格式

```
PC → ESP32: [4B 0xFEED0010 LE] [196608B INT8 NHWC]
ESP32 → PC: [4B 0xFEED0030 LE] [4B infer_time_us LE] [8192B packed mask]
```

掩码打包: 8 像素/byte, LSB first. `v > -2` 判前景 (阈值=-2 补偿 INT8 负偏置).

### 预处理 (PC 侧)

```python
albu.Resize(256, 256) → albu.Normalize() → /255
→ np.round(nhwc * 16384) → clip(-128, 127) → int8
```

预处理完成后转 `tobytes()` 直接发送, ESP32 零拷贝构造 INT8 tensor.

## 快速开始

### 1. 搭建环境

```bash
cd V3/deployment/esp32
双击 setup_env.bat           # 安装 Python 依赖 esp-ppq/torch/opencv/...
conda activate esp32
```

### 2. 编译固件

```bash
双击 esp32_inference/mobilenet/_build_now.bat
```

### 3. 烧录

```bash
双击 esp32_inference/mobilenet/_flash_now.bat
```

或手动:
```powershell
python -m esptool --chip esp32s3 -b 921600 write-flash \
  --flash-mode dio --flash-size 16MB --flash-freq 80m \
  0x0 build/bootloader/bootloader.bin \
  0x8000 build/partition_table/partition-table.bin \
  0x10000 build/mobilenet.bin \
  0x410000 ../model/model.espdl
```

### 4. 运行测试

```bash
python _run_test.py           # 200 张 ESP32 推理 + IoU 评估
```

## 模型量化

```
PyTorch QAT .pth → ESP-PPQ (espdl_quantize_torch) → ESP-DL FlatBuffer (.espdl)
```

关键参数:
- **量化方案**: Per-tensor INT8 对称 power-of-2 缩放
- **Input exponent**: -14 (scale = 2^14 = 16384)
- **校准数据**: 10 张 Kvasir-SEG 图像
- **上采样**: bilinear → nearest (量化友好)

重新量化: `python convert_to_espdl.py`

## 性能 (200 张 @ 921600 baud)

| 环节 | 耗时 (s) | 占比 |
|------|----------|------|
| 图片发送 (196KB) | ~2.1 | 34% |
| ESP32 推理 | ~3.8 | 61% |
| Mask 接收 (8KB packed) | ~0.1 | 2% |
| Python IO + 前后处理 | ~0.2 | 3% |
| **总计** | **~6.2** | — |

纯推理是瓶颈 (61%), UART 传输已从 53% 降到 34%。

## 测试结果

**200 张 Kvasir-SEG 完整测试集:**

| 环境 | IoU vs GT | 说明 |
|------|-----------|------|
| PC FP32 (>0) | 0.6892 | 模型精度上限 |
| PC INT8 (>-2) | 0.6583 | 量化后精度 |
| ESP32 INT8 (>-2) | 0.6115 | 实际部署精度 |

**ESP32 IoU 分布:**
| < 0.3 | 0.3-0.5 | 0.5-0.7 | ≥ 0.7 |
|-------|---------|---------|-------|
| 30 (15%) | 24 (12%) | 43 (21.5%) | 103 (51.5%) |

量化损失 ~0.03 (FP32→INT8), 硬件一致性损失 ~0.05 (PC INT8→ESP32).

## Flash 分区

| 地址 | 大小 | 名称 | 内容 |
|------|------|------|------|
| 0x010000 | 4MB | factory | 固件 (mobilenet.bin) |
| 0x410000 | 2MB | model | model.espdl |
| 0x610000 | ~10MB | storage | 未使用 (流式模式) |

## 常见问题

| 现象 | 解决 |
|------|------|
| `Timeout waiting for READY` | 拔插 USB 复位 ESP32 |
| `PermissionError: COM5` | 重插 USB 释放端口 |
| Bad magic 数据错位 | 检查串口线, 降低波特率 (921600 → 460800) |
| 编译中文路径乱码 | `_build_now.bat` 已内置 `subst X:` 绕过 |
| PSRAM 分配失败 | 检查 `sdkconfig.defaults` 中 PSRAM 配置 |
| IoU=0 的图 | FP32 基线也是 0, 模型对无息肉样本的固有误差 |

## 环境

| 组件 | 版本 |
|------|------|
| Python | 3.13 (conda env: esp32) |
| PyTorch | 2.13 (CPU) |
| ESP-PPQ | 1.3.6 |
| ESP-IDF | v6.0.2 |
| 工具链 | xtensa-esp-elf 15.2.0, CMake 4.0.3, Ninja 1.12.1 |
