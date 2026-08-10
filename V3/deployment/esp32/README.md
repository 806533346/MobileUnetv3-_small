# ESP32-S3 部署: MobileNestedUNetv3 息肉分割

INT8 量化 + UART 流式推理。200 张测试集平均 IoU 0.637。

## 概述

- **模型**: MobileNestedUNetv3 (~3.9M 参数)，Kvasir-SEG 训练
- **芯片**: ESP32-S3 (240MHz, 8MB PSRAM, 16MB Flash)
- **框架**: ESP-IDF v6.0.2 + ESP-DL v3.3.7
- **输入**: 256×256×3 NHWC INT8 (exponent=-14, inv_scale=16384)
- **输出**: 256×256 二值掩码 (阈值=-2)
- **推理**: ~1.5-2s (纯推理)，总计 ~9.5s/张 (含 UART 传输)
- **精度**: Avg IoU 0.637 vs GT，73% 图像 IoU ≥ 0.5

## 架构

```
PC (stream_test.py)                     ESP32-S3 (main.cpp)
      │                                        │
      │  115200 baud, "READY\n", 'G'           │  握手
      │←──────────────────────────────────────→│
      │                                        │
      │  460800 baud                           │  切换波特率
      │                                        │
      │  0xFEED0010 + 196608B INT8 NHWC ────→ │  发图
      │                                        │  推理 (~2s)
      │  ←──────────── 0xFEED0020 + 65536B    │  回传掩码
      │                                        │
      │  ... 循环 200 次 ...                    │
      │                                        │
      │  0xFEEDFFFF ──────────────────────────→│  停止
```

## 目录结构

```
deployment/esp32/
│
├── README.md                       # 本文件
├── IMPLEMENTATION_LOG.md           # 详细实现过程与问题记录
│
├── stream_test.py                  # ★ PC 端 UART 流式测试 (200 张)
├── _run_test.py                    # 测试 wrapper (自动复位 ESP32 + 运行)
├── convert_to_espdl.py             # PyTorch → ESP-DL INT8 量化导出
├── generate_test_data.py           # 生成 storage.bin + test_config.h (旧批量模式)
├── check_preprocess.py             # 诊断 /255 预处理
│
├── _check_gt_iou.py                # ONNX float32 IoU vs GT 基线
├── _validate_with_espdl.py         # PPQ NCHW vs NHWC 量化精度验证
├── _validate_espdl_quant.py        # espdl_quantize_torch 精度
├── _validate_quant.py              # quantize_torch_model 精度
│
├── _fix_onnx_scales.py             # 修复 ONNX Resize FLOAT→INT64
├── _diagnose_onnx.py               # ONNX 模型诊断
│
├── _read_serial.py                 # 串口读取工具
│
├── test_data/                      # 200 张 Kvasir-SEG 图像 + 掩码
│   ├── images/
│   └── masks/
│
└── esp32_inference/
    ├── CMakeLists.txt
    │
    ├── model/                      # ESP-DL 量化模型
    │   ├── model.espdl             # FlatBuffer 模型 (~386KB)
    │   ├── model.onnx              # ONNX 模型
    │   ├── model.json              # PPQ 量化配置 (per-tensor INT8)
    │   └── model.info              # 模型结构文本
    │
    └── mobilenet/                  # ESP-IDF 主项目
        ├── CMakeLists.txt
        ├── sdkconfig.defaults      # ESP32-S3 N16R8 (16MB Flash, 8MB PSRAM)
        ├── partitions.csv          # Flash 分区表: factory + model + storage
        ├── deploy.ps1              # 一键部署脚本
        ├── deploy.bat
        │
        ├── main/
        │   ├── main.cpp            # ★ UART 流式推理固件
        │   ├── main_test.cpp       # 最小启动测试
        │   ├── test_config.h       # 自动生成常量
        │   └── mbedtls/sha256.h    # mbedtls v4 兼容补丁
        │
        ├── components/
        │   └── espressif__esp-dl/  # ESP-DL v3.3.7 (框架)
        │
        └── output/
            └── convert_preds.py    # 预测可视化
```

## UART 流式协议

### 魔数

| 名称 | 值 | 方向 | 含义 |
|------|-----|------|------|
| `MAGIC_IMAGE` | `0xFEED0010` | PC → ESP32 | 图像数据紧跟 |
| `MAGIC_RESULT` | `0xFEED0020` | ESP32 → PC | 推理结果紧跟 |
| `MAGIC_STOP` | `0xFEEDFFFF` | PC → ESP32 | 结束流 |

### 握手流程

```
1. ESP32 上电，安装 UART 驱动，发送 "READY\n" (115200 baud)
2. PC 等待 "READY\n"，发送 'G'
3. ESP32 收到 'G'，双方延迟 200ms 后切换到 460800 baud
4. ESP32 关闭日志输出 (esp_log_level_set("*", ESP_LOG_NONE))
```

### 数据格式

**发送 (196608 bytes):** 原始 INT8 NHWC 数据，无编码，无 header

**接收 (65540 bytes):** 4B `MAGIC_RESULT` (LE) + 65536B uint8 二值掩码 (0/1)

### 预处理 (必须严格匹配训练流程)

```python
albu.Resize(256, 256)
albu.Normalize()                     # mean=0, std=1
.astype("float32") / 255.0           # → ~[-0.008, 0.008]
np.round(nhwc * 16384)               # quantize to INT8 (inv_scale = 2^14)
np.clip(-128, 127).astype(np.int8)
```

### 输出解码

```cpp
for (size_t i = 0; i < output_count; i++) {
    float v = dl::dequantize(quant_out[i], output_scale);
    mask_out[i] = (v > -2) ? 1 : 0;  // 阈值=-2 补偿 INT8 量化偏置
}
```

## 快速开始

### 1. 编译固件

```powershell
# 进入项目目录，重映射中文路径 (if needed)
subst T: "C:\Users\匿名\.espressif"

# 设置环境 (PowerShell)
$env:IDF_PATH = "C:\esp\v6.0.2\esp-idf"
$env:ESP_IDF_VERSION = "6.0.2"
$env:IDF_PYTHON_ENV_PATH = "T:\python_env\idf6.0_py3.13_env"
$env:PATH = "T:\python_env\idf6.0_py3.13_env\Scripts;T:\tools\cmake\4.0.3\bin;T:\tools\Ninja\1.12.1;T:\tools\xtensa-esp-elf\esp-14.2.0_20241118\xtensa-esp-elf\bin;$env:PATH"

cd esp32_inference\mobilenet
& "T:\python_env\idf6.0_py3.13_env\Scripts\python.exe" "C:\esp\v6.0.2\esp-idf\tools\idf.py" build
```

### 2. 烧录

```powershell
python -m esptool --chip esp32s3 -b 460800 --before default-reset --after hard-reset write-flash `
  --flash-mode dio --flash-size 16MB --flash-freq 80m `
  0x0 build\bootloader\bootloader.bin `
  0x8000 build\partition_table\partition-table.bin `
  0x10000 build\mobilenet.bin `
  0x410000 ..\model\model.espdl
```

或使用 `.\deploy.ps1` 一键部署。

### 3. 运行 UART 流式测试

```powershell
cd deployment\esp32
python -u stream_test.py
```

输出示例:
```
Found 200 test images
Opening COM5 at 115200...
Waiting for ESP32 READY...
ESP32 ready!
[  0] cju0qkwl35piu0993l0dewei2: IoU=0.6298  [9s elapsed]
[  1] cju0vtox5ain6099360pu62rp: IoU=0.4891  [19s elapsed]
...
============================================================
Total: 200 images in 1901s
Average IoU vs GT: 0.6370
IoU range: [0.0000, 0.9632]
Breakdown: <0.3: 28, 0.3-0.5: 26, 0.5-0.7: 40, >=0.7: 106
```

## 模型量化

### 流水线

```
PyTorch .pth → ESP-PPQ INT8 quantize → ESP-DL FlatBuffer (.espdl)
```

关键参数:
- **量化方案**: Per-tensor INT8 对称 power-of-2 缩放
- **Input exponent**: -14 (inv_scale = 2^14 = 16384)
- **校准数据**: 50 张 Kvasir-SEG 图像
- **上采样**: bilinear → nearest (量化友好)

### 重新量化

```powershell
python convert_to_espdl.py
```

## 性能

| 环节 | 耗时 (s) | 占比 |
|------|----------|------|
| 图片发送 (196KB → ESP32) | ~5.0 | 53% |
| ESP32 推理 | ~1.5-2.0 | 18% |
| Mask 接收 (65KB ← ESP32) | ~1.5 | 16% |
| Python IO + 前后处理 | ~1.0-1.5 | 13% |
| **总计** | **~9.5** | — |

瓶颈在 UART 传输 (70%)。波特率与稳定性权衡:

| 波特率 | 速度 | 稳定性 |
|--------|------|--------|
| 921600 | ~7s/张 | 不稳定，~25 张崩 |
| 460800 | ~9.5s/张 | 稳定，200 张零错误 |

## 测试结果

**200 张完整测试集 (Kvasir-SEG):**

| 指标 | 值 |
|------|-----|
| 平均 IoU | 0.6370 |
| IoU 范围 | [0.0000, 0.9632] |
| < 0.3 | 28 张 (14%) |
| 0.3 - 0.5 | 26 张 (13%) |
| 0.5 - 0.7 | 40 张 (20%) |
| ≥ 0.7 | 106 张 (53%) |

低分图像 (~0 IoU) 与 ONNX float32 基线一致，属于模型对极小/无息肉样本的固有误差。

## Flash 分区

| 地址 | 大小 | 名称 | 内容 |
|------|------|------|------|
| 0x000000 | 24KB | nvs | NVS 非易失存储 |
| 0x009000 | 4KB | phy_init | PHY 初始化 |
| 0x010000 | 4MB | factory | 固件 |
| 0x410000 | 2MB | model | model.espdl |
| 0x610000 | ~10MB | storage | 测试数据 (流式模式未使用) |

## 常见问题

| 现象 | 解决 |
|------|------|
| `Timeout waiting for READY` | 复位 ESP32: DTR/RTS toggle 或重插 USB |
| `PermissionError: COM5` | 重插 USB 释放端口 |
| Bad magic `0x616D6920` | 数据错位，降低波特率 (460800 → 230400) |
| 编译时中文路径乱码 | `subst T: "C:\Users\匿名\.espressif"` 用 T: 引用路径 |
| `cmake not found` | 先 source `export.bat` 或手动设 PATH 到 cmake 目录 |
| 看门狗超时 | `CONFIG_ESP_TASK_WDT_TIMEOUT_S=30` + 每张图间 `vTaskDelay(1)` |

## 环境

| 组件 | 版本 |
|------|------|
| Python | 3.13 (conda) |
| PyTorch | 2.x |
| ESP-PPQ | esp_ppq (ESP-DL 定制版) |
| ESP-IDF | v6.0.2 |
| 工具链 | xtensa-esp-elf 15.2.0, CMake 4.0.3, Ninja 1.12.1 |
| pyserial | 3.5+ |
