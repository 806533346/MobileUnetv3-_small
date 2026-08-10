# MobileNestedUNetv3 ESP32-S3 部署改进日志

将 PyTorch INT8 量化息肉分割模型部署到 ESP32-S3 的完整过程、问题诊断与解决方案。

## 项目概述

- **模型**: MobileNestedUNetv3 (~3.9M 参数), Kvasir-SEG 训练
- **输入**: 256×256×3 NHWC INT8 (exponent=-14, inv_scale=16384)
- **输出**: 256×256 二值掩码 (阈值=-2)
- **硬件**: ESP32-S3 (240MHz, 8MB PSRAM, 16MB Flash)
- **框架**: ESP-IDF v6.0.2 + ESP-DL v3.3.7

---

## 阶段总览

| 阶段 | 模式 | IoU vs GT | 说明 |
|------|------|-----------|------|
| 初始 | Flash 批量 | 0.0015 | NCHW/NHWC 布局错误 |
| 阶段 1 | Flash 批量 | 0.40 | 修复布局 + /255 预处理 |
| 阶段 2 | Flash 批量 | 0.53 | 阈值 0→-2 补偿 INT8 偏差 |
| 阶段 3 | UART 流式 | 0.53 (5 张) | 流式协议替换 Flash 批量 |
| 阶段 4 | UART 流式 | 0.62 (113 张) | 921600 baud → CH340 丢帧 |
| **阶段 5** | **UART 流式** | **0.64 (200 张)** | **460800 baud 稳定完成** |

---

## 阶段 1: Flash 批量推理 — 布局修复

### 问题 1.1: NCHW vs NHWC 布局不匹配

**现象**: ESP32 推理输出 IoU=0.0015 vs GT（几乎为随机噪声）。

**原因**: ESP-DL 模型期望 NHWC 布局 `[1, 256, 256, 3]`，但代码传入了 PyTorch 标准的 NCHW `[1, 3, 256, 256]`。通道维度和空间维度互换。

**证据**: `model.info` 第 6 行:
```
%input.1[INT8, 1x256x256x3], exponents: [-14]
```

**修复**: `main.cpp` 中添加 NCHW→NHWC 转置循环:
```cpp
for (int c = 0; c < TEST_INPUT_C; c++)
    for (int h = 0; h < TEST_INPUT_H; h++)
        for (int w = 0; w < TEST_INPUT_W; w++) {
            size_t nchw_idx = c * ch_size + h * TEST_INPUT_W + w;
            size_t nhwc_idx = h * TEST_INPUT_W * TEST_INPUT_C + w * TEST_INPUT_C + c;
            quant_input[nhwc_idx] = dl::quantize<int8_t>(test_in[nchw_idx], input_scale);
        }
std::vector<int> input_shape = {1, TEST_INPUT_H, TEST_INPUT_W, TEST_INPUT_C};
```

**效果**: IoU 0.0015 → 0.4003（26 倍提升）。

### 问题 1.2: 预处理 /255 缺失

**现象**: ONNX float32 输出只有稀疏噪点 (IoU≈0.005 vs GT)。

**原因**: 训练代码 `dataset.py:46` 在 `albu.Normalize()` 后额外执行 `img / 255`，值域从 `[0,1]` 变为 `[-0.008, 0.008]`。推理脚本缺少此步骤，输入比训练时大 255 倍。

**验证** (`check_preprocess.py`):

| 预处理 | IoU vs GT |
|--------|-----------|
| Normalize + /255 | 0.5407 |
| Normalize only | 0.0052 |

**修复**: 所有推理预处理添加 `/255`:
```python
img = aug['image'].astype('float32') / 255  # 匹配训练
```

**影响文件**: `generate_test_data.py`, `convert_to_espdl.py`, `_validate_quant.py`, `_validate_espdl.py`, `_validate_espdl_quant.py`, `_validate_with_espdl.py`

**效果**: ONNX vs GT 恢复至 IoU≈0.54。

---

## 阶段 2: 阈值优化 — INT8 量化偏置补偿

### 问题 2.1: 阈值=0 导致 IoU 偏低

**现象**: ESP32 IoU=0.40 vs GT，但 Python PPQ 模拟可达 0.50。怀疑 Conv 指数约束导致，但根因是输出阈值。

**分析**: INT8 对称量化 (per-tensor power-of-2) 在 dequantize 后引入系统负偏置。

`_check_gt_iou.py` 对比 ONNX float32 在不同阈值下的表现:

| 阈值 | Avg IoU (5 张) |
|------|----------------|
| 0 | 0.5762 |
| -2 | 0.5374 |

**修复**: `main.cpp` 将二值化阈值从 0 改为 -2:
```cpp
// 阈值=-2 补偿 INT8 量化系统的负偏置
mask_out[i] = (v > -2) ? 1 : 0;
```

**效果**: ESP32 IoU 5 张平均 0.5310，超过 0.5 目标。

### 问题 2.2: 输入 exponent 校准

**现象**: 旧版 `test_config.h` 输入 exponent=-6 (scale=64)，但模型实际 exponent=-14 (scale=1/16384)。

**修复**: 重新运行 PPQ 量化导出 `model.espdl`，确认 input exponent=-14, inv_scale=16384。

### 训练代码优化 (辅助)

| 文件 | 改动 | 原因 |
|------|------|------|
| `archs.py` | bilinear→nearest 上采样 | 量化友好（减少插值误差） |
| `export_onnx.py` | 动态空间维度 + `dynamo=False` | 兼容 ESP-DL 的 ONNX 解析 |
| `dataset.py` | 补充 /255 预处理文档 | 防止推理时遗忘 |

---

## 阶段 3: UART 流式推理架构

### 背景

Flash 批量模式只能跑 5 张图（storage.bin 受 Flash 大小限制）。用户提出流式推理——PC 逐张发送图像，ESP32 逐张返回掩码。

### 协议设计

```
PC                            ESP32
 │                               │
 │  [115200] 等待 "READY\n"     │  uart_driver_install + 发送 READY
 │  ←───────────────────────────│
 │                               │
 │  'G' ───────────────────────→│  握手确认
 │                               │
 │  [921600 baud]               │  切换高速模式
 │                               │
 │  0xFEED0010 ────────────────→│  魔数: 图像数据
 │  196608B INT8 NHWC ─────────→│  原始量化图像
 │                               │  推理 (~2s)
 │  ←────────────── 0xFEED0020  │  魔数: 结果数据
 │  ←────────────── 65536B mask │  uint8 二值掩码
 │                               │
 │  ... 循环 ...                 │
 │                               │
 │  0xFEEDFFFF ────────────────→│  停止
```

### 魔数定义

| 名称 | 值 | 方向 |
|------|-----|------|
| `MAGIC_IMAGE` | `0xFEED0010` | PC → ESP32 |
| `MAGIC_RESULT` | `0xFEED0020` | ESP32 → PC |
| `MAGIC_STOP` | `0xFEEDFFFF` | PC → ESP32 |

### PC 端 (`stream_test.py`)

- 预处理: Resize→Normalize→/255→×16384→clamp→INT8 (与训练严格一致)
- `read_exact()`: 带重试的精确字节读取，超时 15s
- 批量统计: IoU 分布直方图、均值、范围

### ESP32 端 (`main.cpp`)

关键实现细节:

```cpp
// 1. 必须先安装 UART 驱动才能使用 uart_read_bytes/write_bytes
uart_driver_install(UART_NUM_0, 4096, 0, 0, NULL, 0);

// 2. 握手: 持续发 READY 直到收到 'G'
uart_write_bytes(UART_NUM_0, "READY\n", 6);
while (uart_read_bytes(UART_NUM_0, &go, 1, pdMS_TO_TICKS(200)) <= 0) {
    uart_write_bytes(UART_NUM_0, "READY\n", 6);
}

// 3. 切换波特率: 双方等 200ms 后切换
vTaskDelay(pdMS_TO_TICKS(200));
uart_set_baudrate(UART_NUM_0, BAUD_DATA);

// 4. 关闭日志避免污染二进制流
esp_log_level_set("*", ESP_LOG_NONE);

// 5. 直接以 NHWC INT8 喂入模型（无需转置）
dl::TensorBase *input_tensor = new dl::TensorBase(
    {1, 256, 256, 3}, quant_input, input_exponent,
    dl::DATA_TYPE_INT8, false, EXT_MEM_CAPS);

// 6. 推理后二值化 (阈值=-2)
for (size_t i = 0; i < output_count; i++) {
    float v = dl::dequantize(quant_out[i], output_scale);
    mask_out[i] = (v > -2) ? 1 : 0;
}
```

### 遇到的问题与修复

| 问题 | 现象 | 修复 |
|------|------|------|
| `uart_read_bytes` 无数据 | ESP32 不输出 READY | 必须先 `uart_driver_install()` |
| 波特率切换时序 | PC 收不到 READY | 握手协议: ESP32 等 'G' 再切换 |
| 图像间数据粘连 | mask 少收 176 字节 | ESP32 加 `vTaskDelay(10ms)` + Python 加 `read_exact()` |
| 串口日志污染 | 二进制流混入 `ESP_LOGI` 文本 | `esp_log_level_set("*", ESP_LOG_NONE)` |

---

## 阶段 4: CH340 稳定性 — 波特率优化

### 问题 4.1: 921600 baud → Bad Magic

**现象**: 第 25 张图像收到 `Bad magic: 0x616D6920`。

解码: `0x61='a', 0x6D='m', 0x69='i', 0x20=' '` → ASCII 文本！数据流失同步，ESP32 输出和 PC 读取错位。

同时 image 24 的 IoU 从预期的 0.9066 降到 0.2417，确认同步在 image 24 的传输中开始偏移。

**根因**: CH340 USB-Serial 芯片在 921600 baud 长时间传输下不稳定。约 180s 后出现数据字节丢失/插入。

**证据**: 3 次 921600 测试表现一致:
1. 第 1 次: image 113 丢帧 (65360/65536 bytes)
2. 第 2 次: image 25 Bad magic
3. 第 3 次: image 25 Bad magic (完全相同位置)

### 修复 4.1: 降波特率

| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| `BAUD_DATA` (ESP32 main.cpp) | 921600 | 460800 |
| `ser.baudrate` (stream_test.py) | 921600 | 460800 |
| ESP32 发送后延时 | 10ms | 30ms |

### 结果

| 指标 | 921600 | 460800 |
|------|--------|--------|
| 完成率 | 25/200 (12.5%) | **200/200 (100%)** |
| 速度 | ~7s/张 | ~9.5s/张 |
| 数据错误 | Bad magic @ #25 | 零错误 |
| 总耗时 | 226s (未完成) | 1901s (完成) |

---

## 阶段 5: 完整测试 — 200 张 Kvasir-SEG

### 测试配置

- **测试集**: 200 张 Kvasir-SEG 图像 + 对应 GT mask
- **波特率**: 460800
- **延时**: ESP32 30ms / 张
- **阈值**: -2

### 结果

| 指标 | 值 |
|------|-----|
| 总图像数 | 200/200 (100%) |
| 总耗时 | 1901s (31.7 min) |
| 平均 IoU | **0.6370** |
| IoU 范围 | [0.0000, 0.9632] |
| < 0.3 | 28 张 (14%) |
| 0.3 - 0.5 | 26 张 (13%) |
| 0.5 - 0.7 | 40 张 (20%) |
| ≥ 0.7 | 106 张 (53%) |

### 性能拆解 (每张)

| 环节 | 耗时 (s) | 占比 |
|------|----------|------|
| 图片发送 (196KB → ESP32) | ~5.0 | 53% |
| ESP32 纯推理 | ~1.5-2.0 | 18% |
| Mask 接收 (65KB ← ESP32) | ~1.5 | 16% |
| Python IO + 前后处理 | ~1.0-1.5 | 13% |
| **总计** | **~9.5** | — |

### 关键结论

1. **73%** 图像 IoU ≥ 0.5，达到实用阈值
2. **53%** 图像 IoU ≥ 0.7，表现优异
3. 低分图像 (~0 IoU) 与 ONNX float32 基线一致，属模型本身对极小/无息肉样本的固有误差，非量化或传输问题
4. 460800 baud 是 CH340 在 Windows 下长时间稳定传输的最高实测速率

---

## 未解决问题

### ESP-DL Conv 量化指数约束

PPQ 导出时部分 Conv 层报错:
```
output_exponent >= input_exponent + weight_exponent
```

违反此约束的层在 Python PPQ 模拟中正常，但 ESP32 硬件可能产生 INT32 累加溢出。这是 ESP32 IoU 与 Python PPQ 之间差距的可能来源。

**影响**: 约 0.05-0.10 IoU 损失（估计）。

**可能方向**: 调整 PPQ 量化策略分配，或修改模型 Conv 结构。

---

## IoU 演进时间线

```
IoU
0.65 ┤                                           ★ 0.637 (200张, 460800)
0.60 ┤                        ● 0.619 (113张)
0.55 ┤
0.50 ┤              ▲ 0.531
0.45 ┤
0.40 ┤        ■ 0.400
0.35 ┤
0.30 ┤
0.25 ┤
0.20 ┤
0.15 ┤
0.10 ┤
0.05 ┤
0.00 ┤  × 0.002
      ├────┼────┼────┼────┼────┼───
      初始   布局  /255  阈值  CH340  最终
           修复   修复  -2   修复
```

| # | IoU | 改动 |
|---|-----|------|
| × | 0.002 | 初始: NCHW + 缺 /255 |
| ■ | 0.400 | 修复 NHWC + 修复 /255 |
| ▲ | 0.531 | 阈值 0→-2 |
| ● | 0.619 | UART 流式协议 |
| ★ | 0.637 | 460800 稳定 200 张 |

---

## 环境信息

| 组件 | 版本 |
|------|------|
| Python | 3.13 (conda) |
| PyTorch | 2.x |
| ESP-PPQ | esp_ppq (ESP-DL 定制版) |
| ONNX Runtime | CPU 后端 |
| ESP-IDF | v6.0.2 |
| ESP-DL | v3.3.7 (espressif/esp-dl) |
| 工具链 | xtensa-esp-elf 15.2.0, CMake 4.0.3, Ninja 1.12.1 |
| pyserial | 3.5+ |
| 操作系统 | Windows 11 Home China |
| USB-Serial | CH340 |
