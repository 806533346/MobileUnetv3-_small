# ESP32-S3 MobileNestedUNetv3++ 图像分割推理

将 Kvasir_SEG2026 训练好的 MobileNestedUNetv3++ 模型部署到 ESP32-S3-N16R8 上，使用 ESP-DL v3.3.7 进行 int8 量化推理，通过 Flash 分区中的预存测试图片验证模型正确性（IoU / Dice）。

## 硬件

| 项目 | 规格 |
|------|------|
| 芯片 | ESP32-S3 (revision v0.2) |
| Flash | 16MB (Boya QIO, 运行在 DIO 80MHz) |
| PSRAM | 8MB Octal (AP_3v3, vendor 0x0d) |
| 开发板 | ESP32-S3-N16R8 |

## 软件版本

| 组件 | 版本 |
|------|------|
| ESP-IDF | v6.0.2 |
| ESP-DL | v3.3.7 |
| Python | ESP-IDF 内建: `C:\Espressif\tools\python\v6.0.2\venv` |
| | ML 脚本: conda env `Py` (`E:\anaconda\envs\Py`) |
| 工具链 | xtensa-esp-elf esp-15.2.0_20251204 |
| cmake | 4.0.3 |
| ninja | 1.12.1 |

## 项目结构

```
V3/deployment/esp32/
├── convert_to_espdl.py          # PyTorch → ESP-DL int8 量化 (esp-ppq)
├── generate_test_data.py        # 预处理测试图片 + ONNX 期望输出 → .bin
└── esp32_inference/
    └── mobilenet/
        ├── CMakeLists.txt        # 项目顶层 CMake
        ├── partitions.csv        # 分区表 (16MB Flash)
        ├── sdkconfig.defaults    # Kconfig 默认值 (PSRAM / Flash 配置)
        ├── dependencies.lock     # ESP-IDF 组件版本锁定
        ├── main/
        │   ├── CMakeLists.txt    # 主组件 CMake
        │   ├── idf_component.yml # ESP-DL 依赖声明
        │   ├── main.cpp          # 推理主程序
        │   ├── test_config.h     # 测试图片尺寸常量 (自动生成)
        │   ├── test_input.bin    # float32 输入 (1,474,560 bytes)
        │   └── test_output.bin   # float32 期望输出 (491,520 bytes)
        ├── model/
        │   ├── model.espdl       # int8 量化模型 (385,536 bytes)
        │   └── model.onnx        # ONNX 模型 (用于生成期望输出)
        ├── storage/
        │   └── storage.bin       # 合并后的 Flash 数据 (1,966,080 bytes)
        ├── build/                # ESP-IDF 构建输出
        └── managed_components/   # ESP-DL + dl_fft 组件
```

## Flash 分区表

| 分区名 | 类型 | 偏移 | 大小 | 用途 |
|--------|------|------|------|------|
| nvs | data/nvs | 0x9000 | 24KB | 非易失存储 |
| phy_init | data/phy | 0xf000 | 4KB | PHY 初始化数据 |
| factory | app/factory | 0x10000 | 4MB | 固件 |
| model | data/espdl | 0x410000 | 2MB | .espdl 模型文件 |
| storage | data/fat | 0x610000 | ~10MB | 测试输入 + 期望输出 |

## PC 端准备（只需执行一次）

### 1. 模型量化

```powershell
# 激活 conda 环境
conda activate Py
cd D:\CODE\mobileunetv3++\MobileV3Unet++4\V3\deployment\esp32
python convert_to_espdl.py
```

输出：`esp32_inference/model/model.espdl`

### 2. 生成测试数据

```powershell
python generate_test_data.py
```

输出：
- `esp32_inference/mobilenet/main/test_input.bin` — float32 CHW 输入 (320×384×3)
- `esp32_inference/mobilenet/main/test_output.bin` — float32 期望 mask (320×384)
- `esp32_inference/mobilenet/main/test_config.h` — 尺寸常量 + 图片 ID

### 3. 合并 Flash 数据

```powershell
cd esp32_inference\mobilenet\main
cmd /c "copy /b test_input.bin + test_output.bin ..\storage\storage.bin"
```

## 构建与烧录

```powershell
# 进入项目目录
cd D:\CODE\mobileunetv3++\MobileV3Unet++4\V3\deployment\esp32\esp32_inference\mobilenet

# 设置 ESP-IDF 环境 (手动方式)
$env:IDF_PATH = "C:\esp\v6.0.2\esp-idf"
$env:PATH = "C:\Espressif\tools\python\v6.0.2\venv\Scripts;C:\esp\v6.0.2\esp-idf\tools;C:\Espressif\tools\cmake\4.0.3\bin;C:\Espressif\tools\ninja\1.12.1;C:\Espressif\tools\xtensa-esp-elf\esp-15.2.0_20251204\xtensa-esp-elf\bin;$env:PATH"

# 构建
idf.py build

# 完整烧录 (含分区表 + 固件 + 模型 + 测试数据)
python C:\esp\v6.0.2\esp-idf\components\esptool_py\esptool\esptool.py --chip esp32s3 --port COM3 --baud 921600 --before default_reset --after hard_reset write_flash -z --flash_mode dio --flash_freq 80m --flash_size 16MB 0x0 build\bootloader\bootloader.bin 0x8000 build\partition_table\partition-table.bin 0x10000 build\esp32_seg_inference.bin 0x410000 model\model.espdl 0x610000 storage\storage.bin
```

> **注意：** 如果 sdkconfig 被修改过，需要先 `idf.py fullclean` 再重新构建。

## 串口监控

由于 `idf.py monitor` 在 PowerShell 下不支持 TTY，使用 Python 脚本监控：

```powershell
python -c "
import serial; s = serial.Serial('COM3', 115200, timeout=1)
while True:
    line = s.readline()
    if line: print(line.decode(errors='replace'), end='')
"
```

## 预期输出

推理成功后串口输出示例：

```
I (521) SEG: === APP_MAIN ENTERED ===
I (531) SEG: PSRAM size: 8388608 bytes
I (531) SEG: Model partition: addr=0x410000, size=0x200000
I (541) SEG: Creating Model...
I (551) SEG: Model loaded
I (561) SEG: Input: name=input, exponent=-6, scale=0.0156250000
I (571) SEG: Test input loaded from partition
I (581) SEG: Input quantized, 368640 elements
I (591) SEG: Running inference...
I (xxxx) SEG: Inference time: xxxx ms
I (xxxx) SEG: ==== Results ====
I (xxxx) SEG: IoU:  0.6840
I (xxxx) SEG: Dice: 0.8124
I (xxxx) SEG: Test image: cju0qkwl35piu0993l0dewei2
I (xxxx) SEG: Done
```

## 当前状态

| 步骤 | 状态 |
|------|------|
| ONNX 导出 | ✓ 完成 |
| ESP-DL int8 量化 | ✓ 完成 |
| 测试数据生成 | ✓ 完成 |
| ESP-IDF 项目构建 | ✓ 完成 |
| 固件启动 | ✓ 正常 (无 PSRAM 时) |
| PSRAM 初始化 | **✗ 阻塞 — Octal PSRAM 初始化时 MSPI 总线挂死** |
| 模型加载 | ✗ 需 PSRAM (内部 391KB RAM 不够) |
| 推理验证 | ✗ 待 PSRAM 修复后进行 |

### 已知问题：PSRAM 初始化挂死

现象：启用 `CONFIG_SPIRAM=y` + Octal 模式后，系统在 `esp_psram_impl_enable()` → `s_check_psram_connected()` 中调用 `esp_rom_opiflash_exec_cmd()` 时死锁，该函数永不返回。

- 芯片为 AP_3v3 8MB Octal PSRAM (vendor 0x0d)
- Quad 模式不会挂死，但检测到错误的 PSRAM 线模式
- Octal 模式 (80MHz / 40MHz) 均挂死
- Flash 为 Boya QIO 芯片，运行在 DIO 80MHz
- 问题可能与 MSPI 总线共享或 OPI DTR 命令时序有关

### 待尝试方案

1. 尝试 `CONFIG_SPIRAM_BOOT_INIT=n` 跳过启动时 PSRAM 初始化
2. 调整 CS hold/setup 时序参数
3. 检查 Flash 和 PSRAM 共享 MSPI 总线的兼容性
4. 查阅 ESP-IDF v6.0.2 Octal PSRAM 相关已知 issue
5. 尝试 Flash 降到 40MHz 避免总线竞争
