# 端侧 AI 部署全流程：从 PyTorch 训练到 ESP32-S3 推理

> 以 MobileNestedUNetv3 息肉分割模型为例，完整讲解模型部署的五步流水线。

---

## 全景图

```
训练(.pth) → ONNX导出 → 图优化 → 量化校准 → 推理引擎 → 硬件推理
   ↓            ↓         ↓         ↓          ↓          ↓
 PyTorch    torch.onnx  onnx-    ESP-PPQ   .espdl     ESP-DL
  GPU        .export   simplifier (PPQ)  FlatBuffer  UART流式
```

---

## 第 1 步：ONNX 导出 —— `.pth` → `.onnx`

### 1.1 为什么要导出

`.pth` 文件只是 Python 对象的序列化（`torch.save` 的结果），不能脱离 PyTorch 运行。ONNX 是一种**跨框架的通用计算图格式**，用 Protocol Buffers 描述算子和数据流。

### 1.2 导出原理

```python
torch.onnx.export(
    model,          # PyTorch 模型
    dummy_input,    # 假输入——用来"走一遍"图，追踪算子调用
    'model.onnx',
    opset_version=11,           # 算子集版本（不是越高越好——嵌入式引擎有上限）
    dynamic_axes={...},         # 标记动态维度
    dynamo=False,               # 用 legacy TorchScript 导出器
)
```

PyTorch 用 **JIT Tracing** 机制：喂 dummy input，记录每一步调用了哪个 ATen 算子，映射为 ONNX 标准算子。

### 1.3 你的项目配置

- `opset_version=11`：ESP-DL 支持的最高版本
- `dynamic_axes`：batch/height/width 标记为动态，支持多尺寸输入
- `deep_supervision=False`：训练时有 4 个输出头辅助收敛，部署时关闭只保留最终输出

---

## 第 2 步：图优化 —— 让计算图更干净

### 2.1 什么是计算图

ONNX 图是一个**有向无环图（DAG）**：

```
节点 = 算子 (Conv, ReLU, Add, ...)
边   = 张量 (数据流动的方向和形状)
```

```
input → [Conv] → [BN] → [ReLU] → [Conv] → [BN] → [ReLU] → output
        节点1    节点2    节点3     节点4    节点5    节点6
```

每个节点 = 一次 PSRAM 读写。节点越多，内存访问开销越大。

### 2.2 BatchNorm 折叠（BN Folding）

推理时 BN 的参数 $(\gamma, \beta, \mu, \sigma)$ 是固定常量，可以数学等价地"吸收"进前面的 Conv 权重：

$$W' = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot W$$

$$b' = \frac{\gamma(b - \mu)}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

结果：BN 层从图中删除，Conv 节点直接包含了原 BN 的效果。

### 2.3 ReLU 融合

ReLU 不是数学合并，而是**内核级融合**——在执行 Conv 写回内存时，顺便做 `max(0, x)`：

```
融合前：算完Conv → 写回PSRAM → 读出来 → max(0,x) → 再写回  (2次内存读写)
融合后：算完Conv → max(acc,0) → 写回PSRAM                    (1次内存读写)
```

### 2.4 你的项目实际做的图优化

```
当前状态:
  导出用 opset=11，PyTorch 自动做部分 Conv+BN+ReLU 模式识别
  没有显式调用 onnx-simplifier
  Resize 算子 scales 类型从 FLOAT 改为 INT64（ESP-DL 要求）

可优化项:
  onnx-simplifier 消除冗余的 reshape/transpose/identity 节点
  手动 fuse Conv+BN+Hardswish 模块
```

---

## 第 3 步：量化 —— FP32 → INT8

### 3.1 为什么要量化

| | FP32 | INT8 |
|---|---|---|
| 一个参数 | 4 字节 | 1 字节 |
| 一次乘加 | 浮点单元 | 整数 SIMD |
| 内存带宽需求 | 4x | 1x |

ESP32-S3 只有 512KB SRAM + 8MB PSRAM，没有浮点 SIMD。INT8 不是优化选项，是生存前提。

### 3.2 量化公式

$$x_{\text{float}} = x_{\text{int}} \times 2^{\text{exponent}}$$

核心约束：scale 必须是 **2 的幂次**。因为嵌入式硬件上 `× 2^exp` 只是一次移位指令，而任意浮点除法则需要除法器（ESP32-S3 没有硬件除法器）。

```c
// ESP-DL 的 DL_SCALE 宏
#define DL_SCALE(exp)  ((exp) > 0 ? (1 << (exp)) : 1.0 / (1 << (-exp)))

// 例: exponent = -14
// DL_SCALE(-14) = 1.0 / 16384 ≈ 0.000061
```

### 3.3 PPQ 的工作流程

**PPQ（PPL Quantization）**是量化框架，ESP-PPQ 是乐鑫定制版。`espdl_quantize_torch()` 一个函数内部做了五件事：

```
espdl_quantize_torch(model, calib_dataloader, ...)
  │
  ├── 1. 图预处理
  │     追踪前向图 → 折叠 BN → 消除 identity → 插入 Quantizer-Dequantizer 节点对
  │
  ├── 2. 校准 (Calibration)
  │     跑 calib_dataloader 的图片，在每个 Quantizer 处记录激活值分布
  │     收集 min / max / histogram
  │
  ├── 3. 指数分配 (Exponent Assignment)
  │     基于校准统计，在 Power-of-2 约束下搜索每层最优 exponent
  │     同时满足 Conv 约束: output_exponent >= input_exponent + weight_exponent
  │
  ├── 4. 伪量化验证 (Fake Quantization)
  │     用 FP32 模拟 INT8 行为，跑一遍前向，对比原始 FP32 输出
  │
  └── 5. 导出 (Export)
        量化权重 + 每层 exponent 表 + 图结构 → FlatBuffer → model.espdl
```

### 3.4 校准（Calibration）详解

校准**不需要反向传播，不需要梯度**。只跑前向推理，每层挂"钩子"记录数据：

```
跑 50 张校准图，PPQ 在每层后记录激活值:

Layer          min_observed    max_observed    需要覆盖的范围
─────────────────────────────────────────────────────────────
/stem/Conv     -0.008           0.008          [-0.008, 0.008]
/conv2_0/0     -0.003           0.006          [-0.003, 0.006]
/conv4_0/0     -0.127           0.114          [-0.127, 0.114]
/conv5_0/0     -0.896           1.342          [-0.896, 1.342]
/final/Conv    -8.127          12.453          [-8.127, 12.453]
```

给每层找一个 $2^k$，使得 $[-128 \times 2^k,\; 127 \times 2^k]$ 刚好覆盖该层值域。

**校准数据与训练预处理必须严格一致**，否则 exponent 全部偏差，ESP32 推理输出全乱：

```python
# ⚠️ 训练时在 albu.Normalize() 之后还有 img / 255
# 校准时如果漏了这一行，模型看到的输入值域完全不对
img = augmented['image'].astype('float32') / 255  # ← 关键
```

### 3.5 Conv 指数约束

ESP-DL Conv 算子的硬件约束：

```
output_exponent >= input_exponent + weight_exponent
```

推导过程：

1. INT8 输入 × INT8 权重 → INT32 累加
2. 累加结果需要乘以 $2^{e_{in}+e_w}$ 还原为浮点
3. 再除以 $2^{e_{out}}$ 量化为下一层的 INT8 输入
4. 最终移位量 = $e_{in} + e_w - e_{out}$
5. 如果 $e_{out} < e_{in} + e_w$ → 需要左移 → INT32 高位溢出 → 数据作废

所以 $e_{out}$ 只能 $\ge e_{in} + e_w$，意味着**越深的层 exponent 越大，量化粒度越粗**。深层网络在这个约束下容易出现尾层精度崩塌。

---

## 第 4 步：引擎格式 —— `.espdl` FlatBuffer

### 4.1 FlatBuffer vs Protobuf

| | Protobuf (ONNX) | FlatBuffer (ESP-DL) |
|---|---|---|
| 解析方式 | 需要反序列化 | 零拷贝，直接 mmap |
| 体积 | 较大（含字段名） | 更小（纯二进制偏移） |
| 适合场景 | 训练/转换工具链 | 嵌入式 Flash 直接读取 |

### 4.2 Flash 分区布局

```
ESP32 Flash:
  [factory app] [model.espdl] [NVS] [OTA ...]
       ↓              ↓
   main.cpp      g_model = new dl::Model("model",
                  fbs::MODEL_LOCATION_IN_FLASH_PARTITION, ...)
```

### 4.3 模型加载

```cpp
g_model = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
                        0, dl::MEMORY_MANAGER_GREEDY, nullptr, false);
```

ESP-DL 做的事：
1. 从 Flash 分区读出 FlatBuffer → 解析图结构 + INT8 权重
2. 在 PSRAM 中分配每层输入/输出 buffer
3. 构建算子执行队列（贪心内存管理器回收 buffer）

---

## 第 5 步：硬件推理 —— ESP32-S3 执行

### 5.1 推理循环

```cpp
// 构建输入张量（零拷贝——直接在 PSRAM buffer 上包装）
dl::TensorBase *input_tensor = new dl::TensorBase(
    input_shape, input_buf[proc], g_input_exponent,
    dl::DATA_TYPE_INT8, false, EXT_MEM_CAPS);

// 推理 + 计时
int64_t t0 = esp_timer_get_time();
g_model->run(inputs);
int64_t t1 = esp_timer_get_time();

// 反量化 + 二值化 + Bit打包
for (size_t i = 0; i < g_output_count; i++) {
    float v = dl::dequantize(quant_out[i], output_scale);
    if (v > -2) packed[i >> 3] |= (1 << (i & 7));  // 1 bit/pixel
}
```

### 5.2 三层精度验证

```
第1层: PPQ Python 仿真
  FP32 输出 vs 伪量化输出 → 像素级 IoU/Dice/相关系数
  定位: 量化方案是否有问题

第2层: ONNX FP32 基准
  ONNX Runtime FP32 跑全量测试集 → 精度天花板
  定位: ONNX 导出是否有问题

第3层: ESP32 硬件实测
  UART 流式测试 → ESP32 INT8 IoU
  定位: 硬件执行是否有问题（Conv指数约束、累加器精度）
```

### 5.3 流式推理流水线

```
Core 0 (UART RX): 持续接收图像数据到三缓冲
Core 1 (Inference): 取已就绪的缓冲做推理 + 回传结果

三缓冲 + 计数信号量:
  s_buf_free (初始=3): 可用缓冲数 → RX task 消耗
  s_buf_filled (初始=0): 已就绪缓冲数 → Inference 消耗
  fill_idx / proc_idx: 环形指针

每轮推理:
  RX:   xSemaphoreTake(buf_free) → 读数据 → xSemaphoreGive(buf_filled)
  INF:  xSemaphoreTake(buf_filled) → 推理 → 回传结果 → xSemaphoreGive(buf_free)
```

---

## 性能汇总

| 指标 | 数值 |
|------|------|
| 模型参数量 | 26 万（INT8: 256KB） |
| ESP32 推理时间 | 3.8s/张（±30ms，极稳定） |
| 端到端速度 | 5.1s/张（@921600 baud + Packed Bitmask） |
| 优化前基线 | 9.5s/张 |
| 性能提升 | 46% |
| 通信带宽节省 | 87.5%（65KB → 8KB） |

---

## 最容易踩的坑

1. **校准预处理 ≠ 训练预处理** → 量化 exponent 全错，ESP32 输出乱码
2. **ONNX opset 版本过高** → 导出引擎不认识的算子，ESP-DL 加载失败
3. **Conv 指数约束不满足** → INT32 累加溢出，深层输出崩塌
4. **Python 仿真通过但硬件不过** → TorchExecutor 不强制指数约束，总比硬件乐观
5. **UART 波特率与 Buffer 不匹配** → 数据溢出丢帧，推理结果对不上输入
