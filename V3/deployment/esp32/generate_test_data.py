"""生成 ESP32 测试用二进制数据: 批量预处理输入 + ONNX 期望输出

===== 输出文件 =====
1. storage.bin — ESP32 Flash 存储分区镜像, 布局如下:
   [input_0]...[input_N-1][output_0]...[output_N-1][prediction_space]
   其中 prediction_space 预留给 ESP32 写入预测结果
2. test_config.h — 自动生成的 C 头文件, 定义图像尺寸/数量/步长等宏

===== 数据格式 =====
- 输入: float32 NCHW (3×256×256), 每个 786,432 bytes
- 输出: float32 (256×256), 每个 262,144 bytes (ONNX 模型 raw logits)
- 预测空间: float32 (256×256), 每个 262,144 bytes
- 总大小: NUM_IMAGES × (786432 + 262144 + 262144) = 5 × 1310720 = 6.25 MB

===== 预处理 (必须与训练时一致) =====
  1. albu.Resize(256, 256)
  2. albu.Normalize()            → float32 ~[-2, 2]
  3. .astype('float32') / 255    → float32 ~[-0.008, 0.008]

匹配训练代码 dataset.py:46 的预处理流程。

===== 用法 =====
  python generate_test_data.py
  生成 storage.bin (烧录到 Flash 0x610000) 和 test_config.h
"""
import os, sys
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, V3_DIR)

import cv2, numpy as np, albumentations as albu, onnxruntime as ort

NUM_IMAGES = 5   # 测试图像数量 (ESP32 批量推理)

def preprocess(img_bgr, transform):
    """预处理单张 BGR 图像: albumentations 增强 → float32 → NCHW"""
    augmented = transform(image=img_bgr)
    img = augmented['image'].astype('float32') / 255
    return img.transpose(2, 0, 1)   # HWC → CHW


def main():
    img_dir = os.path.join(SCRIPT_DIR, '..', 'test_data', 'images')
    onnx_path = os.path.join(V3_DIR, 'models', 'Kvasir_SEG2026_MobileNestedUNetv3_woDS', 'model.onnx')
    out_dir = os.path.join(SCRIPT_DIR, 'esp32_inference', 'mobilenet', 'main')

    os.makedirs(out_dir, exist_ok=True)

    # 与训练 val_transform 一致: Resize + Normalize (训练后还有 /255)
    transform = albu.Compose([albu.Resize(256, 256), albu.Normalize()])
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # 读取图像列表 (取前 NUM_IMAGES 张)
    img_ids = sorted([os.path.splitext(os.path.basename(p))[0]
                      for p in glob(os.path.join(img_dir, '*.jpg'))])
    img_ids = img_ids[:NUM_IMAGES]

    C, H, W = 3, 256, 256
    input_size = C * H * W            # 196608 个 float32
    output_size = H * W               # 65536 个 float32 (单通道)
    input_bytes = input_size * 4      # 786432 bytes
    output_bytes = output_size * 4    # 262144 bytes
    stride = input_bytes + output_bytes + output_bytes  # 每张图像在 storage 中的占位 = 1310720

    # 打包所有输入/输出为连续 buffer
    combined_input = bytearray()
    combined_output = bytearray()

    for img_id in img_ids:
        img = cv2.imread(os.path.join(img_dir, img_id + '.jpg'))
        inp = preprocess(img, transform).astype(np.float32)  # NCHW float32
        output = session.run(None, {input_name: inp[None]})[0]  # ONNX 推理
        pred = output[0, 0].astype(np.float32)                   # 单通道 logits
        combined_input += inp.tobytes()
        combined_output += pred.tobytes()
        print(f'  {img_id}')

    # ESP32 storage 分区布局:
    # [{input_0}...{input_4}][{output_0}...{output_4}][prediction_space]
    storage = bytearray(NUM_IMAGES * stride)
    for i in range(NUM_IMAGES):
        # 输入区: 第 i 张图像紧接排列
        in_start = i * input_bytes
        # 输出区: 在所有输入之后
        out_start = NUM_IMAGES * input_bytes + i * output_bytes
        storage[in_start:in_start + input_bytes] = combined_input[i * input_bytes:(i + 1) * input_bytes]
        storage[out_start:out_start + output_bytes] = combined_output[i * output_bytes:(i + 1) * output_bytes]
    # 预测区从 NUM_IMAGES * (input_bytes + output_bytes) 开始, 预留给 ESP32 写入

    storage_path = os.path.join(SCRIPT_DIR, 'esp32_inference', 'mobilenet', 'storage.bin')
    with open(storage_path, 'wb') as f:
        f.write(storage)

    print(f'\nTotal: {NUM_IMAGES} images, storage.bin: {len(storage)} bytes ({len(storage)/1024/1024:.1f} MB)')

    # 自动生成 test_config.h — C 预处理器宏, 供 main.cpp / pack_storage_bin.py 使用
    header = f'''// Auto-generated for {NUM_IMAGES} images
#pragma once
#define NUM_TEST_IMAGES  {NUM_IMAGES}
#define TEST_INPUT_C     {C}
#define TEST_INPUT_H     {H}
#define TEST_INPUT_W     {W}
#define TEST_INPUT_SIZE  {input_size}
#define TEST_OUTPUT_SIZE {output_size}
#define TEST_STRIDE      {stride}
'''
    with open(os.path.join(out_dir, 'test_config.h'), 'w') as f:
        f.write(header)
    print(f'config: test_config.h')
    print('done')


if __name__ == '__main__':
    main()
