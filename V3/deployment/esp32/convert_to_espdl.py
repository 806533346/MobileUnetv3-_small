"""PyTorch 模型 → ESP-DL INT8 量化 → .espdl FlatBuffer 模型文件

===== 量化流程 =====
1. 加载训练好的 PyTorch 模型 (.pth)
2. 用校准数据集 (CalibDataset) 统计各层激活值分布
3. ESP-PPQ (espdl_quantize_torch) 对模型做 Power-of-2 INT8 量化:
   - 输入/输出/权重均量化为 INT8
   - 指数 (exponent) 为 2 的幂, 便于硬件移位运算
   - DL_SCALE(exp) = exp>0 ? 1<<exp : 1.0/(1<<(-exp))
4. 导出为 ESP-DL FlatBuffer 格式 (.espdl), 可直接烧录到 Flash

===== ESP-DL Conv 量化约束 =====
ESP-DL 要求 Conv 算子的量化指数满足:
  output_exponent >= input_exponent + weight_exponent
不满足时 PPQ 会报错, 且 ESP32 硬件上 INT32 累加可能溢出。
Python PPQ 模拟 (TorchExecutor) 不强制执行此约束, 因此 Python 端
验证结果可能优于 ESP32 硬件实测结果。

===== 预处理注意事项 (CRITICAL) =====
训练时 dataset.py:46 在 albu.Normalize() 之后额外执行了 img / 255,
导致模型输入范围约为 [-0.008, 0.008]。当前 CalibDataset.__getitem__
缺少此 /255 步骤 (第 41 行), 导致量化校准和实际 ESP32 推理的输入分布
与训练时不一致。修复方法: 在 astype('float32') 后添加 /255。
"""
import os
import sys
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # V3/deployment/esp32/
V3_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))     # V3/
sys.path.insert(0, V3_DIR)

import cv2
import numpy as np
import albumentations as albu
import torch
import yaml
from torch.utils.data import DataLoader
from esp_ppq.api import espdl_quantize_torch

import archs


class CalibDataset(torch.utils.data.Dataset):
    """量化校准数据集: 提供代表性的输入样本用于统计各层激活范围

    PPQ 在校准阶段运行模型前向传播, 收集每层的 min/max 值,
    据此为每层分配合适的量化指数 (exponent)。

    ⚠️ 预处理必须与训练时完全一致: Resize(256,256) → Normalize → /255
    当前缺少 /255 步骤, 导致校准分布与训练时不匹配。
    """

    def __init__(self, img_dir, img_ext='.jpg'):
        self.img_ids = [os.path.splitext(os.path.basename(p))[0]
                        for p in glob(os.path.join(img_dir, '*' + img_ext))]
        self.img_dir = img_dir
        self.img_ext = img_ext
        # 与训练 val_transform 一致: Resize + Normalize
        self.transform = albu.Compose([
            albu.Resize(256, 256),
            albu.Normalize(),            # 归一化到 mean=0, std=1 → float32 ~[-2, 2]
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        augmented = self.transform(image=img)
        img = augmented['image'].astype('float32') / 255
        img = img.transpose(2, 0, 1)  # HWC→CHW (PyTorch 格式)
        return torch.from_numpy(img)


def main():
    model_name = 'Kvasir_SEG2026_MobileNestedUNetv3_QAT_256x256'

    # 加载训练好的 PyTorch 模型
    with open(os.path.join(V3_DIR, 'models', model_name, 'config.yml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = archs.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=False,              # 部署时关闭深度监督, 仅使用最终输出
    )
    pth_path = os.path.join(V3_DIR, 'models', model_name, 'model.pth')
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()

    espdl_path = os.path.join(SCRIPT_DIR, 'esp32_inference', 'model', 'model.espdl')
    img_dir = os.path.join(SCRIPT_DIR, '..', 'test_data', 'images')

    os.makedirs(os.path.dirname(espdl_path), exist_ok=True)

    dataset = CalibDataset(img_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f'calibration images: {len(dataset)}')
    print('converting via PyTorch path...')

    # ESP-PPQ 量化导出
    espdl_quantize_torch(
        model=model,
        espdl_export_file=espdl_path,        # 输出 .espdl 文件路径
        calib_dataloader=dataloader,          # 校准数据加载器
        calib_steps=min(len(dataset), 50),    # 校准步数: 取前 50 张
        input_shape=[[1, 3, 256, 256]],       # 模型输入形状 (NCHW)
        target='esp32s3',                     # 目标芯片
        num_of_bits=8,                        # 量化位宽: INT8
        export_test_values=False,             # 不导出测试值 (减小文件体积)
        verbose=1,                            # 打印量化详情
    )

    print(f'done: {espdl_path}')


if __name__ == '__main__':
    main()
