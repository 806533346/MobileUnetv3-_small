"""
数据集加载器: 读取图像和多类别 mask, 通过 albumentations 做数据增强

===== 关键预处理流程 (部署时必须严格复现) =====
训练时的预处理顺序:
  1. cv2.imread() → uint8 [0, 255]
  2. albumentations 增强 (含 albu.Normalize() → float32 ~[-2, 2])
  3. img.astype('float32') / 255  →  ~[-0.008, 0.008]

步骤 3 的 /255 是历史遗留: 训练 pipeline 在此处误将已 Normalize 的
float32 值再次除以 255, 导致模型实际输入值缩小约 255 倍。
模型已适配此分布, 推理时若缺少 /255, 输入值将是训练时的 255 倍,
输出会严重退化 (IoU 从 0.54 跌至 0.005)。

因此 ESP32 部署的 generate_test_data.py / convert_to_espdl.py 中,
预处理必须包含:
  img = augmented['image'].astype('float32') / 255

===== Mask 存储结构 =====
多类别 mask 按类别分目录: mask_dir/0/xxx.jpg, mask_dir/1/xxx.jpg, ...
每个类别文件为单通道灰度图 (JPEG 压缩, 值域非严格 0/255), 通过 /255 归一化到 [0,1]
"""
import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    """语义分割数据集, 支持多类别 mask 和 albumentations 增强

    Args:
        img_ids:    图像文件名列表 (不含扩展名)
        img_dir:    图像目录路径
        mask_dir:   mask 根目录 (mask_dir/0/, mask_dir/1/, ... 各存每个类别)
        img_ext:    图像扩展名 (如 '.jpg')
        mask_ext:   mask 扩展名 (如 '.jpg')
        num_classes: 分割类别数 (1 = 二分类)
        transform:  albumentations Compose 数据增强流水线
    """

    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # Step 1: 读取原始图像 (BGR uint8)
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        # Step 2: 读取 mask — 每个类别存放在独立子目录 mask_dir/{class_id}/{img_id}.jpg
        #         JPEG 压缩导致 mask 值非严格 0/255, 而是 0~8 (暗) 和 247~255 (亮)
        #         通过 /255 归一化 + >0.5 阈值即可得到正确二值 mask
        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        # Step 3: albumentations 数据增强 — image 和 mask 同步变换, 保持空间一致性
        #         变换链包含: RandomRotate90 → Flip → ColorJitter → Resize → albu.Normalize
        #         执行后 img 为 float32, mask 保持为原始值
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Step 4: 后处理 — 注意此处的 /255 !
        #   img:  albu.Normalize 后已是 float32 ~[-2, 2], 再次 /255 → ~[-0.008, 0.008]
        #   mask: uint8 [0, 255] → /255 → float32 [0, 1]
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)           # HWC → CHW (PyTorch 格式)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)          # HWC → CHW

        return img, mask, {'img_id': img_id}
