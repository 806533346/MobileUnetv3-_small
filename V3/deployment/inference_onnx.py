"""ONNX 本地部署推理，在测试集上计算 IoU 和 Dice"""
import os
import sys
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, V3_DIR)

import cv2
import numpy as np
import onnxruntime as ort
import albumentations as albu
from dataset import Dataset


def main():
    onnx_path = os.path.join(V3_DIR, 'models', 'Kvasir_SEG2026_MobileNestedUNetv3_woDS', 'model.onnx')
    img_dir = os.path.join(SCRIPT_DIR, 'test_data', 'images')
    mask_dir = os.path.join(SCRIPT_DIR, 'test_data', 'masks')
    out_dir = os.path.join(SCRIPT_DIR, 'output')
    img_ext = '.jpg'

    os.makedirs(out_dir, exist_ok=True)

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    img_ids = [os.path.splitext(os.path.basename(p))[0]
               for p in glob(os.path.join(img_dir, '*' + img_ext))]
    print(f'images: {len(img_ids)}')

    # 复用训练时的预处理, 确保与 ONNX 模型训练输入一致
    dataset = Dataset(
        img_ids=img_ids,
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=img_ext,
        mask_ext=img_ext,
        num_classes=1,
        transform=albu.Compose([
            albu.Resize(256, 256),
            albu.Normalize(),
        ]),
    )

    batch_size = 8  # 与训练 config 中一致

    batch_ious, batch_dices, batch_weights = [], [], []
    inter_sum, union_sum, tp_sum, px_sum = 0, 0, 0, 0
    for idx in range(len(dataset)):
        img, mask, meta = dataset[idx]
        img_np = np.array(img, dtype=np.float32)[None]   # (1, C, H, W)
        mask_np = np.array(mask, dtype=np.float32)[0]     # (H, W)

        output = session.run(None, {'input': img_np})[0]  # (1, 1, H, W)
        pred = output[0, 0]                               # (H, W)
        pred_bin = pred > 0

        cv2.imwrite(os.path.join(out_dir, meta['img_id'] + '.jpg'),
                    (pred_bin * 255).astype('uint8'))

        mask_bin = mask_np > 0.5
        inter = (pred_bin & mask_bin).sum()
        union = (pred_bin | mask_bin).sum()
        inter_sum += inter
        union_sum += union
        tp_sum += 2 * inter
        px_sum += pred_bin.sum() + mask_bin.sum()

        # 每 batch_size 张或最后一批, 计算 batch 级指标
        if (idx + 1) % batch_size == 0 or idx == len(dataset) - 1:
            n = (idx % batch_size) + 1  # 当前 batch 的实际样本数
            batch_ious.append((inter_sum + 1e-5) / (union_sum + 1e-5))
            batch_dices.append((tp_sum + 1e-5) / (px_sum + 1e-5))
            batch_weights.append(n)
            inter_sum, union_sum, tp_sum, px_sum = 0, 0, 0, 0

    # 与 val.py AverageMeter 一致: 按 batch 大小加权平均
    print(f'IoU:  {np.average(batch_ious, weights=batch_weights):.4f}')
    print(f'Dice: {np.average(batch_dices, weights=batch_weights):.4f}')
    print(f'Results saved to {out_dir}')


if __name__ == '__main__':
    main()
