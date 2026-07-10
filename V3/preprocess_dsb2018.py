"""
DSB2018 数据集预处理: 将原始数据 resize 到指定尺寸, 合并多个 mask 为单一二值 mask
原始格式: inputs/stage1_train/<id>/images/<id>.png + masks/*.png
输出格式: inputs/dsb2018_96/images/*.png + masks/0/*.png
"""
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def main():
    img_size = 96

    paths = glob('inputs/stage1_train/*')

    os.makedirs('inputs/dsb2018_%d/images' % img_size, exist_ok=True)
    os.makedirs('inputs/dsb2018_%d/masks/0' % img_size, exist_ok=True)

    for i in tqdm(range(len(paths))):
        path = paths[i]
        img = cv2.imread(os.path.join(path, 'images',
                         os.path.basename(path) + '.png'))

        # 合并多个实例 mask: 任意一个实例标记为 1 即可
        mask = np.zeros((img.shape[0], img.shape[1]))
        for mask_path in glob(os.path.join(path, 'masks', '*')):
            mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
            mask[mask_] = 1

        # 统一为 3 通道 RGB, 去掉 RGBA 的 alpha 通道
        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        if img.shape[2] == 4:
            img = img[..., :3]

        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))
        cv2.imwrite(os.path.join('inputs/dsb2018_%d/images' % img_size,
                    os.path.basename(path) + '.png'), img)
        cv2.imwrite(os.path.join('inputs/dsb2018_%d/masks/0' % img_size,
                    os.path.basename(path) + '.png'), (mask * 255).astype('uint8'))


if __name__ == '__main__':
    main()
