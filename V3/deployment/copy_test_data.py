"""从 inputs 数据集复制 test 集到 deployment/test_data"""
import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

dataset = 'Kvasir_SEG2026'
img_dir = os.path.join('V3', 'inputs', dataset, 'images')
mask_dir = os.path.join('V3', 'inputs', dataset, 'masks')
img_ext = '.jpg'

img_ids = glob(os.path.join(img_dir, '*' + img_ext))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

# 和训练时一致的划分: 60/20/20
_, rest_ids = train_test_split(img_ids, test_size=0.4, random_state=41)
_, test_ids = train_test_split(rest_ids, test_size=0.5, random_state=41)

print(f'total: {len(img_ids)}, test: {len(test_ids)}')

out_img_dir = os.path.join('deployment', 'test_data', 'images')
out_mask_dir = os.path.join('deployment', 'test_data', 'masks', '0')
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

for img_id in test_ids:
    shutil.copy(os.path.join(img_dir, img_id + img_ext), out_img_dir)
    shutil.copy(os.path.join(mask_dir, '0', img_id + img_ext), out_mask_dir)

print('done')
