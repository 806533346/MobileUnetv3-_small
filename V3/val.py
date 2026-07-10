"""
验证/测试脚本: 加载训练好的模型, 在测试集上计算指标并保存分割结果
用法: python val.py --name dsb2018_96_MobileUNetV3_woDS
"""
import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as albu
from albumentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from metrics import dice_coef, recall, specificity, precision, f1_score, f2_score
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='Kvasir_SEG2026_MobileNestedUNetv3_woDS',
                        help='model name')

    config = parser.parse_args()

    return config


def main():
    args = parse_args()

    # 从训练时保存的 config.yml 恢复全部配置
    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'], config['deep_supervision'])
    model = model.cuda()

    # 加载训练好的权重
    model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))
    model.eval()

    # 数据划分: 与训练时一致, 取出测试集 (20%)
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, rest_img_ids = train_test_split(img_ids, test_size=0.4, random_state=41)
    _, test_img_ids = train_test_split(rest_img_ids, test_size=0.5, random_state=41)

    test_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()
    avg_dice = AverageMeter()
    avg_recall = AverageMeter()
    avg_specificity = AverageMeter()
    avg_precision = AverageMeter()
    avg_f1 = AverageMeter()
    avg_f2 = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(test_loader, total=len(test_loader)):
            input = input.cuda()
            target = target.cuda()

            # 深度监督模式下取最后一个输出 (最深解码层)
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            rec = recall(output, target)
            spec = specificity(output, target)
            prec = precision(output, target)
            f1 = f1_score(output, target)
            f2 = f2_score(output, target)

            avg_meter.update(iou, input.size(0))
            avg_dice.update(dice, input.size(0))
            avg_recall.update(rec, input.size(0))
            avg_specificity.update(spec, input.size(0))
            avg_precision.update(prec, input.size(0))
            avg_f1.update(f1, input.size(0))
            avg_f2.update(f2, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            # 保存分割结果: sigmoid 输出 * 255 写入 jpg
            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)
    print('Dice Coefficient: %.4f' % avg_dice.avg)
    print('Recall: %.4f' % avg_recall.avg)
    print('Specificity: %.4f' % avg_specificity.avg)
    print('Precision: %.4f' % avg_precision.avg)
    print('F1 Score: %.4f' % avg_f1.avg)
    print('F2 Score: %.4f' % avg_f2.avg)

    plot_examples(input, target, model, num_examples=3)

    torch.cuda.empty_cache()


def plot_examples(datax, datay, model, num_examples=6):
    """可视化对比: 原图 / 预测分割 (阈值0.4) / 真值"""
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18, 4 * num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx + 1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][0].set_title("Original Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0, :, :].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][2].set_title("Target image")
    plt.show()


if __name__ == '__main__':
    main()