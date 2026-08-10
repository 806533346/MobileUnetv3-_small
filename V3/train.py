"""
训练脚本: MobileUNetV3 / MobileNestedUNetv3 语义分割训练
用法: python train.py --dataset Kvasir_SEG2026 --arch MobileNestedUNetv3
"""
import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as albu
from albumentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from metrics import dice_coef, recall, specificity, precision, f1_score, f2_score
from utils import AverageMeter, str2bool

# 可选模型架构和损失函数列表, 由各模块的 __all__ 导出
ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='MobileNestedUNetv3',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: MobileNestedUNetv3)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='Kvasir_SEG2026',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.jpg',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.jpg',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD', 'RMSProp'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD', 'RMSProp']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=20, type=int,
                        metavar='N', help='early stopping (default: 15)')

    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    """一个 epoch 的训练, 返回各指标的平均值"""
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'recall': AverageMeter(),
                  'specificity': AverageMeter(),
                  'precision': AverageMeter(),
                  'f1': AverageMeter(),
                  'f2': AverageMeter()
                  }

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # 深度监督: 4个输出头各自算 loss 取平均, 评估指标用最深层的输出
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
            dice = dice_coef(outputs[-1], target)
            rec = recall(outputs[-1], target)
            spec = specificity(outputs[-1], target)
            prec = precision(outputs[-1], target)
            f1 = f1_score(outputs[-1], target)
            f2 = f2_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            rec = recall(output, target)
            spec = specificity(output, target)
            prec = precision(output, target)
            f1 = f1_score(output, target)
            f2 = f2_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        avg_meters['recall'].update(rec, input.size(0))
        avg_meters['specificity'].update(spec, input.size(0))
        avg_meters['precision'].update(prec, input.size(0))
        avg_meters['f1'].update(f1, input.size(0))
        avg_meters['f2'].update(f2, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
            ('recall', avg_meters['recall'].avg),
            ('specificity', avg_meters['specificity'].avg),
            ('precision', avg_meters['precision'].avg),
            ('f1', avg_meters['f1'].avg),
            ('f2', avg_meters['f2'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('recall', avg_meters['recall'].avg),
                        ('specificity', avg_meters['specificity'].avg),
                        ('precision', avg_meters['precision'].avg),
                        ('f1', avg_meters['f1'].avg),
                        ('f2', avg_meters['f2'].avg),
                        ])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'recall': AverageMeter(),
                  'specificity': AverageMeter(),
                  'precision': AverageMeter(),
                  'f1': AverageMeter(),
                  'f2': AverageMeter()
                  }

    # 切换到评估模式
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                # 深度监督时用最后一个输出 (最深解码层) 计算指标
                iou = iou_score(outputs[-1], target)
                dice = dice_coef(outputs[-1], target)
                rec = recall(outputs[-1], target)
                spec = specificity(outputs[-1], target)
                prec = precision(outputs[-1], target)
                f1 = f1_score(outputs[-1], target)
                f2 = f2_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice = dice_coef(output, target)
                rec = recall(output, target)
                spec = specificity(output, target)
                prec = precision(output, target)
                f1 = f1_score(output, target)
                f2 = f2_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['recall'].update(rec, input.size(0))
            avg_meters['specificity'].update(spec, input.size(0))
            avg_meters['precision'].update(prec, input.size(0))
            avg_meters['f1'].update(f1, input.size(0))
            avg_meters['f2'].update(f2, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('recall', avg_meters['recall'].avg),
                ('specificity', avg_meters['specificity'].avg),
                ('precision', avg_meters['precision'].avg),
                ('f1', avg_meters['f1'].avg),
                ('f2', avg_meters['f2'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('recall', avg_meters['recall'].avg),
                        ('specificity', avg_meters['specificity'].avg),
                        ('precision', avg_meters['precision'].avg),
                        ('f1', avg_meters['f1'].avg),
                        ('f2', avg_meters['f2'].avg),
                        ])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # 损失函数: 通过字符串从 losses 模块动态获取对应类
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # 通过字符串从 archs 模块动态实例化模型
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'], config['deep_supervision'],
                                           input_channels=config['input_channels'])

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'RMSProp':
        optimizer = optim.RMSprop(params, lr=config['lr'], weight_decay=config['weight_decay'])

    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # 数据加载: 按 60%/20%/20% 划分 train/val/test
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    print(len(img_ids))
    train_img_ids, rest_img_ids = train_test_split(img_ids, test_size=0.4, random_state=41)
    val_img_ids, test_img_ids = train_test_split(rest_img_ids, test_size=0.5, random_state=41)

    # 训练增强: 随机旋转90° + 翻转 + 颜色扰动(色相/亮度/对比度其一)
    train_transform = Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        OneOf([
            albu.HueSaturationValue(),
            albu.RandomBrightnessContrast(),
        ], p=1),
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    # 验证增强: 仅 resize + 归一化, 不做数据增强
    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)  # 不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('recall', []),
        ('specificity', []),
        ('precision', []),
        ('f1', []),
        ('f2', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_recall', []),
        ('val_specificity', []),
        ('val_precision', []),
        ('val_f1', []),
        ('val_f2', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        train_log = train(config, train_loader, model, criterion, optimizer)
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
              % (
              train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['recall'].append(train_log['recall'])
        log['specificity'].append(train_log['specificity'])
        log['precision'].append(train_log['precision'])
        log['f1'].append(train_log['f1'])
        log['f2'].append(train_log['f2'])
        log['val_recall'].append(val_log['recall'])
        log['val_specificity'].append(val_log['specificity'])
        log['val_precision'].append(val_log['precision'])
        log['val_f1'].append(val_log['f1'])
        log['val_f2'].append(val_log['f2'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        # 按 val_iou 保存最佳模型
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # 早停: 连续 N 个 epoch 未提升即停止训练
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
