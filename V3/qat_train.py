"""
QAT (Quantization-Aware Training) 微调脚本

在训练时插入伪量化节点，模拟 ESP-DL INT8 幂次方量化噪声，
让模型学会在量化约束下保持精度。

原理:
  forward:  FP32 → INT8 → FP32 (引入量化噪声)
  backward: STE (Straight-Through Estimator) — 梯度直接穿过量化节点

用法:
  python qat_train.py --arch MobileNestedUNetv3 --pretrained models/xxx/model.pth
"""
import argparse
import os
from collections import OrderedDict
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import albumentations as albu
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score, dice_coef, recall, specificity, precision, f1_score, f2_score
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


class Int8FakeQuant(nn.Module):
    """模拟 INT8 对称逐张量量化 + STE

    forward:  clamp(round(x / scale), -128, 127) * scale
    backward: identity (STE — 梯度绕过量化节点)
    """
    def __init__(self):
        super().__init__()
        self.qmin = -128.0
        self.qmax = 127.0

    def forward(self, x):
        if not self.training or x.numel() == 0:
            return x
        max_val = x.abs().max().detach()
        scale = (max_val / self.qmax).clamp(min=1e-8)
        x_q = (x / scale).clamp(self.qmin, self.qmax).round()
        x_dq = x_q * scale
        return x + (x_dq - x).detach()  # STE


# ---- 激活量化插入点 ----
# 在每一组 Conv-BN-Act 之后插入伪量化，模拟 ESP-DL 逐层 INT8 传递
_HOOK_HANDLES = []

def _make_quant_hook(quant):
    def hook(module, input, output):
        return quant(output)
    return hook

def apply_qat_hooks(model):
    """对模型中所有激活层 (ReLU / Hardswish / Hardsigmoid) 插入 fake quant hook"""
    quant = Int8FakeQuant()
    for name, m in model.named_modules():
        if isinstance(m, (nn.ReLU, nn.Hardswish, nn.Hardsigmoid)):
            h = m.register_forward_hook(_make_quant_hook(quant))
            _HOOK_HANDLES.append(h)
    return model

def remove_qat_hooks():
    for h in _HOOK_HANDLES:
        h.remove()
    _HOOK_HANDLES.clear()


# ---- 权重量化 ----
def quantize_weights(model):
    """对 Conv2d 权重做 INT8 对称量化 (in-place)，模拟 Flash 中 INT8 权重"""
    quant = Int8FakeQuant()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = quant(m.weight.data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=None,
                        help='path to FP32 .pth checkpoint')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('--arch', default='MobileNestedUNetv3', choices=ARCH_NAMES)
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)
    parser.add_argument('--dataset', default='Kvasir_SEG2026')
    parser.add_argument('--img_ext', default='.jpg')
    parser.add_argument('--mask_ext', default='.jpg')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='QAT fine-tune 用更低学习率')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--scheduler', default='CosineAnnealingLR')
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--early_stopping', default=20, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    return parser.parse_args()


def train_one_epoch(config, train_loader, model, criterion, optimizer):
    avg_meters = {k: AverageMeter() for k in
                  ['loss', 'iou', 'dice', 'recall', 'specificity', 'precision', 'f1', 'f2']}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda() if torch.cuda.is_available() else input
        target = target.cuda() if torch.cuda.is_available() else target

        # QAT: 每步量化权重 + 激活 hook 自动注入量化噪声
        quantize_weights(model)

        output = model(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice = dice_coef(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        pbar.set_postfix(loss=avg_meters['loss'].avg, iou=avg_meters['iou'].avg,
                         dice=avg_meters['dice'].avg)
        pbar.update(1)
    pbar.close()
    return OrderedDict((k, v.avg) for k, v in avg_meters.items())


def validate(config, val_loader, model, criterion):
    avg_meters = {k: AverageMeter() for k in
                  ['loss', 'iou', 'dice', 'recall', 'specificity', 'precision', 'f1', 'f2']}
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda() if torch.cuda.is_available() else input
            target = target.cuda() if torch.cuda.is_available() else target
            # 验证时也量化权重，模拟部署精度
            quantize_weights(model)
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            pbar.set_postfix(val_iou=avg_meters['iou'].avg, val_dice=avg_meters['dice'].avg)
            pbar.update(1)
        pbar.close()
    return OrderedDict((k, v.avg) for k, v in avg_meters.items())


def main():
    config = vars(parse_args())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model_name = f"{config['dataset']}_{config['arch']}_QAT_{config['input_h']}x{config['input_w']}"
    config['name'] = model_name
    os.makedirs(f'models/{model_name}', exist_ok=True)

    with open(f'models/{model_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # ---- 模型 ----
    model = archs.__dict__[config['arch']](
        config['num_classes'], config['deep_supervision'],
        input_channels=config['input_channels'])

    # 加载预训练 FP32 权重
    if config['pretrained'] and os.path.exists(config['pretrained']):
        print(f'Loading pretrained: {config["pretrained"]}')
        state = torch.load(config['pretrained'], map_location='cpu')
        model.load_state_dict(state, strict=False)
    else:
        print('Warning: no pretrained model, training from scratch')

    # 插入 QAT hooks
    apply_qat_hooks(model)
    model = model.to(device)

    # ---- 损失 ----
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[config['loss']]().to(device)

    cudnn.benchmark = True

    # ---- 优化器 ----
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'],
                                                eta_min=config['min_lr'])

    # ---- 数据 ----
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_ids, rest = train_test_split(img_ids, test_size=0.4, random_state=41)
    val_ids, _ = train_test_split(rest, test_size=0.5, random_state=41)

    train_transform = Compose([
        albu.RandomRotate90(), albu.Flip(),
        OneOf([albu.HueSaturationValue(), albu.RandomBrightnessContrast()], p=1),
        albu.Resize(config['input_h'], config['input_w']), albu.Normalize(),
    ])
    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']), albu.Normalize(),
    ])

    train_loader = torch.utils.data.DataLoader(
        Dataset(train_ids, os.path.join('inputs', config['dataset'], 'images'),
                os.path.join('inputs', config['dataset'], 'masks'),
                config['img_ext'], config['mask_ext'], config['num_classes'], train_transform),
        batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        Dataset(val_ids, os.path.join('inputs', config['dataset'], 'images'),
                os.path.join('inputs', config['dataset'], 'masks'),
                config['img_ext'], config['mask_ext'], config['num_classes'], val_transform),
        batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # ---- 训练 ----
    log = OrderedDict([('epoch', []), ('lr', []), ('loss', []), ('iou', []), ('dice', []),
                       ('val_loss', []), ('val_iou', []), ('val_dice', [])])
    best_iou, trigger = 0, 0

    for epoch in range(config['epochs']):
        print(f'\nEpoch [{epoch+1}/{config["epochs"]}]')
        train_log = train_one_epoch(config, train_loader, model, criterion, optimizer)
        val_log = validate(config, val_loader, model, criterion)
        scheduler.step()

        print(f'loss {train_log["loss"]:.4f} iou {train_log["iou"]:.4f} dice {train_log["dice"]:.4f}  '
              f'val_loss {val_log["loss"]:.4f} val_iou {val_log["iou"]:.4f} val_dice {val_log["dice"]:.4f}')

        for k in log:
            log[k].append(locals().get(k, None) if k in ['epoch', 'lr'] else
                          train_log.get(k, val_log.get(k, 0)))
        log['epoch'][-1] = epoch
        log['lr'][-1] = config['lr']
        pd.DataFrame(log).to_csv(f'models/{model_name}/log.csv', index=False)

        trigger += 1
        if val_log['iou'] > best_iou:
            # 保存时先移除 hook，导出干净权重
            remove_qat_hooks()
            torch.save(model.state_dict(), f'models/{model_name}/model.pth')
            apply_qat_hooks(model)
            best_iou = val_log['iou']
            print(f'=> saved best model (val_iou={best_iou:.4f})')
            trigger = 0
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print('=> early stopping')
            break

    print(f'\nQAT done. Best val_iou={best_iou:.4f}')
    remove_qat_hooks()


if __name__ == '__main__':
    main()
