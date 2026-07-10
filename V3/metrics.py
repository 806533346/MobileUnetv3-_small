"""
评估指标: IoU, Dice, Recall, Specificity, Precision, F1, F2
所有函数接受 sigmoid 之前的 logits, 内部自动做 sigmoid
"""
import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    """IoU (Jaccard): 交集 / 并集, 阈值 0.5"""
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    """Dice 系数: 2 * |A∩B| / (|A|+|B|)"""
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def recall(output, target):
    """召回率: TP / (TP + FN)"""
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    true_positive = (output * target).sum()
    false_negative = ((1 - output) * target).sum()
    return (true_positive + smooth) / (true_positive + false_negative + smooth)


def specificity(output, target):
    """特异度: TN / (TN + FP)"""
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    true_negative = ((1 - output) * (1 - target)).sum()
    false_positive = (output * (1 - target)).sum()
    return (true_negative + smooth) / (true_negative + false_positive + smooth)


def precision(output, target):
    """精确率: TP / (TP + FP)"""
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    true_positive = (output * target).sum()
    false_positive = (output * (1 - target)).sum()
    return (true_positive + smooth) / (true_positive + false_positive + smooth)


def f1_score(output, target):
    """F1: 精确率和召回率的调和平均"""
    smooth = 1e-5
    recall_val = recall(output, target)
    precision_val = precision(output, target)
    return (2. * recall_val * precision_val) / (recall_val + precision_val + smooth)


def f2_score(output, target, beta=2):
    """F2: 召回率权重更高 (beta=2), 适合漏检代价大的场景"""
    smooth = 1e-5
    recall_val = recall(output, target)
    precision_val = precision(output, target)
    return ((1 + beta**2) * recall_val * precision_val) / ((beta**2 * recall_val) + precision_val + smooth)
