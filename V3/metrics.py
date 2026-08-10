"""
评估指标: IoU, Dice, Recall, Specificity, Precision, F1, F2

===== 输入格式 =====
所有函数接受 sigmoid 之前的 raw logits (任意值域) 和 GT mask (0/1 二值)。
内部自动执行: logits → sigmoid → >0.5 阈值 → 二值预测

===== 指标说明 =====
- IoU (Jaccard): |A∩B| / |A∪B|, 最常用的分割指标
- Dice (F1):     2|A∩B| / (|A|+|B|), 对不平衡更鲁棒
- Recall:        TP / (TP+FN), 漏检率 (高 → 少漏)
- Specificity:   TN / (TN+FP), 误检率 (高 → 少误)
- Precision:     TP / (TP+FP), 预测阳性中真阳比例
- F2:            加权召回率 (beta=2), 漏检代价 > 误检代价
"""
import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    """IoU (Jaccard Index): TP/(TP+FP+FN), 阈值 0.5"""
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()   # logits → prob
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5                                   # 二值化
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    """Dice 系数 (与 F1 等价): 2|A∩B| / (|A|+|B|)"""
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def recall(output, target):
    """召回率 (Sensitivity): TP / (TP+FN) — 正样本中有多少被正确检测"""
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    true_positive = (output * target).sum()
    false_negative = ((1 - output) * target).sum()
    return (true_positive + smooth) / (true_positive + false_negative + smooth)


def specificity(output, target):
    """特异度: TN / (TN+FP) — 负样本中有多少被正确排除"""
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    true_negative = ((1 - output) * (1 - target)).sum()
    false_positive = (output * (1 - target)).sum()
    return (true_negative + smooth) / (true_negative + false_positive + smooth)


def precision(output, target):
    """精确率: TP / (TP+FP) — 阳性预测中有多少是真阳"""
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    true_positive = (output * target).sum()
    false_positive = (output * (1 - target)).sum()
    return (true_positive + smooth) / (true_positive + false_positive + smooth)


def f1_score(output, target):
    """F1: 精确率和召回率的调和平均 = 2*P*R/(P+R)"""
    smooth = 1e-5
    recall_val = recall(output, target)
    precision_val = precision(output, target)
    return (2. * recall_val * precision_val) / (recall_val + precision_val + smooth)


def f2_score(output, target, beta=2):
    """F2 (加权 F-score): 召回率权重 = beta²×precision。
    beta=2 → 召回率权重是精确率的 4 倍, 适合"宁可误检不可漏检"的息肉检测场景"""
    smooth = 1e-5
    recall_val = recall(output, target)
    precision_val = precision(output, target)
    return ((1 + beta**2) * recall_val * precision_val) / ((beta**2 * recall_val) + precision_val + smooth)
