"""
损失函数: BCEDiceLoss (BCE + Dice 联合) 和 LovaszHingeLoss

===== 损失函数对比 =====
- BCEDiceLoss: 像素级 BCE + 区域级 Dice, 各 0.5 权重
  - BCE: 优化每个像素的分类准确性
  - Dice: 优化预测区域与 GT 的重叠度, 对类别不平衡鲁棒
- LovaszHingeLoss: 直接优化 IoU 的替代损失
  - 对 IoU 的优化更直接, 但需要额外安装 LovaszSoftmax 包
  - 适合前景占比小的场景 (如小息肉)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    """BCE + Dice 联合损失

    输入:  raw logits (未经 Sigmoid), 形状 (N, 1, H, W)
    输出: 标量 loss = 0.5 * BCEWithLogits + (1 - Dice)
    """

    def forward(self, input, target):
        # BCEWithLogitsLoss: 内部包含 Sigmoid + BCE, 数值稳定
        bce = F.binary_cross_entropy_with_logits(input, target)

        # Dice Loss: 1 - 2|A∩B|/(|A|+|B|)
        smooth = 1e-5                                    # 防止除零
        input = torch.sigmoid(input)                      # logits → 概率
        num = target.size(0)                              # batch size
        input = input.view(num, -1)                       # (N, H*W)
        target = target.view(num, -1)                     # (N, H*W)
        intersection = (input * target)                    # 逐像素交集
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num                       # batch 平均 dice loss
        return 0.5 * bce + dice                           # 等权组合


class LovaszHingeLoss(nn.Module):
    """Lovasz-Hinge loss: 对 IoU 的连续平滑近似

    需要安装: pip install LovaszSoftmax
    输入: raw logits (N, H, W) 或 (N, 1, H, W)
    """

    def forward(self, input, target):
        input = input.squeeze(1)   # (N, 1, H, W) → (N, H, W)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)
        return loss
