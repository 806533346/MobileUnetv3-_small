"""
工具函数: 命令行参数解析辅助、参数量统计、滑动平均
"""
import argparse


def str2bool(v):
    """命令行布尔值解析: 支持 true/false, 1/0"""
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """滑动平均: 记录当前值和累计均值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
