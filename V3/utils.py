"""
工具函数: 命令行参数解析、参数量统计、滑动平均器

===== 模块内容 =====
- str2bool:     将命令行字符串转换为布尔值 (用于 argparse)
- count_params: 统计模型可训练参数总量
- AverageMeter: 训练时跟踪 loss/metric 的滑动平均
"""
import argparse


def str2bool(v):
    """命令行布尔值解析: 支持 'true'/'false' 或 1/0。
    用于 argparse 中比 type=bool 更可靠的布尔参数处理"""
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    """统计模型中需要梯度的参数总数 (可训练参数)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """滑动平均器: 跟踪当前值和累计加权均值

    用法:
        meter = AverageMeter()
        for batch in dataloader:
            loss = model(batch)
            meter.update(loss.item(), n=batch.size(0))
        print(f'Average loss: {meter.avg}')
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0      # 最近一次更新的值
        self.avg = 0      # 累计加权平均值
        self.sum = 0      # 累计和 (val * n)
        self.count = 0    # 累计样本数

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
