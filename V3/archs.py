"""
模型定义: MobileUNetV3 和 MobileNestedUNetv3 (UNet++)
编码器使用 MobileNetV3 倒置残差块, 解码器使用标准 VGGBlock
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import torch.onnx
import torchvision.models as models
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time

__all__ = ['MobileUNetV3', 'MobileNestedUNetv3']


class VGGBlock(nn.Module):
    """标准卷积块: Conv-BN-ReLU 重复两次, 用作解码器基本单元"""
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class MobileUNetV3(nn.Module):
    """U-Net 结构, 编码器用 MobileNetV3 Block, 解码器用 VGGBlock"""
    def __init__(self, num_classes, input_channels=3, act=nn.Hardswish,**kwargs):
        super().__init__()

        # 上采样: 双线性插值, 2倍放大
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # stem: 步长2的卷积, 直接下采样到 1/2
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        # 编码器: 逐级下采样, 通道数递增 16 -> 24 -> 48 -> 96
        self.conv1_0 = Block(3, 16, 16, 16, nn.ReLU, True, 2)
        self.conv2_0 = nn.Sequential(
            Block(3, 16, 72, 24, nn.ReLU, False, 2),
            Block(3, 24, 88, 24, nn.ReLU, False, 1),
        )

        self.conv3_0 = nn.Sequential(
            Block(5, 24, 96, 40, act, True, 2),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 48, act, True, 1),
            Block(5, 48, 144, 48, act, True, 1),
        )
        self.conv4_0 = nn.Sequential(
            Block(5, 48, 288, 96, act, True, 2),
            Block(5, 96, 576, 96, act, True, 1),
            Block(5, 96, 576, 96, act, True, 1),
        )

        nb_filter = [3, 16, 24, 48, 96]
        # 解码器: 逐级上采样 + 跳跃连接, 通道数递减 96 -> 48 -> 24 -> 16 -> 3
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[1]+nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv0_5 = VGGBlock(nb_filter[1] + nb_filter[0], nb_filter[0], nb_filter[0])

        # 1x1 卷积输出最终分割图
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        # 编码路径: 逐级下采样
        x0_0 = self.hs1(self.bn1(self.conv1(input)))   # 1/2
        x1_0 = self.conv1_0(x0_0)                        # 1/4
        x2_0 = self.conv2_0(x1_0)                        # 1/8
        x3_0 = self.conv3_0(x2_0)                        # 1/16
        x4_0 = self.conv4_0(x3_0)                        # 1/32

        # 解码路径: 上采样 + 跳跃连接, 逐级恢复分辨率
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        x0_5 = self.conv0_5(torch.cat([input, self.up(x0_4)], 1))

        output = self.final(x0_5)
        return output

class MobileNestedUNetv3(nn.Module):
    """UNet++ 嵌套结构, 跳跃路径上添加密集卷积块, 支持深度监督"""
    def __init__(self, num_classes, deep_supervision, input_channels=3, act=nn.Hardswish, **kwargs):
        super().__init__()

        nb_filter = [3, 16, 16, 24, 48, 96]
        self.deep_supervision = deep_supervision

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # stem: 步长2卷积, 下采样到 1/2
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        # 编码器: 4级下采样, 通道数 16 -> 24 -> 48 -> 96
        self.conv2_0 = Block(3, 16, 16, 16, nn.ReLU, True, 2)
        self.conv3_0 = nn.Sequential(
            Block(3, 16, 72, 24, nn.ReLU, False, 2),
            Block(3, 24, 88, 24, nn.ReLU, False, 1),
        )

        self.conv4_0 = nn.Sequential(
            Block(5, 24, 96, 40, act, True, 2),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 48, act, True, 1),
            Block(5, 48, 144, 48, act, True, 1),
        )
        self.conv5_0 = nn.Sequential(
            Block(5, 48, 288, 96, act, True, 2),
            Block(5, 96, 576, 96, act, True, 1),
            Block(5, 96, 576, 96, act, True, 1),
        )

        # UNet++ 密集跳跃连接: conv<i>_<j> 表示第 j 层解码器的第 i 个卷积块
        # 第 1 列 (j=1)
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        # 第 2 列 (j=2)
        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        # 第 3 列 (j=3)
        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        # 第 4 列 (j=4)
        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        # 深度监督模式下, 每列都输出一个分割结果
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        # UNet++ 前向传播: 编码器逐级下采样, 同时构建密集跳跃连接
        # x<i>_<j>: i=编码层级(0最浅), j=解码列(越大越深)
        x0_0 = input
        x1_0 = self.hs1(self.bn1(self.conv1(input)))              # stem: 1/2
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1)) # 密集连接: 浅层特征 + 上采样深层特征

        x2_0 = self.conv2_0(x1_0)                                  # 编码: 1/4
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(x2_0)                                  # 编码: 1/8
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(x3_0)                                  # 编码: 1/16
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # 深度监督: 4个解码层各自输出, 辅助梯度传播
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


'''MobileNetV3 基础组件: h-swish 激活、h-sigmoid、SE 注意力模块、倒置残差块'''

import torch.nn as nn
import torch.nn.functional as F


class hswish(nn.Module):
    """Hardswish: x * ReLU6(x+3) / 6, 比 swish 计算更快"""

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    """Hardsigmoid: ReLU6(x+3) / 6"""

    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    """Squeeze-and-Excitation 注意力模块: 自适应调整通道权重"""

    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                                     # squeeze: 全局平均池化
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),  # excitation: 降维
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),  # 升维还原
            nn.Hardsigmoid()                                              # 门控: 0~1 权重
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    """MobileNetV3 倒置残差块: 1x1扩展 -> 深度卷积 -> SE -> 1x1投影, 可选跳跃连接"""

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        # 1x1 扩展卷积: 升维, 增加特征容量
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        # 深度卷积: 逐通道卷积, 实现空间下采样
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        # 1x1 投影卷积: 降维到输出通道数
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        # 跳跃连接: 当输入输出维度不匹配时, 用卷积对齐
        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))  # 扩展
        out = self.act2(self.bn2(self.conv2(out))) # 深度卷积 + 下采样
        out = self.se(out)                          # SE 注意力
        out = self.bn3(self.conv3(out))             # 投影

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)                 # 残差连接 + 激活




def calculate_model_size(model):
    """计算模型总大小 (参数字节数 + 缓冲区字节数), 返回 MB"""
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    all_size = (param_size + buffer_size) / 1024 / 1024
    return all_size

def calculate_fps(model, device, input_size=(3, 224, 224), num_iter=100):
    """计算模型推理 FPS: 预热10次后测量100次推理的平均耗时"""
    model.eval()
    input_tensor = torch.randn(1, *input_size).to(device)
    # 预热 GPU, 排除初次调用的初始化开销
    for _ in range(10):
        _ = model(input_tensor)
    # 计时测量
    start_time = time.time()
    for _ in range(num_iter):
        _ = model(input_tensor)
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_iter / total_time
    return fps

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNestedUNetv3(num_classes=2, deep_supervision=True).to(device)
    summary(model, input_size=(3, 224, 224))

    input_tensor = torch.randn(8, 3, 224, 224).to(device)
    flops = FlopCountAnalysis(model, input_tensor)
    print(flop_count_table(flops))

    model_size = calculate_model_size(model)
    print(f"Model Size: {model_size:.2f} MB")

    fps = calculate_fps(model, device)
    print(f"FPS: {fps:.2f}")