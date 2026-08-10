"""
ONNX 导出脚本: 将训练好的 .pth 模型转换为 ONNX 格式
用法: python export_onnx.py --name Kvasir_SEG2026_MobileNestedUNetv3_woDS
"""
import argparse
import os
import yaml
import torch

import archs


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pth model to ONNX')
    parser.add_argument('--name', default='Kvasir_SEG2026_MobileNestedUNetv3_woDS', help='model name, e.g. Kvasir_SEG2026_MobileNestedUNetv3_woDS')
    parser.add_argument('--opset', default=11, type=int, help='ONNX opset version')
    parser.add_argument('--dynamic_batch', default=False, type=lambda x: x.lower() in ['true', '1', 'yes'],
                        help='export with dynamic batch size')
    return parser.parse_args()


def main():
    args = parse_args()

    # 从训练时保存的 config.yml 恢复模型参数
    config_path = os.path.join('models', args.name, 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'config not found: {config_path}')

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    # 动态实例化模型并加载权重
    model = archs.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision'],
    )

    pth_path = os.path.join('models', args.name, 'model.pth')
    state_dict = torch.load(pth_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    if config['deep_supervision']:
        print('warning: deep_supervision=True, model has multiple outputs.')

    # 按训练配置构造 dummy input
    C = config['input_channels']
    H = config['input_h']
    W = config['input_w']
    batch_size = 1

    dummy_input = torch.randn(batch_size, C, H, W)

    onnx_path = os.path.join('models', args.name, 'model.onnx')

    # batch + spatial 维度全部标记为 dynamic, 支持任意尺寸推理
    dynamic_axes = {'input': {0: 'batch', 2: 'height', 3: 'width'},
                    'output': {0: 'batch', 2: 'height', 3: 'width'}}

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'] if not config['deep_supervision']
                      else ['output_%d' % i for i in range(4)],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        dynamo=False,
    )

    print(f'saved to {onnx_path}')

    # 用 onnx 官方库校验模型合法性
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print('ONNX model check passed')


if __name__ == '__main__':
    main()
