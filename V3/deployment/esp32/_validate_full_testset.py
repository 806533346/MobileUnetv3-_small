"""全量测试集: QAT 模型 FP32 vs PPQ INT8 保真度对比"""
import os, sys, numpy as np, cv2, yaml, torch, albumentations as albu
from glob import glob
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, V3_DIR)
import archs
from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor import TorchExecutor

H, W = 256, 256
img_dir = os.path.join(SCRIPT_DIR, '..', 'test_data', 'images')
transform = albu.Compose([albu.Resize(H, W), albu.Normalize()])

img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))
print(f'Total test images: {len(img_paths)}')

# ---- 构建校准数据 (取前10张) ----
calib_imgs = []
for p in img_paths[:10]:
    img = cv2.imread(p)
    aug = transform(image=img)
    arr = (aug['image'].astype('float32') / 255).transpose(2, 0, 1)
    calib_imgs.append(torch.from_numpy(arr))
calib_data = torch.stack(calib_imgs)
calib_loader = DataLoader(TensorDataset(calib_data), batch_size=1)


def load_model(model_dir):
    with open(os.path.join(model_dir, 'config.yml')) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    m = archs.__dict__[cfg['arch']](cfg['num_classes'], input_channels=cfg['input_channels'],
                                     deep_supervision=False)
    m.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location='cpu'))
    m.eval()
    return m


def quantize_model(model, name):
    """量化模型, 返回 TorchExecutor"""
    quant_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=os.path.join(SCRIPT_DIR, f'esp32_inference/model/_test_{name}.espdl'),
        calib_dataloader=calib_loader, calib_steps=10,
        input_shape=[[1, 3, H, W]], target='esp32s3', num_of_bits=8, verbose=0,
    )
    executor = TorchExecutor(quant_graph, device='cpu')
    input_name = (list(quant_graph.inputs.keys())[0] if isinstance(quant_graph.inputs, dict)
                  else quant_graph.inputs[0].name)
    return executor, input_name


def process_image(img_path):
    img = cv2.imread(img_path)
    aug = transform(image=img)
    arr = (aug['image'].astype('float32') / 255).transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0)


# ---- 加载模型 ----
print('Loading models...')
model_dir = os.path.join(V3_DIR, 'models', 'Kvasir_SEG2026_MobileNestedUNetv3_QAT_256x256')
model = load_model(model_dir)

# ---- 量化 ----
print('Quantizing model (this takes ~2 min)...')
executor, input_name = quantize_model(model, 'fullset')

# ---- 全量测试 ----
print(f'Testing {len(img_paths)} images...')
results = []
for img_path in tqdm(img_paths):
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    input_t = process_image(img_path)

    with torch.no_grad():
        fp32_out = model(input_t).squeeze().numpy()

    quant_out = executor.forward({input_name: input_t})[0].squeeze().cpu().numpy()

    # FP32 vs INT8 保真度
    f_bin = fp32_out > 0
    q_bin = quant_out > 0
    inter = (f_bin & q_bin).sum()
    iou = inter / max((f_bin | q_bin).sum(), 1)
    dice = 2 * inter / max(f_bin.sum() + q_bin.sum(), 1)
    corr = np.corrcoef(fp32_out.flatten(), quant_out.flatten())[0, 1]
    max_diff = np.abs(fp32_out - quant_out).max()

    results.append({'id': img_id, 'iou': iou, 'dice': dice, 'corr': corr, 'max_diff': max_diff})

# ---- 汇总 ----
ious = np.array([r['iou'] for r in results])
dices = np.array([r['dice'] for r in results])
corrs = np.array([r['corr'] for r in results])
diffs = np.array([r['max_diff'] for r in results])

print(f'\n{"="*60}')
print(f'QAT 模型 FP32 vs INT8 全量测试 ({len(results)} 张)')
print(f'{"="*60}')
print(f'{"Metric":<20} {"Mean":>10} {"Median":>10} {"Min":>10} {"Max":>10}')
print(f'{"IoU":20} {ious.mean():10.4f} {np.median(ious):10.4f} {ious.min():10.4f} {ious.max():10.4f}')
print(f'{"Dice":20} {dices.mean():10.4f} {np.median(dices):10.4f} {dices.min():10.4f} {dices.max():10.4f}')
print(f'{"Correlation":20} {corrs.mean():10.4f} {np.median(corrs):10.4f} {corrs.min():10.4f} {corrs.max():10.4f}')
print(f'{"MaxDiff":20} {diffs.mean():10.4f} {np.median(diffs):10.4f} {diffs.min():10.4f} {diffs.max():10.4f}')

print(f'\nIoU 分布:')
for th in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    n = (ious >= th).sum()
    print(f'  >= {th:.2f}: {n:4d} 张 ({n/len(ious)*100:5.1f}%)')

# 最差 5 张
worst = sorted(results, key=lambda x: x['iou'])[:5]
print(f'\n最差 5 张 (FP32vsINT8 IoU):')
for r in worst:
    print(f'  {r["id"]}: IoU={r["iou"]:.4f} Dice={r["dice"]:.4f} MaxDiff={r["max_diff"]:.4f}')
