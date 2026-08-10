"""PC INT8 (PPQ) vs GT — directly comparable to ESP32 INT8 vs GT."""
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
img_dir = os.path.join(SCRIPT_DIR, "..", "test_data", "images")
mask_dir = os.path.join(SCRIPT_DIR, "..", "test_data", "masks", "0")
img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
transform = albu.Compose([albu.Resize(H, W), albu.Normalize()])

# Build calib data (first 10 images)
calib_imgs = []
for p in img_paths[:10]:
    img = cv2.imread(p)
    aug = transform(image=img)
    arr = (aug["image"].astype("float32") / 255).transpose(2, 0, 1)
    calib_imgs.append(torch.from_numpy(arr))
calib_data = torch.stack(calib_imgs)
calib_loader = DataLoader(TensorDataset(calib_data), batch_size=1)

# Load model & quantize
model_dir = os.path.join(V3_DIR, "models", "Kvasir_SEG2026_MobileNestedUNetv3_QAT_256x256")
with open(os.path.join(model_dir, "config.yml")) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
model = archs.__dict__[cfg["arch"]](cfg["num_classes"], input_channels=cfg["input_channels"],
                                     deep_supervision=False)
model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu"))
model.eval()

print("Quantizing...")
quant_graph = espdl_quantize_torch(
    model=model,
    espdl_export_file=os.path.join(SCRIPT_DIR, "esp32_inference", "model", "_test_pc_int8.espdl"),
    calib_dataloader=calib_loader, calib_steps=10,
    input_shape=[[1, 3, H, W]], target="esp32s3", num_of_bits=8, verbose=0,
)
executor = TorchExecutor(quant_graph, device='cpu')
input_name = (list(quant_graph.inputs.keys())[0] if isinstance(quant_graph.inputs, dict)
              else quant_graph.inputs[0].name)

# Test vs GT with ESP32-equivalent threshold (-2)
results_int8 = []
results_fp32 = []

for img_path in tqdm(img_paths):
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    aug = transform(image=img)
    arr = (aug["image"].astype("float32") / 255).transpose(2, 0, 1)
    input_t = torch.from_numpy(arr).unsqueeze(0)

    # FP32
    with torch.no_grad():
        fp32_out = model(input_t).squeeze().numpy()
    fp32_bin = fp32_out > 0

    # INT8 (PC PPQ)
    quant_out = executor.forward({input_name: input_t})[0].squeeze().cpu().numpy()
    int8_bin = quant_out > -2  # same threshold as ESP32

    # GT
    gt_path = os.path.join(mask_dir, img_id + ".jpg")
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
    gt_bin = (gt.astype("float32") / 255.0) > 0.5

    # IoU vs GT
    for name, pred in [("FP32", fp32_bin), ("INT8", int8_bin)]:
        inter = float((pred & gt_bin).sum())
        union = float((pred | gt_bin).sum())
        iou = inter / max(union, 1)
        if name == "FP32":
            results_fp32.append((img_id, iou))
        else:
            results_int8.append((img_id, iou))

# Summary
def summarize(results, label):
    ious = np.array([r[1] for r in results])
    print(f"\n{'='*60}")
    print(f"{label} vs GT (200 images)")
    print(f"{'='*60}")
    print(f"  Mean IoU:   {ious.mean():.4f}")
    print(f"  Median IoU: {np.median(ious):.4f}")
    print(f"  >= 0.7:     {(ious >= 0.7).sum()}")
    print(f"  0.5-0.7:    {((ious >= 0.5) & (ious < 0.7)).sum()}")
    print(f"  0.3-0.5:    {((ious >= 0.3) & (ious < 0.5)).sum()}")
    print(f"  < 0.3:      {(ious < 0.3).sum()}")
    return ious

ious_fp32 = summarize(results_fp32, "PC FP32")
ious_int8 = summarize(results_int8, "PC INT8 (thresh=-2)")

print(f"\n{'='*60}")
print(f"{'SUMMARY':^60}")
print(f"{'='*60}")
print(f"  {'PC FP32 vs GT':25s} {ious_fp32.mean():.4f}")
print(f"  {'PC INT8 vs GT':25s} {ious_int8.mean():.4f}")
print(f"  {'ESP32 INT8 vs GT':25s} 0.6115")
print(f"  {'PC INT8 - ESP32':25s} {ious_int8.mean() - 0.6115:+.4f}")
