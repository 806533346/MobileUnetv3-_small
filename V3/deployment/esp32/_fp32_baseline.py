"""FP32 model baseline: IoU vs GT on full 200-image test set."""
import os, sys, numpy as np, cv2, yaml, torch, albumentations as albu
from glob import glob
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, V3_DIR)
import archs

H, W = 256, 256
img_dir = os.path.join(SCRIPT_DIR, "..", "test_data", "images")
mask_dir = os.path.join(SCRIPT_DIR, "..", "test_data", "masks", "0")
img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
transform = albu.Compose([albu.Resize(H, W), albu.Normalize()])

model_dir = os.path.join(V3_DIR, "models", "Kvasir_SEG2026_MobileNestedUNetv3_QAT_256x256")
with open(os.path.join(model_dir, "config.yml")) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
model = archs.__dict__[cfg["arch"]](cfg["num_classes"], input_channels=cfg["input_channels"],
                                     deep_supervision=False)
model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu"))
model.eval()

results = []
for img_path in tqdm(img_paths):
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    aug = transform(image=img)
    arr = (aug["image"].astype("float32") / 255).transpose(2, 0, 1)
    input_t = torch.from_numpy(arr).unsqueeze(0)

    with torch.no_grad():
        out = model(input_t).squeeze().numpy()
    pred = out > 0

    gt_path = os.path.join(mask_dir, img_id + ".jpg")
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
    gt_bin = (gt.astype("float32") / 255.0) > 0.5

    inter = float((pred & gt_bin).sum())
    union = float((pred | gt_bin).sum())
    iou = inter / max(union, 1)
    results.append((img_id, iou))

ious = np.array([r[1] for r in results])
print()
print("=" * 60)
print("FP32 Model vs GT (200 images)")
print("=" * 60)
print(f"  Mean IoU:   {ious.mean():.4f}")
print(f"  Median IoU: {np.median(ious):.4f}")
print(f"  Min IoU:    {ious.min():.4f}")
print(f"  Max IoU:    {ious.max():.4f}")
print(f"  >= 0.7:     {(ious >= 0.7).sum()}")
print(f"  0.5-0.7:    {((ious >= 0.5) & (ious < 0.7)).sum()}")
print(f"  0.3-0.5:    {((ious >= 0.3) & (ious < 0.5)).sum()}")
print(f"  < 0.3:      {(ious < 0.3).sum()}")

worst = sorted(results, key=lambda x: x[1])[:5]
print(f"\nWorst 5:")
for rid, riou in worst:
    print(f"  {rid}: IoU={riou:.4f}")
