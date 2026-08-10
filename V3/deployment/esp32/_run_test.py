"""Wrapper: reset ESP32, run stream test, save results to JSON."""
import sys, os, time, json, struct, glob
import serial
import serial.tools.list_ports
import numpy as np
import cv2
import albumentations as albu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "test_results.txt")
JSON_FILE = os.path.join(SCRIPT_DIR, "test_results.json")

log_fh = open(LOG_FILE, 'w', buffering=1)

def log(msg):
    print(msg, flush=True)
    log_fh.write(msg + '\n')
    log_fh.flush()

# Reset ESP32
log("Resetting ESP32...")
try:
    ser = serial.Serial('COM5', 115200, timeout=1)
    ser.setDTR(False); ser.setRTS(True); time.sleep(0.1)
    ser.setDTR(True); time.sleep(0.1)
    ser.close()
    log("Reset done, waiting 4s for boot...")
    time.sleep(4)
except Exception as e:
    log(f"Reset failed: {e}")

# === stream test logic (inline to avoid subprocess issues) ===
MAGIC_IMAGE  = 0xFEED0010
MAGIC_RESULT = 0xFEED0030
MAGIC_STOP   = 0xFEEDFFFF
H, W = 256, 256
DATA_SIZE = H * W * 3
MASK_SIZE = H * W // 8  # 8192 packed bits (8 pixels/byte)
INV_SCALE = 16384.0

def read_exact(ser, count, timeout=15):
    buf = bytearray()
    deadline = time.time() + timeout
    while len(buf) < count and time.time() < deadline:
        chunk = ser.read(count - len(buf))
        if chunk:
            buf.extend(chunk)
    return bytes(buf)

def preprocess(img_bgr, transform):
    aug = transform(image=img_bgr)
    nhwc = aug["image"].astype("float32") / 255.0
    quant = np.clip(np.round(nhwc * INV_SCALE), -128, 127).astype(np.int8)
    return quant.tobytes()

def load_gt_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return (mask.astype("float32") / 255.0) > 0.5

# Find port
port = None
for p in serial.tools.list_ports.comports():
    if "CH340" in (p.description or "") or "USB" in (p.description or ""):
        port = p.device
        break

if not port:
    log("ESP32 not found!")
    log_fh.close()
    sys.exit(1)

img_dir = os.path.join(SCRIPT_DIR, "..", "test_data", "images")
mask_dir = os.path.join(SCRIPT_DIR, "..", "test_data", "masks", "0")
img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
log(f"Found {len(img_paths)} test images")

transform = albu.Compose([albu.Resize(H, W), albu.Normalize()])

# === Run test ===
log(f"Opening {port} at 115200...")
ser = serial.Serial(port, 115200, timeout=5)

log("Waiting for ESP32 READY...")
line = b""
while True:
    ch = ser.read(1)
    if ch:
        line += ch
        if line.endswith(b"READY\n"):
            log("ESP32 ready!")
            break
    else:
        log("Timeout waiting for READY")
        ser.close()
        log_fh.close()
        sys.exit(1)

ser.write(b'G')
time.sleep(0.2)
ser.baudrate = 921600
ser.timeout = 10
time.sleep(0.3)
ser.reset_input_buffer()

results = []
start_time = time.time()

for idx, img_path in enumerate(img_paths):
    img_id = os.path.splitext(os.path.basename(img_path))[0]

    img = cv2.imread(img_path)
    data = preprocess(img, transform)

    ser.write(struct.pack("<I", MAGIC_IMAGE))
    ser.write(data)

    magic_bytes = read_exact(ser, 4)
    if len(magic_bytes) < 4:
        log(f"[{idx}] Failed to receive result magic")
        break
    magic = struct.unpack("<I", magic_bytes)[0]
    if magic != MAGIC_RESULT:
        log(f"[{idx}] Bad magic: 0x{magic:08X}")

    # ESP32 also sends 4B infer_time_us (discard for now)
    _time_bytes = read_exact(ser, 4, timeout=5)
    infer_us = struct.unpack("<i", _time_bytes)[0] if len(_time_bytes) >= 4 else -1

    mask_bytes = read_exact(ser, MASK_SIZE, timeout=20)
    if len(mask_bytes) < MASK_SIZE:
        log(f"[{idx}] Incomplete mask: {len(mask_bytes)}/{MASK_SIZE} - skipping")
        continue
    esp32_mask = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8))[:H*W].reshape(H, W).astype(bool)

    mask_path = os.path.join(mask_dir, img_id + ".jpg")
    gt_mask = load_gt_mask(mask_path)

    inter = float((esp32_mask & gt_mask).sum())
    union = float((esp32_mask | gt_mask).sum())
    iou = inter / max(union, 1)
    results.append((img_id, iou))

    elapsed = time.time() - start_time
    log(f"[{idx:3d}] {img_id}: IoU={iou:.4f}  [{elapsed:.0f}s elapsed]")

# Stop
ser.write(struct.pack("<I", MAGIC_STOP))
ser.close()

# === Summary ===
ious = [r[1] for r in results]
summary = {
    "total": len(results),
    "time_s": round(time.time() - start_time, 1),
    "avg_iou": round(float(np.mean(ious)), 4) if ious else 0,
    "min_iou": round(float(np.min(ious)), 4) if ious else 0,
    "max_iou": round(float(np.max(ious)), 4) if ious else 0,
    "le_0_3": sum(1 for x in ious if x < 0.3),
    "0_3_to_0_5": sum(1 for x in ious if 0.3 <= x < 0.5),
    "0_5_to_0_7": sum(1 for x in ious if 0.5 <= x < 0.7),
    "ge_0_7": sum(1 for x in ious if x >= 0.7),
}

with open(JSON_FILE, 'w') as f:
    json.dump(summary, f, indent=2)

log(f"\n{'='*60}")
log(f"Total: {len(results)} images in {time.time() - start_time:.0f}s")
log(f"Average IoU vs GT: {summary['avg_iou']:.4f}")
log(f"IoU range: [{summary['min_iou']:.4f}, {summary['max_iou']:.4f}]")
log(f"Breakdown: <0.3: {summary['le_0_3']}, 0.3-0.5: {summary['0_3_to_0_5']}, 0.5-0.7: {summary['0_5_to_0_7']}, >=0.7: {summary['ge_0_7']}")
log(f"Saved results to {JSON_FILE}")
log("Done.")
log_fh.close()
