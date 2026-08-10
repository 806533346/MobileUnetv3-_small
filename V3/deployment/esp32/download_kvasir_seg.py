"""Download and prepare Kvasir-SEG dataset for MobileV3Unet++ training.

Downloads kvasir-seg.zip (~46MB), extracts it, and reorganizes
into the expected structure: V3/inputs/Kvasir_SEG2026/
"""
import os
import sys
import shutil
import zipfile
import urllib.request

URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
DATASET_NAME = "Kvasir_SEG2026"

# Find project root (2 levels up from this script: deployment/esp32 → ../../)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
INPUTS_DIR = os.path.join(PROJECT_ROOT, "inputs")
TARGET_DIR = os.path.join(INPUTS_DIR, DATASET_NAME)
ZIP_PATH = os.path.join(INPUTS_DIR, "kvasir-seg.zip")


def download():
    print(f"Downloading {URL} ...")
    os.makedirs(INPUTS_DIR, exist_ok=True)

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            print(f"\r  {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({pct}%)", end="", flush=True)

    urllib.request.urlretrieve(URL, ZIP_PATH, reporthook)
    print("\nDone.")


def extract():
    print(f"Extracting to {TARGET_DIR} ...")
    os.makedirs(TARGET_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(TARGET_DIR)

    # The zip usually contains images/ and masks/ at the top level.
    # If they're nested in a subfolder, find and move them.
    images_dir = os.path.join(TARGET_DIR, "images")
    masks_dir = os.path.join(TARGET_DIR, "masks")

    if not os.path.isdir(images_dir):
        # Search for nested images/ folder
        for root, dirs, _ in os.walk(TARGET_DIR):
            if "images" in dirs and "masks" in dirs:
                for d in dirs:
                    src = os.path.join(root, d)
                    dst = os.path.join(TARGET_DIR, d)
                    if src != dst:
                        for f in os.listdir(src):
                            shutil.move(os.path.join(src, f), os.path.join(dst, f))
                        os.rmdir(src)
                break

    # Reorganize masks into masks/0/ (single-class format)
    masks_0 = os.path.join(TARGET_DIR, "masks", "0")
    if os.path.isdir(masks_dir) and not os.path.isdir(masks_0):
        os.makedirs(masks_0, exist_ok=True)
        for f in os.listdir(masks_dir):
            src = os.path.join(masks_dir, f)
            if os.path.isfile(src):
                shutil.move(src, os.path.join(masks_0, f))


def cleanup():
    if os.path.isfile(ZIP_PATH):
        os.remove(ZIP_PATH)
        print(f"Removed {ZIP_PATH}")


def verify():
    images_dir = os.path.join(TARGET_DIR, "images")
    masks_dir = os.path.join(TARGET_DIR, "masks", "0")
    n_images = len(os.listdir(images_dir)) if os.path.isdir(images_dir) else 0
    n_masks = len(os.listdir(masks_dir)) if os.path.isdir(masks_dir) else 0
    print(f"\nDataset ready: {TARGET_DIR}")
    print(f"  images: {n_images}")
    print(f"  masks/0: {n_masks}")
    if n_images == 1000 and n_masks == 1000:
        print("  OK — 1000 image/mask pairs")
    else:
        print(f"  WARNING: expected 1000 pairs, got {n_images}/{n_masks}")


def main():
    if os.path.isdir(os.path.join(TARGET_DIR, "images")) and \
       os.path.isdir(os.path.join(TARGET_DIR, "masks", "0")):
        print(f"Dataset already exists at {TARGET_DIR}")
        verify()
        return

    download()
    extract()
    cleanup()
    verify()


if __name__ == "__main__":
    main()
