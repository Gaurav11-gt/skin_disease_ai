"""Image preprocessing: hair removal (black-hat + inpainting), colour constancy
(Shades of Gray) and resize. Used offline (CLI) and at inference (app).

CLI (optional, recommended - runs once, makes training fast):
  python preprocess.py --splits_dir data/splits --out_dir data/processed
Then train with --data_dir data/processed --preprocessed
"""
import argparse
from multiprocessing import Pool
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def remove_hair(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(bgr, mask, 1, cv2.INPAINT_TELEA)


def shades_of_gray(bgr, p=6):
    img = bgr.astype(np.float32)
    illum = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p)
    illum = illum / (np.linalg.norm(illum) + 1e-8)
    img = img / (illum * np.sqrt(3) + 1e-8)
    return np.clip(img, 0, 255).astype(np.uint8)


def preprocess_image(bgr, size=256):
    out = remove_hair(bgr)
    out = shades_of_gray(out)
    if size:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return out


def _work(args):
    src, dst, size = args
    if Path(dst).exists():
        return
    img = cv2.imread(src)
    if img is None:
        return
    cv2.imwrite(dst, preprocess_image(img, size), [cv2.IMWRITE_JPEG_QUALITY, 95])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        df = pd.read_csv(Path(a.splits_dir) / f"{split}.csv")
        dsts = [str(out / "images" / f"{i}.jpg") for i in df["image_id"]]
        jobs = [(s, d, a.size) for s, d in zip(df["path"], dsts)]
        with Pool(a.workers) as p:
            list(tqdm(p.imap_unordered(_work, jobs), total=len(jobs), desc=split))
        df["path"] = dsts
        df.to_csv(out / f"{split}.csv", index=False)
