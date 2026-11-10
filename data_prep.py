"""Create leakage-free train/val/test splits for HAM10000.

HAM10000 has several images of the same lesion (lesion_id). A random split puts
near-duplicates in train AND test and inflates accuracy. We split by lesion_id
with stratification (StratifiedGroupKFold) -> ~80/10/10.

Usage:
  python data_prep.py --raw_dir /path/to/HAM10000 --out_dir data/splits
raw_dir must contain HAM10000_metadata.csv and the .jpg files (any subfolders).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from config import CLASS_TO_IDX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    raw = Path(a.raw_dir)
    meta = pd.read_csv(raw / "HAM10000_metadata.csv")
    paths = {p.stem: str(p) for p in raw.rglob("*.jpg")}
    meta["path"] = meta["image_id"].map(paths)
    missing = meta["path"].isna().sum()
    if missing:
        print(f"WARNING: {missing} images listed in metadata not found; dropping them")
        meta = meta.dropna(subset=["path"])
    meta["label"] = meta["dx"].map(CLASS_TO_IDX)

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=a.seed)
    folds = [f[1] for f in sgkf.split(meta, meta["label"], groups=meta["lesion_id"])]
    test_idx, val_idx = folds[0], folds[1]
    train_idx = np.concatenate(folds[2:])

    splits = {"train": meta.iloc[train_idx], "val": meta.iloc[val_idx], "test": meta.iloc[test_idx]}
    # sanity: no lesion shared between splits
    ids = {k: set(v["lesion_id"]) for k, v in splits.items()}
    assert not (ids["train"] & ids["test"]) and not (ids["train"] & ids["val"]) and not (ids["val"] & ids["test"])

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, df in splits.items():
        df[["image_id", "lesion_id", "dx", "label", "path"]].to_csv(out / f"{k}.csv", index=False)
        print(k, len(df), df["dx"].value_counts().to_dict())


if __name__ == "__main__":
    main()
