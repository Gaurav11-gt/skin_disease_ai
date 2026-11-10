"""Tiny fake HAM10000-like dataset for smoke-testing the pipeline (NOT for results)."""
import numpy as np, pandas as pd, cv2
from pathlib import Path
from config import CLASSES
r = Path("data/fake_raw"); r.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0); rows = []
for i in range(300):
    dx = CLASSES[i % 7]; lid = f"L{i // 2}"
    img = (rng.random((90, 120, 3)) * 80 + (CLASSES.index(dx) * 25)).astype(np.uint8)
    cv2.imwrite(str(r / f"IMG{i}.jpg"), img)
    rows.append(dict(lesion_id=lid, image_id=f"IMG{i}", dx=dx))
pd.DataFrame(rows).to_csv(r / "HAM10000_metadata.csv", index=False)
