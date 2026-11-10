"""Collect outputs/*/metrics.json -> mean±std table (Markdown + LaTeX) for the paper."""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

KEYS = [("accuracy", "Acc"), ("balanced_accuracy", "Bal. Acc"), ("macro_f1", "Macro-F1"),
        ("macro_auc", "AUC"), ("melanoma_recall", "Mel. recall"), ("ece_before", "ECE")]
rows = defaultdict(list)
for f in sorted(Path("outputs").glob("*/metrics.json")):
    name = re.sub(r"_s\d+$", "", f.parent.name)
    rows[name].append(json.load(open(f)))

md = ["| Model | n | " + " | ".join(k[1] for k in KEYS) + " |", "|---|---|" + "---|" * len(KEYS)]
tex = ["\\begin{tabular}{l" + "c" * len(KEYS) + "}", "\\hline", "Model & " + " & ".join(k[1] for k in KEYS) + " \\\\ \\hline"]
for name, ms in rows.items():
    cells = []
    for k, _ in KEYS:
        v = np.array([m[k] for m in ms], dtype=float) * (1 if k == "ece_before" else 100)
        cells.append(f"{v.mean():.2f} ± {v.std():.2f}" if k != "ece_before" else f"{v.mean():.3f} ± {v.std():.3f}")
    md.append(f"| {name} | {len(ms)} | " + " | ".join(cells) + " |")
    tex.append(name.replace("_", "\\_") + " & " + " & ".join(c.replace("±", "$\\pm$") for c in cells) + " \\\\")
tex += ["\\hline", "\\end{tabular}"]
Path("results_table.md").write_text("\n".join(md)); Path("results_table.tex").write_text("\n".join(tex))
print("\n".join(md))
