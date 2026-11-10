"""Evaluation: accuracy, balanced acc, macro-F1, AUC, per-class report, confusion
matrix, ROC curves, calibration (ECE) with temperature scaling fitted on VAL set.

  python evaluate.py --run outputs/hybrid_effnet_s42 --data_dir data/splits
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix, classification_report)
from config import CLASSES


@torch.no_grad()
def get_logits(model, loader, device):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        L.append(model(x.to(device)).float().cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


def fit_temperature(logits, labels):
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=1e-2), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(min=1e-2))


def ece_score(probs, labels, n_bins=15):
    conf, pred = probs.max(1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return float(e)


def compute_metrics(y, probs):
    y, probs = np.asarray(y), np.asarray(probs)
    pred = probs.argmax(1)
    m = {"accuracy": accuracy_score(y, pred),
         "balanced_accuracy": balanced_accuracy_score(y, pred),
         "macro_f1": f1_score(y, pred, average="macro"),
         "weighted_f1": f1_score(y, pred, average="weighted")}
    try:
        m["macro_auc"] = roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=list(range(len(CLASSES))))
    except ValueError:
        m["macro_auc"] = float("nan")
    rec = {}
    for i, c in enumerate(CLASSES):
        mask = y == i
        rec[c] = float((pred[mask] == i).mean()) if mask.any() else float("nan")
    m["per_class_recall"] = rec
    m["melanoma_recall"] = rec["mel"]
    return m


def plots(y, probs, out):
    pred = probs.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES); ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > .5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Normalised confusion matrix")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, c in enumerate(CLASSES):
        if (y == i).any() and (y != i).any():
            fpr, tpr, _ = roc_curve(y == i, probs[:, i])
            ax.plot(fpr, tpr, label=f"{c} (AUC={roc_auc_score(y == i, probs[:, i]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=7)
    ax.set_title("One-vs-rest ROC"); fig.tight_layout(); fig.savefig(out / "roc.png", dpi=200); plt.close(fig)


def full_evaluation(model, va_loader, te_loader, device, out_dir):
    out = Path(out_dir)
    vl, vy = get_logits(model, va_loader, device)
    tl, ty = get_logits(model, te_loader, device)
    T = fit_temperature(vl, vy)
    p_raw, p_cal = F.softmax(tl, 1), F.softmax(tl / T, 1)
    m = compute_metrics(ty.numpy(), p_raw.numpy())
    m["temperature"] = T
    m["ece_before"] = ece_score(p_raw, ty)
    m["ece_after_temp_scaling"] = ece_score(p_cal, ty)
    json.dump(m, open(out / "metrics.json", "w"), indent=2)
    json.dump({"temperature": T}, open(out / "temperature.json", "w"))
    np.save(out / "test_probs.npy", p_raw.numpy()); np.save(out / "test_labels.npy", ty.numpy())
    open(out / "classification_report.txt", "w").write(
        classification_report(ty.numpy(), p_raw.argmax(1).numpy(), target_names=CLASSES, digits=4, zero_division=0))
    plots(ty.numpy(), p_raw.numpy(), out)
    print(json.dumps({k: v for k, v in m.items() if k != "per_class_recall"}, indent=2))
    return m


if __name__ == "__main__":
    from dataset import make_loaders
    from models import build_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data_dir", default="data/splits")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(Path(a.run) / "best.pt", map_location=dev)
    args = ck["args"]
    model = build_model(args["model"], len(CLASSES), pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    _, va, te, _ = make_loaders(a.data_dir, args["img_size"], args["batch_size"], a.workers)
    full_evaluation(model, va, te, dev, a.run)
