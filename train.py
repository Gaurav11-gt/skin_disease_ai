"""Train a model. Two-stage transfer learning:
  stage 1 (freeze_epochs): backbone frozen, only the new head trains
  stage 2: whole network fine-tuned (backbone LR = 0.1 x head LR)
Model selection on validation macro-F1; test set touched only once at the end.

  python train.py --model hybrid_effnet --data_dir data/splits --epochs 30 --seed 42
"""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
from config import CLASSES
from dataset import make_loaders
from models import build_model, MODELS
from evaluate import full_evaluation


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def set_backbone_trainable(model, flag):
    for p in model.backbone.parameters():
        p.requires_grad = flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid_effnet", choices=MODELS)
    ap.add_argument("--data_dir", default="data/splits")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--preprocessed", action="store_true", help="data_dir holds hair-removed/colour-constant images")
    a = ap.parse_args()

    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    run = Path(a.out_dir) / f"{a.model}_s{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(run / "args.json", "w"), indent=2)

    tr, va, te, train_df = make_loaders(a.data_dir, a.img_size, a.batch_size, a.workers)
    model = build_model(a.model, len(CLASSES), pretrained=not a.no_pretrained).to(dev)

    counts = train_df["label"].value_counts().reindex(range(len(CLASSES))).fillna(1).values.astype(float)
    w = np.ones(len(CLASSES)) if a.no_class_weights else counts.sum() / counts
    w = np.sqrt(w); w = w / w.mean()          # softened inverse-frequency weights
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev), label_smoothing=0.1)

    bb = list(model.backbone.parameters()); bb_ids = {id(p) for p in bb}
    hd = [p for p in model.parameters() if id(p) not in bb_ids]
    groups = [{"params": hd, "lr": a.lr}] + ([{"params": bb, "lr": a.lr * 0.1}] if bb else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    freeze = a.freeze_epochs if (bb and not a.no_pretrained) else 0
    hist, best_f1, bad = [], -1.0, 0
    for ep in range(a.epochs):
        frozen = ep < freeze
        set_backbone_trainable(model, not frozen)
        model.train()
        if frozen:
            model.backbone.eval()             # keep BN statistics of the pretrained net
        tl, n, t0 = 0.0, 0, time.time()
        for x, y in tqdm(tr, desc=f"ep{ep + 1}/{a.epochs}{' [frozen]' if frozen else ''}", leave=False):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(y); n += len(y)
        sched.step()

        model.eval(); P, Y, vl = [], [], 0.0
        with torch.no_grad():
            for x, y in va:
                o = model(x.to(dev)).float().cpu()
                vl += nn.functional.cross_entropy(o, y).item() * len(y)
                P.append(o.argmax(1)); Y.append(y)
        P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
        vf1 = f1_score(Y, P, average="macro"); vacc = float((P == Y).mean())
        hist.append({"epoch": ep + 1, "train_loss": tl / n, "val_loss": vl / len(Y), "val_acc": vacc, "val_macro_f1": vf1})
        print(f"ep{ep + 1:02d} train_loss={tl / n:.4f} val_loss={vl / len(Y):.4f} val_acc={vacc:.4f} val_f1={vf1:.4f} ({time.time() - t0:.0f}s)")
        json.dump(hist, open(run / "history.json", "w"), indent=2)

        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            torch.save({"model": model.state_dict(), "args": vars(a), "classes": CLASSES}, run / "best.pt")
        else:
            bad += 1
            if bad >= a.patience and ep >= freeze:
                print("early stopping"); break

    ck = torch.load(run / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    full_evaluation(model, va, te, dev, run)
    print(f"done -> {run}")


if __name__ == "__main__":
    main()
