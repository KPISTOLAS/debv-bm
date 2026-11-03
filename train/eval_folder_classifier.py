"""Score a train_bioclip.py checkpoint on an image-folder test split.

Reports overall accuracy (Equation 1), macro precision/recall, F1 as the harmonic
mean of macro P and R (as in Table 2), per-class recall, the confusion matrix,
GPU latency, and 95% bootstrap intervals for accuracy and macro-F1.

Usage:
  python eval_folder_classifier.py --data DATA --config CFG.yaml --weights best.pt --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_prompts import prompt_for_class
from train_bioclip import FolderCrops, load_open_clip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM folder classifier evaluation")
    p.add_argument("--data", type=Path, required=True, help="Root with test/<class>/ images")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--pretrained", default="hf-hub:imageomics/bioclip")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def macro_scores(cm: np.ndarray) -> tuple[float, float, float, float]:
    tp = np.diag(cm).astype(float)
    pred = cm.sum(axis=0).astype(float)
    true = cm.sum(axis=1).astype(float)
    present = true > 0
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)[present].mean()
    rec = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)[present].mean()
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return tp.sum() / cm.sum(), prec, rec, f1


def confusion(y: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    return cm


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    names = list(cfg["names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = load_open_clip(args.model, args.pretrained, device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    ds = FolderCrops(args.data / args.split, {n: i for i, n in enumerate(names)}, preprocess)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    modality = str(cfg.get("modality", "rgb"))
    with torch.no_grad():
        text = F.normalize(model.encode_text(tokenizer([prompt_for_class(n, modality) for n in names]).to(device)), dim=-1)

    ys, ps, gpu_s, n_img = [], [], 0.0, 0
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device, non_blocking=True)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = (F.normalize(model.encode_image(images), dim=-1) @ text.T).argmax(dim=-1)
            if device == "cuda":
                torch.cuda.synchronize()
            gpu_s += time.perf_counter() - t0
            n_img += images.size(0)
            ys.append(y.numpy())
            ps.append(pred.cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    k = len(names)
    cm = confusion(y, p, k)
    acc, prec, rec, f1 = macro_scores(cm)

    rng = np.random.default_rng(0)
    boot_acc, boot_f1 = [], []
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(y), len(y))
        a, _, _, f = macro_scores(confusion(y[idx], p[idx], k))
        boot_acc.append(a)
        boot_f1.append(f)

    def ci(v: list[float]) -> list[float]:
        return [round(100 * float(np.percentile(v, 2.5)), 2), round(100 * float(np.percentile(v, 97.5)), 2)]

    per_class_recall = {n: round(100 * cm[i, i] / cm[i].sum(), 2) if cm[i].sum() else None for i, n in enumerate(names)}
    result = {
        "weights": str(args.weights),
        "split": args.split,
        "n": int(len(y)),
        "accuracy": round(100 * acc, 2),
        "macro_precision": round(100 * prec, 2),
        "macro_recall": round(100 * rec, 2),
        "f1_of_macro_pr": round(100 * f1, 2),
        "accuracy_ci95": ci(boot_acc),
        "f1_ci95": ci(boot_f1),
        "latency_ms_per_image": round(1000 * gpu_s / max(n_img, 1), 2),
        "latency_note": f"image-encoder forward + argmax, batch {args.batch}, {device}",
        "per_class_recall": per_class_recall,
        "support": {n: int(cm[i].sum()) for i, n in enumerate(names)},
        "names": names,
        "confusion": cm.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n", "accuracy", "macro_precision", "macro_recall", "f1_of_macro_pr", "accuracy_ci95", "f1_ci95", "latency_ms_per_image")}))


if __name__ == "__main__":
    main()
