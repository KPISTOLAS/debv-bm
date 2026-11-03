"""Image-level bootstrap confidence intervals for detector mAP (Table 1).

Each checkpoint is validated once on the test split with the Table 1 protocol
(Ultralytics val, conf 0.001, NMS IoU 0.5, 640 px). The validator's own per-image
matching statistics are captured, test images are resampled with replacement, and
mAP@0.5 / mAP@0.5:0.95 are recomputed. The same resamples are used for every model of
a domain, so pairwise differences get paired intervals.

Usage:
  python bootstrap_detector_ci.py --data DATA.yaml --weights-dir W --out ci.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils import metrics as ul_metrics
from ultralytics.utils.metrics import ap_per_class


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM detector bootstrap CIs")
    p.add_argument("--data", type=Path, required=True, help="Ultralytics data YAML with a test: split")
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="0")
    return p.parse_args()


def per_image_stats(model: YOLO, data: Path, device: str) -> tuple[dict[str, dict], tuple[float, float]]:
    """Run Ultralytics val on the test split and keep its per-image stats, keyed by image name."""
    captured: dict[str, dict] = {}
    original = ul_metrics.DetMetrics.update_stats

    def capture(self, stat):
        captured[stat["im_name"]] = {k: stat[k] for k in ("tp", "conf", "pred_cls", "target_cls")}
        original(self, stat)

    ul_metrics.DetMetrics.update_stats = capture
    try:
        m = model.val(data=str(data), split="test", imgsz=640, batch=16, conf=0.001, iou=0.5,
                      device=device, plots=False, verbose=False)
    finally:
        ul_metrics.DetMetrics.update_stats = original
    return captured, (float(m.box.map50), float(m.box.map))


def maps(stats: list[dict], idx: np.ndarray) -> tuple[float, float]:
    tp = np.concatenate([stats[i]["tp"] for i in idx])
    conf = np.concatenate([stats[i]["conf"] for i in idx])
    pred_cls = np.concatenate([stats[i]["pred_cls"] for i in idx])
    target_cls = np.concatenate([stats[i]["target_cls"] for i in idx])
    if not len(target_cls):
        return 0.0, 0.0
    ap = ap_per_class(tp, conf, pred_cls, target_cls)[5]
    return float(ap[:, 0].mean()), float(ap.mean())


def ci(v: np.ndarray) -> list[float]:
    return [round(100 * float(np.percentile(v, 2.5)), 2), round(100 * float(np.percentile(v, 97.5)), 2)]


def main() -> None:
    args = parse_args()
    weights = sorted(args.weights_dir.glob("*.pt"))
    stats, val_maps = {}, {}
    for pt in weights:
        per_image, val_maps[pt.stem] = per_image_stats(YOLO(str(pt)), args.data, args.device)
        stats[pt.stem] = per_image
    names = sorted(next(iter(stats.values())))
    if any(sorted(s) != names for s in stats.values()):
        raise SystemExit("models were validated on different image sets")
    stats = {m: [s[n] for n in names] for m, s in stats.items()}

    n = len(names)
    rng = np.random.default_rng(args.seed)
    draws = [rng.integers(0, n, n) for _ in range(args.bootstrap)]
    full = np.arange(n)
    boot = {m: np.array([maps(s, d) for d in draws]) for m, s in stats.items()}

    out = {"images": n, "bootstrap": args.bootstrap, "seed": args.seed, "models": {}, "pairs": {}}
    for m, s in stats.items():
        m50, m5095 = maps(s, full)
        if abs(m50 - val_maps[m][0]) > 1e-4:
            raise SystemExit(f"{m}: recomputed mAP50 {m50:.4f} differs from val {val_maps[m][0]:.4f}")
        out["models"][m] = {
            "mAP50": round(100 * m50, 2), "mAP50_ci95": ci(boot[m][:, 0]),
            "mAP50_95": round(100 * m5095, 2), "mAP50_95_ci95": ci(boot[m][:, 1]),
        }
    for a, b in itertools.combinations(sorted(stats), 2):
        d = boot[a] - boot[b]
        out["pairs"][f"{a}-{b}"] = {
            "d_mAP50": round(out["models"][a]["mAP50"] - out["models"][b]["mAP50"], 2), "d_mAP50_ci95": ci(d[:, 0]),
            "d_mAP50_95": round(out["models"][a]["mAP50_95"] - out["models"][b]["mAP50_95"], 2), "d_mAP50_95_ci95": ci(d[:, 1]),
        }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["models"], indent=1))


if __name__ == "__main__":
    main()
