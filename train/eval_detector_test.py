"""Score YOLO detector checkpoints on a held-out test split (Table 1).

Every .pt in --weights-dir is evaluated with Ultralytics' standard mAP protocol
(low confidence threshold, NMS IoU 0.5); P/R are reported at the max-F1 confidence,
F1 is their harmonic mean, and inference time is the forward pass per image.
--split val scores the validation split (detector selection); --single-cls merges all
classes into one, i.e. class-agnostic mAP as used by the class-agnostic Tier 2 pipeline.

Usage:
  python eval_detector_test.py --data DATA.yaml --weights-dir WEIGHTS --out results.json [--device cpu]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description="DEBV-BM detector test-split evaluation")
    ap.add_argument("--data", type=Path, required=True, help="Ultralytics data YAML with a test: split")
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--single-cls", action="store_true", help="Class-agnostic mAP")
    ap.add_argument("--pattern", default="*.pt", help="Glob for the checkpoints in --weights-dir")
    args = ap.parse_args()

    results = {}
    for pt in sorted(args.weights_dir.glob(args.pattern)):
        model = YOLO(str(pt))
        m = model.val(data=str(args.data), split=args.split, single_cls=args.single_cls, imgsz=640, batch=16, conf=0.001, iou=0.5,
                      device=args.device, plots=False, verbose=False)
        p, r = float(m.box.mp) * 100, float(m.box.mr) * 100
        results[pt.stem] = {
            "precision": round(p, 2),
            "recall": round(r, 2),
            "f1": round(2 * p * r / (p + r), 2) if p + r else 0.0,
            "mAP50": round(float(m.box.map50) * 100, 2),
            "mAP50_95": round(float(m.box.map) * 100, 2),
            "inference_ms": round(float(m.speed["inference"]), 2),
            "postprocess_ms": round(float(m.speed["postprocess"]), 2),
        }
        print(pt.stem, results[pt.stem])
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
