"""Fine-tune YOLOv11 on a YOLO-format data.yaml (relative paths; CLI only)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM YOLO fine-tune")
    p.add_argument("--data", type=Path, required=True, help="Ultralytics data.yaml")
    p.add_argument("--weights", type=str, default="yolo11l.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="debvbm_yolo")
    p.add_argument("--project", type=Path, default=Path("runs/detect"))
    p.add_argument("--val-only", action="store_true")
    p.add_argument("--split", default="test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Missing data.yaml: {args.data}")
    model = YOLO(str(args.weights))
    if args.val_only:
        metrics = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, plots=False)
        print(
            {
                "mAP50": float(metrics.box.map50) * 100,
                "mAP50_95": float(metrics.box.map) * 100,
                "precision": float(metrics.box.mp) * 100,
                "recall": float(metrics.box.mr) * 100,
            }
        )
        return
    # Do not pass seed: paper runs used Ultralytics library defaults, not a chosen global seed.
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=str(args.project),
    )


if __name__ == "__main__":
    main()
