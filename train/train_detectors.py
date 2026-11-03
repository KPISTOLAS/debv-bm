"""Train detectors locally with the Table 1 protocol (the reproducibility runs).

AdamW, lr0 0.001, batch 16, 640 px, up to 100 epochs, patience 20, seed 0, deterministic,
Ultralytics default augmentation; checkpoints are selected on the data YAML's val split and
best.pt is copied to <runs>/weights_<name>/<model>.pt for eval_detector_test.py.

Usage:
  python train_detectors.py --data plants=path/to/data.yaml --data waid=path/to/waid.yaml \
      --models yolov8n yolo11n --runs runs_yolo
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM local detector training")
    p.add_argument("--data", action="append", required=True, help="NAME=path/to/data.yaml (repeatable)")
    p.add_argument("--models", nargs="+", default=["yolov8n", "yolo11n"])
    p.add_argument("--runs", type=Path, default=Path("runs_yolo"))
    p.add_argument("--device", default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    domains = dict(item.split("=", 1) for item in args.data)
    runs = args.runs.resolve()
    for domain, data in domains.items():
        for name in args.models:
            t0 = time.time()
            print(f"{time.strftime('%H:%M')} start {domain} {name}", flush=True)
            YOLO(f"{name}.pt").train(
                data=str(Path(data).resolve()), epochs=100, batch=16, imgsz=640, optimizer="AdamW", lr0=0.001,
                patience=20, seed=0, deterministic=True, device=args.device, workers=8, plots=False,
                project=str(runs), name=f"{domain}_{name}", exist_ok=True, verbose=False,
            )
            best = runs / f"{domain}_{name}" / "weights" / "best.pt"
            if not best.exists():
                raise SystemExit(f"no best.pt for {domain} {name}")
            dest = runs / f"weights_{domain}" / f"{name}.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, dest)
            print(f"{time.strftime('%H:%M')} done {domain} {name} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
