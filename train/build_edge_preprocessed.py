"""Pass a test split through the Raspberry Pi 5 edge preprocessing (train/deploy-mismatch check).

Every test image goes through edge_pi5/capture_preprocess.py exactly as on the edge node:
CLAHE on CIELAB L + mean-lightness rescaling (only with --colour, never with --thermal),
downsampling to a long side of 1280 px, the QC gate (Laplacian variance, mean luma), and
JPEG quality 90. YOLO labels are normalized, so they
are copied unchanged. All images are written (so mAP stays comparable with Table 1); the QC
decision of each image is logged in qc.json. Score the output with eval_detector_test.py and
eval_cascade.py.

Usage:
  python build_edge_preprocessed.py --images TEST/images --labels TEST/labels --names a b c --out data/edge_test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "edge_pi5"))
from capture_preprocess import preprocess, qc_ok

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    p = argparse.ArgumentParser(description="Edge preprocessing of a labelled test split")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--blur-thr", type=float, default=60.0)
    p.add_argument("--dark-thr", type=float, default=12.0)
    p.add_argument("--bright-thr", type=float, default=245.0)
    p.add_argument("--thermal", action="store_true", help="Thermal frames: no colour steps")
    p.add_argument("--colour", action="store_true", help="Apply the RGB colour steps (field-deployment setting)")
    args = p.parse_args()
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    qc = {}
    for img_path in sorted(q for q in args.images.iterdir() if q.suffix.lower() in IMAGE_EXTS):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        bgr = preprocess(bgr, args.max_side, thermal=args.thermal, colour=args.colour)
        ok, reason, stats = qc_ok(bgr, args.blur_thr, args.dark_thr, args.bright_thr)
        qc[img_path.stem] = {"passed": ok, "reason": reason, **{k: round(v, 2) for k, v in stats.items()}}
        cv2.imwrite(str(args.out / "images" / f"{img_path.stem}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        lab = args.labels / f"{img_path.stem}.txt"
        if lab.exists():
            shutil.copy2(lab, args.out / "labels" / lab.name)
    rejected = {k: v["reason"] for k, v in qc.items() if not v["passed"]}
    (args.out / "qc.json").write_text(json.dumps({"n": len(qc), "rejected": len(rejected), "reasons": rejected, "images": qc}, indent=2), encoding="utf-8")
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images\nval: images\ntest: images\nnames: {json.dumps(args.names)}\n", encoding="utf-8")
    print(f"{len(qc)} images, {len(rejected)} rejected by QC: {sorted(set(rejected.values()))}")


if __name__ == "__main__":
    main()
