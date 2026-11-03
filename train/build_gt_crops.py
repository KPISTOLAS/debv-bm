"""Write ground-truth box crops (Table 2 protocol) as an image-folder dataset for train_bioclip.py.

Crops are tight YOLO boxes, as in evaluate_classifier.py. Splits are image-level:
if the source has no validation split, --val-frac of the training *images* (seeded)
becomes validation, so crops of one image never land in two splits.

Usage:
  python build_gt_crops.py --src WAID_ROOT --layout images-labels --config ../configs/diurnal.yaml --out OUT --val-frac 0.2
  python build_gt_crops.py --src FIR_ROOT --layout split-first --config ../configs/nocturnal.yaml --out OUT
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classifier import crop_bgr, list_images, parse_yolo_label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM ground-truth crop dataset")
    p.add_argument("--src", type=Path, required=True)
    p.add_argument(
        "--layout",
        choices=["images-labels", "split-first"],
        required=True,
        help="images-labels: images/<split>, labels/<split>; split-first: <split>/images, <split>/labels",
    )
    p.add_argument("--config", type=Path, required=True, help="YAML with names in label-id order")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-frac", type=float, default=0.2, help="Used only when the source has no val split")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def split_dirs(src: Path, layout: str, split: str) -> tuple[Path, Path]:
    if layout == "images-labels":
        return src / "images" / split, src / "labels" / split
    return src / split / "images", src / split / "labels"


def main() -> None:
    args = parse_args()
    names = list(yaml.safe_load(args.config.read_text(encoding="utf-8"))["names"])

    sources: dict[str, list[tuple[Path, Path]]] = {}
    for split, aliases in {"train": ["train"], "val": ["val", "valid"], "test": ["test"]}.items():
        for alias in aliases:
            img_dir, lbl_dir = split_dirs(args.src, args.layout, alias)
            if img_dir.is_dir():
                sources[split] = [(p, lbl_dir / f"{p.stem}.txt") for p in list_images(img_dir)]
                break
    held_out = []
    if "val" not in sources:
        train = sorted(sources["train"])
        random.Random(args.seed).shuffle(train)
        n_val = round(len(train) * args.val_frac)
        sources["val"], sources["train"] = train[:n_val], train[n_val:]
        held_out = sorted(p.name for p, _ in sources["val"])

    counts = {s: Counter() for s in sources}
    for split, pairs in sources.items():
        for img_path, lbl_path in pairs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for k, gt in enumerate(parse_yolo_label(lbl_path, w, h, len(names))):
                dest = args.out / split / names[gt["cls"]] / f"{img_path.stem}_{k}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest), crop_bgr(img, gt["bbox"]))
                counts[split][names[gt["cls"]]] += 1

    manifest = {
        "src": str(args.src),
        "seed": args.seed,
        "images": {s: len(v) for s, v in sources.items()},
        "crops": {s: dict(sorted(c.items())) for s, c in counts.items()},
        "val_images_from_train": held_out,
    }
    (args.out / "crop_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"images": manifest["images"], "crops": {s: sum(c.values()) for s, c in counts.items()}}))


if __name__ == "__main__":
    main()
