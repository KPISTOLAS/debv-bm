"""Build a single-class crown-detection set from the Marche photographs (plant cascade, Table 6).

Each labelled crown is located in its source photograph (masked template matching, see
eval_plant_cascade.locate) and written as a YOLO box of class 0 (Tree). The split follows
configs/splits/marche_split_manifest.json, so detector and BioCLIP head see the same
train/val/test photographs. Photographs are downsampled to a long side of 1280 px, as on the
edge nodes. Only the labelled crowns are boxed; other trees in a photograph are unlabelled.

Usage:
  python build_marche_detect.py --archive Image_Dataset.zip \
      --manifest ../configs/splits/marche_split_manifest.json --out data/marche_detect
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_plant_cascade import CROWN, locate, read

LONG_SIDE = 1280


def main() -> None:
    p = argparse.ArgumentParser(description="Marche photographs -> YOLO crown detection set")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    splits = json.loads(args.manifest.read_text(encoding="utf-8"))["photos"]
    zf = zipfile.ZipFile(args.archive)
    crowns: dict[str, list[str]] = {}
    for name in zf.namelist():
        if m := CROWN.search(name):
            crowns.setdefault(m.group(1), []).append(name)
    originals = set(zf.namelist())
    worst, skipped = 0.0, []
    for split, photos in splits.items():
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for photo_id in photos:
            src = f"Image_Dataset/TrainDataset/Original_Images/{photo_id}.jpg"
            if src not in originals:  # nine photographs ship without their original image
                skipped.append(photo_id)
                continue
            photo = read(zf, src)
            h, w = photo.shape[:2]
            rows = []
            for name in crowns[photo_id]:
                (x1, y1, x2, y2), err = locate(photo, read(zf, name))
                worst = max(worst, err)
                rows.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            s = LONG_SIDE / max(h, w)
            cv2.imwrite(str(args.out / "images" / split / f"{photo_id}.jpg"), cv2.resize(photo, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
            (args.out / "labels" / split / f"{photo_id}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [Tree]\n", encoding="utf-8")
    print(f"done; worst template-fit MAE {worst:.2f}; no original image: {sorted(skipped)}")


if __name__ == "__main__":
    main()
