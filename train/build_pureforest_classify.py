"""Convert PureForest aerial patches to an image-folder classification set (RGB JPG).

Source: Gaydon & Roche 2024, huggingface.co/datasets/IGNF/PureForest (Etalab 2.0).
Download metadata/PureForest-patches.csv and data/imagery-<species>.zip into one folder.
Patches are 250x250 4-band TIFFs in NIR,R,G,B order; we keep R,G,B. The official
forest-level split (metadata `split` column) is kept as is, and the 18 species map to
the 13 official classes (configs/plants_pureforest.yaml).

Usage:
  python build_pureforest_classify.py convert --raw RAW --out data/pureforest            # every imagery zip in RAW
  python build_pureforest_classify.py convert --raw RAW --out data/pureforest Abies_alba # selected species
  python build_pureforest_classify.py zip --out data/pureforest [--cap-train N --cap-val N]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def load_meta(raw: Path) -> dict[str, dict]:
    with open(raw / "PureForest-patches.csv", encoding="utf-8") as f:
        return {r["patch_id"]: r for r in csv.DictReader(f)}


def slug(class_name: str) -> str:
    return class_name.lower().replace(" ", "_")


def convert(raw: Path, out: Path, species: list[str]) -> None:
    meta = load_meta(raw)
    if not species:
        species = sorted(p.stem.removeprefix("imagery-") for p in raw.glob("imagery-*.zip"))
    for sp in species:
        done = skipped = 0
        with zipfile.ZipFile(raw / f"imagery-{sp}.zip") as zf:
            for name in zf.namelist():
                if not name.endswith(".tiff"):
                    continue
                patch_id = Path(name).stem.split("-", 1)[1]  # strip TRAIN-/VAL-/TEST-
                row = meta[patch_id]
                dest = out / row["split"] / slug(row["class_name"]) / f"{patch_id}.jpg"
                if dest.exists():
                    skipped += 1
                    continue
                a = np.array(Image.open(io.BytesIO(zf.read(name))))
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(a[..., 1:4]).save(dest, quality=95)
                done += 1
        print(f"{sp}: converted {done}, already present {skipped}")


def build_zips(out: Path, cap_train: int, cap_val: int) -> None:
    files = defaultdict(list)
    for f in out.rglob("*.jpg"):
        files[(f.parts[-3], f.parts[-2])].append(f)

    full = out.with_name(out.name + "_full.zip")
    capped = out.with_name(out.name + f"_cap{cap_train}.zip")
    rng = random.Random(0)
    counts = {"full": Counter(), "capped": Counter()}
    with zipfile.ZipFile(full, "w", zipfile.ZIP_STORED) as zfull, zipfile.ZipFile(capped, "w", zipfile.ZIP_STORED) as zcap:
        for (split, cls), fs in sorted(files.items()):
            fs.sort()
            for f in fs:
                zfull.write(f, f.relative_to(out).as_posix())
            counts["full"][f"{split}/{cls}"] = len(fs)
            cap = {"train": cap_train, "val": cap_val}.get(split)
            keep = fs if cap is None or len(fs) <= cap else rng.sample(fs, cap)
            for f in sorted(keep):
                zcap.write(f, f.relative_to(out).as_posix())
            counts["capped"][f"{split}/{cls}"] = len(keep)
    manifest = {k: dict(sorted(v.items())) for k, v in counts.items()}
    out.with_name(out.name + "_zip_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for z in (full, capped):
        print(f"wrote {z} ({z.stat().st_size / 1e9:.2f} GB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="PureForest -> image-folder classification set")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--raw", type=Path, required=True, help="Folder with PureForest-patches.csv and imagery-*.zip")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("species", nargs="*", help="Species zips to convert (default: all present)")
    z = sub.add_parser("zip")
    z.add_argument("--out", type=Path, required=True)
    z.add_argument("--cap-train", type=int, default=2000)
    z.add_argument("--cap-val", type=int, default=500)
    args = ap.parse_args()
    if args.cmd == "convert":
        convert(args.raw, args.out, args.species)
    else:
        build_zips(args.out, args.cap_train, args.cap_val)


if __name__ == "__main__":
    main()
