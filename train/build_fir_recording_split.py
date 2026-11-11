"""Re-split FIR by source recording, so that no recording contributes to two splits.

The released FIR split (Roboflow export) is random at image level, and frames of the same
recording appear in train, validation and test (see audit_fir_videos.py). This script pools
all three splits and reassigns whole recordings:

  test : -FIR2023-8-9-_MP4, -FIR2023-2-_MP4   (together they contain all seven classes)
  val  : yt-tC1Y9rwzkrQ, -FIR2022-12-_MP4, -FIR-_mp4
  train: every other recording, including single images without a frame index

Images and labels are copied unchanged. Writes data.yaml and split_manifest.json.

Usage:
  python build_fir_recording_split.py --src FIR_ROOT --out data/fir_recsplit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

PATTERN = re.compile(r"^(.*?)[-_]?(\d+)_(jpg|png|jpeg)\.rf\.")
TEST_DEFAULT = {"-FIR2023-8-9-_MP4", "-FIR2023-2-_MP4"}
VAL_DEFAULT = {"yt-tC1Y9rwzkrQ", "-FIR2022-12-_MP4", "-FIR-_mp4"}
NAMES = ["Bird", "Boar", "Cat", "Deer", "Flying-squirrel", "Fox", "Raccoon"]


def long_path(p: Path) -> str:
    s = str(p.resolve())
    return "\\\\?\\" + s if os.name == "nt" and not s.startswith("\\\\?\\") else s


def recording(stem: str) -> str:
    m = PATTERN.match(stem)
    return m.group(1) if m else "other"


def main() -> None:
    ap = argparse.ArgumentParser(description="FIR recording-grouped split")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test", nargs="+", default=sorted(TEST_DEFAULT), help="Test recordings (default: the paper split)")
    ap.add_argument("--val", nargs="+", default=sorted(VAL_DEFAULT), help="Validation recordings")
    args = ap.parse_args()
    TEST, VAL = set(args.test), set(args.val) - set(args.test)  # noqa: N806
    manifest: dict[str, dict[str, list[str]]] = {"train": {}, "val": {}, "test": {}}
    for split in ("train", "valid", "test"):
        img_dir = long_path(args.src / split / "images")
        lab_dir = long_path(args.src / split / "labels")
        for fname in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(fname)[0]
            rec = recording(stem)
            dest = "test" if rec in TEST else "val" if rec in VAL else "train"
            (args.out / dest / "images").mkdir(parents=True, exist_ok=True)
            (args.out / dest / "labels").mkdir(parents=True, exist_ok=True)
            shutil.copy2(os.path.join(img_dir, fname), long_path(args.out / dest / "images" / fname))
            lab = os.path.join(lab_dir, stem + ".txt")
            if os.path.exists(lab):
                shutil.copy2(lab, long_path(args.out / dest / "labels" / f"{stem}.txt"))
            manifest[dest].setdefault(rec, []).append(stem)
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(NAMES)}\nnames: {json.dumps(NAMES)}\n", encoding="utf-8")
    summary = {s: {r: len(v) for r, v in recs.items() if r in TEST | VAL} | {"n_images": sum(map(len, recs.values()))}
               for s, recs in manifest.items()}
    (args.out / "split_manifest.json").write_text(json.dumps({"test_recordings": sorted(TEST), "val_recordings": sorted(VAL),
                                                             "summary": summary, "images": manifest}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
