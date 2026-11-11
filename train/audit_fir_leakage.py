"""Measure frame-level leakage between the FIR test split and the training/validation splits.

Two checks:
  1. Source video: FIR file names carry the recording and frame number
     (e.g. "-FIR-2023-7-_MP4-3351_jpg.rf.<hash>.jpg"). For every test image from a recording,
     the gap (in frames) to the nearest training/validation frame of the same recording is computed.
  2. Visual near-duplicates: a 64-bit difference hash (dHash, 9x8 grayscale) of every image;
     a test image is a near-duplicate if some training/validation image is within a small
     Hamming distance.

Usage:
  python audit_fir_leakage.py --root FIR_ROOT --out fir_leakage.json [--max-hamming 6]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

NAME = re.compile(r"^(?P<video>.+?)[-_](?P<frame>\d+)_(?:jpg|png|jpeg)\.rf\.", re.I)


def dhash(path: Path) -> int:
    long = "\\\\?\\" + str(path) if os.name == "nt" else str(path)  # paths over 260 characters on Windows
    g = np.asarray(Image.open(long).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-hamming", type=int, default=6)
    args = ap.parse_args()

    split_dirs = {"train": "train", "val": "valid", "test": "test"}
    files = {k: sorted((args.root / v / "images").resolve().glob("*")) for k, v in split_dirs.items()}
    ref = files["train"] + files["val"]

    # 1. source video and frame gap
    ref_frames: dict[str, list[int]] = {}
    for p in ref:
        m = NAME.match(p.name)
        if m:
            ref_frames.setdefault(m["video"], []).append(int(m["frame"]))
    gaps, no_video, unparsed = [], 0, 0
    test_video = Counter()
    for p in files["test"]:
        m = NAME.match(p.name)
        if not m:
            unparsed += 1
            continue
        test_video[m["video"]] += 1
        frames = ref_frames.get(m["video"])
        if not frames:
            no_video += 1
            continue
        gaps.append(min(abs(int(m["frame"]) - f) for f in frames))
    gaps = np.array(gaps)

    # 2. visual near-duplicates
    ref_h = np.array([dhash(p) for p in ref], dtype=np.uint64)
    near, dists = 0, []
    for p in files["test"]:
        h = np.uint64(dhash(p))
        x = np.bitwise_xor(ref_h, h)
        d = int(min(bin(int(v)).count("1") for v in x))
        dists.append(d)
        near += d <= args.max_hamming
    dists = np.array(dists)

    n = len(files["test"])
    out = {
        "test_images": n,
        "train_val_images": len(ref),
        "test_named_by_video": int(n - unparsed),
        "test_video_also_in_train_val": int(len(gaps)),
        "test_video_not_in_train_val": int(no_video),
        "test_unparsed_names": int(unparsed),
        "test_images_per_video": dict(test_video.most_common()),
        "frame_gap_to_nearest_train_val_frame": {
            "median": float(np.median(gaps)) if len(gaps) else None,
            "le_1": int((gaps <= 1).sum()), "le_5": int((gaps <= 5).sum()),
            "le_10": int((gaps <= 10).sum()), "le_30": int((gaps <= 30).sum()),
        },
        "dhash_max_hamming": args.max_hamming,
        "test_with_near_duplicate_in_train_val": int(near),
        "test_with_near_duplicate_pct": round(100 * near / n, 1),
        "hamming_distance_median": float(np.median(dists)),
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "test_images_per_video"}, indent=1))


if __name__ == "__main__":
    main()
