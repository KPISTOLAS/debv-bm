"""Cross-split near-duplicate audit with perceptual hashes (content-based, not file names).

For every test image, the 64-bit DCT perceptual hash (pHash) is compared with the hashes of
all training and validation images; the minimum Hamming distance is recorded. Near-identical
frames (re-exports, augmentations, consecutive video or flight frames with little motion)
have small distances. Counts are reported at several thresholds, and the test images whose
nearest train/val image is farther than --keep-above are written to a list, so the detector
can be re-scored on a duplicate-free subset with eval_detector_test.py.

A dataset with a known frame-level leak (the image-level FIR split) is a useful positive
control for the thresholds.

Usage:
  python audit_near_duplicates.py --train DIR [DIR ...] --test DIR --out audit.json \
      --keep-list test_clean.txt [--keep-above 10]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def long_path(p: Path) -> str:
    s = str(p.resolve())
    return "\\\\?\\" + s if os.name == "nt" and not s.startswith("\\\\?\\") else s


def phash(path: Path) -> int | None:
    data = np.fromfile(long_path(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)[:8, :8].flatten()
    bits = d > np.median(d[1:])
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hashes(dirs: list[Path]) -> tuple[list[str], np.ndarray]:
    names, hs = [], []
    for d in dirs:
        for f in sorted(os.listdir(long_path(d))):
            if Path(f).suffix.lower() in EXTS:
                h = phash(d / f)
                if h is not None:
                    names.append(f)
                    hs.append(h)
    return names, np.array(hs, dtype=np.uint64)


def popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Perceptual-hash near-duplicate audit")
    ap.add_argument("--train", type=Path, nargs="+", required=True, help="Training and validation image dirs")
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--test-list", type=Path, default=None, help="Optional list of test image paths to restrict to")
    ap.add_argument("--thresholds", type=int, nargs="+", default=[0, 4, 8, 10, 12, 16])
    ap.add_argument("--keep-above", type=int, default=10)
    ap.add_argument("--keep-list", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tr_names, tr = hashes(args.train)
    te_names, te = hashes([args.test])
    if args.test_list:
        allowed = {Path(line.strip()).name for line in args.test_list.read_text(encoding="utf-8").splitlines() if line.strip()}
        keep = [i for i, n in enumerate(te_names) if n in allowed]
        te_names, te = [te_names[i] for i in keep], te[keep]
    dmin = np.empty(len(te), dtype=np.int64)
    nearest = []
    for i, h in enumerate(te):
        d = popcount64(np.bitwise_xor(tr, h))
        j = int(d.argmin())
        dmin[i] = d[j]
        nearest.append(tr_names[j])
    out = {
        "n_train_val": len(tr), "n_test": len(te),
        "min_distance_quantiles": {q: float(np.percentile(dmin, q)) for q in (1, 5, 25, 50)},
        "n_test_within": {str(t): int((dmin <= t).sum()) for t in args.thresholds},
        "pct_test_within": {str(t): round(100 * float((dmin <= t).mean()), 1) for t in args.thresholds},
        "keep_above": args.keep_above,
        "n_kept": int((dmin > args.keep_above).sum()),
        "closest_pairs": sorted(({"test": te_names[i], "train": nearest[i], "d": int(dmin[i])} for i in range(len(te))),
                                key=lambda r: r["d"])[:25],
    }
    if args.keep_list:
        args.keep_list.write_text("\n".join(str((args.test / te_names[i]).resolve()) for i in range(len(te))
                                            if dmin[i] > args.keep_above) + "\n", encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_train_val", "n_test", "min_distance_quantiles", "n_test_within", "n_kept")}))


if __name__ == "__main__":
    main()
