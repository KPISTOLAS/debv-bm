"""Nocturnal row of the error-propagation table on the recording-level FIR split.

Same estimators and settings as error_propagation.py (pseudo-count 0.01, sparse assemblage
of 3 taxa with 50 detections, 40 sites x 3 visits, focal Fox with true psi = 0.10), but the
confusion matrix is that of the BioCLIP head trained and tested on the recording-level split
(build_fir_recording_split.py) and the per-visit detection probability is that split's
YOLOv8n test recall.

Usage:
  python error_propagation_firrec.py --cls firrec_bioclip_cls.json --p-det 0.5418 --out ep_firrec.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import error_propagation as ep  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Error propagation, FIR recording-level split")
    ap.add_argument("--cls", type=Path, required=True, help="eval_folder_classifier.py result on the split")
    ap.add_argument("--p-det", type=float, required=True, help="Detector test recall on the split")
    ap.add_argument("--smooth", type=float, default=0.01)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = ep.DOMAINS["Nocturnal animals"]
    res = json.loads(args.cls.read_text(encoding="utf-8"))
    names = res["names"]
    counts = np.array(res["confusion"], dtype=float) + args.smooth
    c = counts / counts.sum(axis=1, keepdims=True)
    focal = names.index(d["focal"])
    out = {
        "accuracy_pct": res["accuracy"], "error_pct": round(100 - res["accuracy"], 2), "p_det": args.p_det,
        "richness_sparse": ep.richness_pair(c, d["n_present"], d["n_det"]),
        "occupancy": ep.occupancy_pair(c, args.p_det, np.array(d["psi"]), focal),
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
