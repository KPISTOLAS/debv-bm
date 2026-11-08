"""Diurnal row of the error-propagation analysis for the official-WAID BioCLIP head (6 classes).

Reuses error_propagation.py unchanged except for the diurnal domain definition: the six
official classes, p_det = recall of the deployed YOLOv11l on the de-duplicated official test
split (Table 1), and kiang (rarest of the low-recall classes) as the focal taxon with psi = 0.10.
Only this domain is run, so the RNG stream starts at the same state as the diurnal row of the
original run and the nocturnal and plant rows of error_propagation.py stay valid.

Usage:
  python error_propagation_waid_official.py --waid bioclip_test_dedup.json --out ep.json [--smooth 0.01]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import error_propagation as ep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waid", type=Path, required=True)
    ap.add_argument("--smooth", type=float, default=0.01)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = {"p_det": 0.9579, "focal": "kiang", "psi": [0.80, 0.55, 0.35, 0.40, 0.10, 0.40], "n_present": 3, "n_det": 50}
    res = json.loads(args.waid.read_text(encoding="utf-8"))
    names = res["names"]
    counts = np.array(res["confusion"], dtype=float) + args.smooth
    c = counts / counts.sum(axis=1, keepdims=True)
    focal = names.index(d["focal"])
    row = {"accuracy_pct": res["accuracy"], "error_pct": round(100 - res["accuracy"], 2), "focal": d["focal"],
           "p_det": d["p_det"], "psi": d["psi"],
           "richness_sparse": ep.richness_pair(c, d["n_present"], d["n_det"]),
           "occupancy": ep.occupancy_pair(c, d["p_det"], np.array(d["psi"]), focal)}
    args.out.write_text(json.dumps({"smoothing": f"+{args.smooth} per cell, row-normalised",
                                    "domains": {"Diurnal animals (official WAID)": row}}, indent=2), encoding="utf-8")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
