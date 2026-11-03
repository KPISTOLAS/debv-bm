"""Propagate measured BioCLIP confusion matrices through richness and occupancy estimators.

Inputs are the eval_folder_classifier.py results (WAID, FIR, Marche test splits).
Each confusion matrix gets a pseudo-count of --smooth per cell before row-normalising;
0.01 is the smallest that keeps C invertible when a class has zero recall, and 0.1 is
the sensitivity run reported in the paper.

Usage:
  python error_propagation.py --waid waid.json --fir fir.json --marche marche.json --out ep.json [--smooth 0.01]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(17)
N_SIMS = 2000
N_OCC = 600
K_SITES = 40
J_VISITS = 3

# p_det = Table 1 YOLOv11l recall on each test split; focal = rare taxon with true psi = 0.10.
DOMAINS = {
    "Diurnal animals": {
        "arg": "waid",
        "p_det": 0.9187,
        "focal": "deer",
        "psi": [0.80, 0.55, 0.35, 0.40, 0.30, 0.40, 0.15, 0.10, 0.15],
        "n_present": 3,
        "n_det": 50,
    },
    "Nocturnal animals": {
        "arg": "fir",
        "p_det": 0.8234,
        "focal": "Fox",
        "psi": [0.35, 0.55, 0.40, 0.30, 0.20, 0.10, 0.40],
        "n_present": 3,
        "n_det": 50,
    },
    "Plants (Marche)": {
        "arg": "marche",
        "p_det": 0.8290,
        "focal": "quercus_ilex",
        "psi": None,  # 0.10 for the focal oak, 0.50 elsewhere
        "n_present": 5,
        "n_det": 80,
    },
}


def _classify(c: np.ndarray, true: np.ndarray) -> np.ndarray:
    u = RNG.random(true.shape)
    cdf = np.cumsum(c, axis=1)
    return (u[..., None] > cdf[true]).sum(axis=-1)


def invert_counts(n_pred: np.ndarray, c: np.ndarray) -> np.ndarray:
    n_hat = np.linalg.solve(c.T, n_pred.astype(float))
    return np.clip(n_hat, 0.0, None)


def richness_pair(c: np.ndarray, n_present: int, n_det: int) -> dict:
    n = c.shape[0]
    naive = np.empty(N_SIMS)
    corr = np.empty(N_SIMS)
    for s in range(N_SIMS):
        present = RNG.choice(n, size=n_present, replace=False)
        w = np.full(n_present, 0.3 / max(n_present - 1, 1))
        w[0] = 0.7 if n_present > 1 else 1.0
        w /= w.sum()
        true = RNG.choice(present, size=n_det, p=w)
        pred = _classify(c, true)
        naive[s] = np.unique(pred).size - n_present
        n_pred = np.bincount(pred, minlength=n).astype(float)
        n_hat = invert_counts(n_pred, c)
        corr[s] = np.sum(n_hat >= 0.5) - n_present
    return {
        "S_true": n_present,
        "N": n_det,
        "naive_mean": round(float(naive.mean()), 2),
        "naive_sd": round(float(naive.std(ddof=1)), 2),
        "corr_mean": round(float(corr.mean()), 2),
        "corr_sd": round(float(corr.std(ddof=1)), 2),
    }


def occupancy_pair(c: np.ndarray, p_det: float, psi: np.ndarray, focal: int) -> dict:
    z_all = []
    pred_all = []
    for _ in range(N_OCC):
        z = RNG.random((K_SITES, c.shape[0])) < psi
        det = (RNG.random((K_SITES, J_VISITS, c.shape[0])) < p_det) & z[:, None, :]
        idx = np.argwhere(det)
        pred = np.zeros(K_SITES, dtype=bool)
        if idx.size:
            yhat = _classify(c, idx[:, 2])
            pred[idx[yhat == focal, 0]] = True
        z_all.append(z[:, focal])
        pred_all.append(pred)
    z_all = np.concatenate(z_all)
    pred_all = np.concatenate(pred_all)
    naive = float(pred_all.mean())
    p10 = float(pred_all[~z_all].mean()) if np.any(~z_all) else 0.0
    p11 = float(pred_all[z_all].mean()) if np.any(z_all) else naive
    denom = p11 - p10
    corr = float(np.clip((naive - p10) / denom, 0.0, 1.0)) if abs(denom) > 1e-8 else naive
    return {
        "psi_true": round(float(psi[focal]), 2),
        "naive": round(naive, 3),
        "corr": round(corr, 3),
        "p10": round(p10, 3),
        "p11": round(p11, 3),
        "bias_pp": round(100 * (naive - float(psi[focal])), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="DEBV-BM error propagation from measured confusion matrices")
    ap.add_argument("--waid", type=Path, required=True)
    ap.add_argument("--fir", type=Path, required=True)
    ap.add_argument("--marche", type=Path, required=True)
    ap.add_argument("--smooth", type=float, default=0.01)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = {}
    for name, d in DOMAINS.items():
        res = json.loads(getattr(args, d["arg"]).read_text(encoding="utf-8"))
        names = res["names"]
        counts = np.array(res["confusion"], dtype=float) + args.smooth
        c = counts / counts.sum(axis=1, keepdims=True)
        focal = names.index(d["focal"])
        if d["psi"] is None:
            psi = np.full(len(names), 0.50)
            psi[focal] = 0.10
        else:
            psi = np.array(d["psi"])
        rows[name] = {
            "accuracy_pct": res["accuracy"],
            "error_pct": round(100 - res["accuracy"], 2),
            "focal": d["focal"],
            "p_det": d["p_det"],
            "richness_sparse": richness_pair(c, d["n_present"], d["n_det"]),
            "occupancy": occupancy_pair(c, d["p_det"], psi, focal),
        }

    payload = {
        "smoothing": f"+{args.smooth} per cell, row-normalised",
        "n_sims_richness": N_SIMS,
        "n_sims_occupancy": N_OCC,
        "sites": K_SITES,
        "visits": J_VISITS,
        "domains": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, r in rows.items():
        rich, occ = r["richness_sparse"], r["occupancy"]
        print(f"{name:18s} err {r['error_pct']:5.2f}  dS {rich['naive_mean']:+.2f} -> {rich['corr_mean']:+.2f}  psi {occ['naive']:.3f} -> {occ['corr']:.3f}")


if __name__ == "__main__":
    main()
