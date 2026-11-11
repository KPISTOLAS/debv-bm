"""Detector -> BioCLIP cascade on a test split (Table 6), with the deployed models.

For each test image the YOLO detector runs at the deployed operating point (conf 0.25,
NMS IoU 0.5). Detections are matched class-agnostically to ground-truth boxes (one-to-one,
IoU >= 0.5); the detector's own class is ignored, as in the Tier 2 pipeline. Each matched
instance is classified twice by the BioCLIP head: on its ground-truth crop (Reference) and
on the detector crop (Cascade). Crops are tight boxes passed through the same JPEG round trip
as build_gt_crops.py, so Reference over all ground-truth boxes reproduces Table 2.

Additions (review revision):
  * --qc-json: the edge QC log of build_edge_preprocessed.py; images it rejected produce no
    predictions, but their ground-truth animals stay in every recall denominator.
  * Intervals and the detector-vs-BioCLIP comparison are clustered by image: percentile
    bootstrap over images, and an image-level sign-flip randomization test of the paired
    difference (instances within one image are not independent). The instance-level
    Wilson intervals and exact McNemar test are kept for comparison.
  * Final species-level precision, recall and F1 over all emitted predictions, for two
    operating modes: detector-only (the detector's class) and detector+BioCLIP (BioCLIP
    label on every detection crop). A prediction is a true positive only if it is matched
    (IoU >= --match-iou, one-to-one) to a ground-truth animal of the same species; unmatched
    detections, detections on empty images, and matched detections with the wrong species
    are false positives. Reported per class, macro-averaged (mean of per-class F1) and
    micro-averaged, with image-bootstrap intervals.

Usage:
  python eval_cascade.py --images TEST/images --labels TEST/labels --detector yolo11l.pt \
      --config ../configs/diurnal.yaml --bioclip best.pt --out cascade.json [--qc-json qc.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb, sqrt
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_prompts import prompt_for_class
from evaluate_classifier import list_images, parse_yolo_label
from train_bioclip import load_open_clip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM detector -> BioCLIP cascade")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--detector", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--bioclip", type=Path, required=True)
    p.add_argument("--pretrained", default="hf-hub:imageomics/bioclip")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--qc-json", type=Path, default=None, help="qc.json from build_edge_preprocessed.py")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(iou: np.ndarray, thr: float) -> list[tuple[int, int]]:
    pairs = [(iou[g, d], g, d) for g, d in zip(*np.nonzero(iou >= thr))]
    used_g, used_d, out = set(), set(), []
    for _, g, d in sorted(pairs, reverse=True):
        if g not in used_g and d not in used_d:
            used_g.add(g)
            used_d.add(d)
            out.append((int(g), int(d)))
    return out


def crop_rgb(img: np.ndarray, box) -> Image.Image:
    """Tight crop, JPEG round trip as in build_gt_crops.py (cv2.imwrite default quality)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    ok, buf = cv2.imencode(".jpg", img[y1:y2, x1:x2])
    return Image.fromarray(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def species_prf(rows: list[dict], mode: str, k_cls: int) -> dict:
    """Species-level P/R/F1 over all emitted predictions of `mode` ('detector' or 'bioclip')."""
    tp, fp, fn = np.zeros(k_cls), np.zeros(k_cls), np.zeros(k_cls)
    for r in rows:
        for c in r["gt"]:
            fn[c] += 1
        for d in r["dets"]:
            lab = d["det_cls"] if mode == "detector" else d["clip_cls"]
            if lab < 0:
                continue
            if d["gt"] >= 0 and r["gt"][d["gt"]] == lab:
                tp[lab] += 1
                fn[lab] -= 1
            else:
                fp[lab] += 1
    p = np.divide(tp, tp + fp, out=np.zeros(k_cls), where=(tp + fp) > 0)
    rc = np.divide(tp, tp + fn, out=np.zeros(k_cls), where=(tp + fn) > 0)
    f = np.divide(2 * p * rc, p + rc, out=np.zeros(k_cls), where=(p + rc) > 0)
    present = (tp + fn) > 0
    mp = 100 * tp.sum() / max(tp.sum() + fp.sum(), 1)
    mr = 100 * tp.sum() / max(tp.sum() + fn.sum(), 1)
    return {
        "macro_precision": 100 * float(p[present].mean()),
        "macro_recall": 100 * float(rc[present].mean()),
        "macro_f1": 100 * float(f[present].mean()),
        "micro_precision": float(mp), "micro_recall": float(mr),
        "micro_f1": float(2 * mp * mr / (mp + mr)) if mp + mr else 0.0,
        "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()),
        "_p": p, "_r": rc, "_f": f, "_support": tp + fn,
    }


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    names = list(cfg["names"])
    k_cls = len(names)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip, preprocess, tokenizer = load_open_clip("ViT-B-16", args.pretrained, device)
    clip.load_state_dict(torch.load(args.bioclip, map_location=device)["state_dict"])
    clip.eval()
    with torch.no_grad():
        text = F.normalize(clip.encode_text(tokenizer([prompt_for_class(n, str(cfg.get("modality", "rgb"))) for n in names]).to(device)), dim=-1)

    def classify(crops: list[Image.Image]) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for i in range(0, len(crops), 32):
                x = torch.stack([preprocess(c) for c in crops[i:i + 32]]).to(device)
                preds.append((F.normalize(clip.encode_image(x), dim=-1) @ text.T).argmax(-1).cpu().numpy())
        return np.concatenate(preds) if preds else np.zeros(0, dtype=int)

    rejected: set[str] = set()
    if args.qc_json:
        qc = json.loads(args.qc_json.read_text(encoding="utf-8"))
        rejected = {k for k, v in qc["images"].items() if not v["passed"]}

    detector = YOLO(str(args.detector))
    n_det = 0
    images: list[dict] = []
    gt_y, gt_crops, det_crops, m_gt_crops = [], [], [], []
    for img_path in list_images(args.images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = parse_yolo_label(args.labels / f"{img_path.stem}.txt", w, h, len(names))
        rec = {"name": img_path.stem, "gt": [g["cls"] for g in gts], "gated": img_path.stem in rejected,
               "dets": [], "matches": []}
        for g in gts:
            gt_y.append(g["cls"])
            gt_crops.append(crop_rgb(img, g["bbox"]))
        if not rec["gated"]:
            res = detector.predict(img, conf=args.conf, iou=args.nms_iou, imgsz=640, verbose=False)[0]
            det = res.boxes.xyxy.cpu().numpy()
            det_cls = res.boxes.cls.cpu().numpy().astype(int)
            n_det += len(det)
            for di in range(len(det)):
                x1, y1, x2, y2 = det[di]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
                ok = x2 > x1 and y2 > y1
                rec["dets"].append({"det_cls": int(det_cls[di]), "crop_idx": len(det_crops) if ok else -1, "gt": -1})
                if ok:
                    det_crops.append(crop_rgb(img, (x1, y1, x2, y2)))
            if gts and len(det):
                gt_boxes = np.array([g["bbox"] for g in gts], dtype=float)
                for gi, di in greedy_match(iou_matrix(gt_boxes, det), args.match_iou):
                    d = rec["dets"][di]
                    if d["crop_idx"] < 0:
                        continue
                    d["gt"] = gi
                    rec["matches"].append({"gt_cls": gts[gi]["cls"], "ref_idx": len(m_gt_crops), "det": di})
                    m_gt_crops.append(crop_rgb(img, gts[gi]["bbox"]))
        images.append(rec)

    gt_y = np.array(gt_y)
    all_ref = int((classify(gt_crops) == gt_y).sum())
    det_lab = classify(det_crops)
    m_ref = classify(m_gt_crops)
    for r in images:
        for d in r["dets"]:
            d["clip_cls"] = int(det_lab[d["crop_idx"]]) if d["crop_idx"] >= 0 else -1
        for m in r["matches"]:
            d = r["dets"][m["det"]]
            m["ref_ok"] = int(m_ref[m["ref_idx"]] == m["gt_cls"])
            m["cas_ok"] = int(d["clip_cls"] == m["gt_cls"])
            m["det_ok"] = int(d["det_cls"] == m["gt_cls"])
        r["n_gt"] = len(r["gt"])

    # ---- instance-level summary (as in v1.3.0) ----
    ms = [m for r in images for m in r["matches"]]
    n_gt, n_m = int(sum(r["n_gt"] for r in images)), len(ms)
    ref = sum(m["ref_ok"] for m in ms)
    cas = sum(m["cas_ok"] for m in ms)
    det_own = sum(m["det_ok"] for m in ms)
    only_det = sum(m["det_ok"] and not m["cas_ok"] for m in ms)
    only_clip = sum(m["cas_ok"] and not m["det_ok"] for m in ms)
    k, n_disc = min(only_det, only_clip), only_det + only_clip
    p_mcnemar = min(1.0, 2 * sum(comb(n_disc, j) for j in range(k + 1)) / 2**n_disc) if n_disc else 1.0

    # ---- image-clustered statistics ----
    def ratio(rows, num, den):
        a = sum(num(r) for r in rows)
        b = sum(den(r) for r in rows)
        return 100.0 * a / b if b else float("nan")

    stats = {
        "loc_recall": (lambda r: len(r["matches"]), lambda r: r["n_gt"]),
        "reference": (lambda r: sum(m["ref_ok"] for m in r["matches"]), lambda r: len(r["matches"])),
        "cascade": (lambda r: sum(m["cas_ok"] for m in r["matches"]), lambda r: len(r["matches"])),
        "detector_class_acc": (lambda r: sum(m["det_ok"] for m in r["matches"]), lambda r: len(r["matches"])),
        "e2e_bioclip": (lambda r: sum(m["cas_ok"] for m in r["matches"]), lambda r: r["n_gt"]),
        "e2e_detector_only": (lambda r: sum(m["det_ok"] for m in r["matches"]), lambda r: r["n_gt"]),
    }

    def det_minus_clip(rows):
        n = sum(len(r["matches"]) for r in rows)
        return 100.0 * sum(m["det_ok"] - m["cas_ok"] for r in rows for m in r["matches"]) / n if n else float("nan")

    rng = np.random.default_rng(args.seed)
    n_img = len(images)
    boots = [[images[j] for j in rng.integers(0, n_img, n_img)] for _ in range(args.bootstrap)]

    def ci(fn):
        v = np.array([fn(b) for b in boots], dtype=float)
        v = v[np.isfinite(v)]
        return [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)] if len(v) else None

    clustered = {}
    for key, (num, den) in stats.items():
        clustered[key] = {"value": round(ratio(images, num, den), 2),
                          "ci95_image_bootstrap": ci(lambda b, num=num, den=den: ratio(b, num, den))}
    d_img = np.array([sum(m["det_ok"] - m["cas_ok"] for m in r["matches"]) for r in images], dtype=float)
    nz = d_img[d_img != 0]
    if len(nz):
        flips = rng.choice([-1.0, 1.0], size=(20000, len(nz)))
        p_sign = float((np.abs((flips * nz).sum(axis=1)) >= abs(nz.sum()) - 1e-9).mean())
    else:
        p_sign = 1.0
    clustered["detector_minus_bioclip_pp"] = {"value": round(det_minus_clip(images), 2),
                                              "ci95_image_bootstrap": ci(det_minus_clip),
                                              "sign_flip_p_image_level": round(p_sign, 4),
                                              "n_images_discordant": int(len(nz))}

    modes = {}
    for mode in ("detector", "bioclip"):
        pt = species_prf(images, mode, k_cls)
        modes[mode] = {key: (round(v, 2) if isinstance(v, float) else v) for key, v in pt.items() if not key.startswith("_")}
        modes[mode]["per_class"] = {names[c]: {"precision": round(100 * float(pt["_p"][c]), 2),
                                               "recall": round(100 * float(pt["_r"][c]), 2),
                                               "f1": round(100 * float(pt["_f"][c]), 2),
                                               "support": int(pt["_support"][c])} for c in range(k_cls)}
        for key in ("macro_precision", "macro_recall", "macro_f1", "micro_f1"):
            modes[mode][f"{key}_ci95"] = ci(lambda b, key=key, mode=mode: species_prf(b, mode, k_cls)[key])
    diff = lambda b: species_prf(b, "detector", k_cls)["macro_f1"] - species_prf(b, "bioclip", k_cls)["macro_f1"]  # noqa: E731
    modes["detector_minus_bioclip_macro_f1"] = {"value": round(diff(images), 2), "ci95": ci(diff)}

    out = {
        "detector": str(args.detector), "bioclip": str(args.bioclip), "conf": args.conf,
        "qc_gate": str(args.qc_json) if args.qc_json else None,
        "n_images": n_img, "n_images_gated": int(sum(r["gated"] for r in images)),
        "n_gt_in_gated_images": int(sum(r["n_gt"] for r in images if r["gated"])),
        "n_empty_images": int(sum(r["n_gt"] == 0 for r in images)),
        "n_gt": n_gt, "n_detections": n_det, "n_matched": n_m,
        "loc_recall": round(100 * n_m / n_gt, 2), "loc_recall_ci95": wilson(n_m, n_gt),
        "reference_all_gt": round(100 * all_ref / n_gt, 2),
        "reference": round(100 * ref / n_m, 2), "reference_ci95": wilson(ref, n_m),
        "cascade": round(100 * cas / n_m, 2), "cascade_ci95": wilson(cas, n_m),
        "drop_pp": round(100 * (ref - cas) / n_m, 2),
        "detector_class_acc": round(100 * det_own / n_m, 2), "detector_class_acc_ci95": wilson(det_own, n_m),
        "only_detector_correct": only_det, "only_bioclip_correct": only_clip, "mcnemar_p": round(p_mcnemar, 4),
        "e2e": round(100 * cas / n_gt, 2), "e2e_ci95": wilson(cas, n_gt),
        "image_clustered": clustered,
        "species_level_all_predictions": modes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({key: out[key] for key in ("n_images", "n_images_gated", "n_gt_in_gated_images", "n_gt", "n_detections",
                                                "n_matched", "loc_recall", "cascade", "detector_class_acc", "e2e", "mcnemar_p")}))
    print(json.dumps(clustered))
    print(json.dumps({m: {key: modes[m][key] for key in ("macro_precision", "macro_recall", "macro_f1", "macro_f1_ci95",
                                                         "micro_f1", "tp", "fp", "fn")} for m in ("detector", "bioclip")}))
    print(json.dumps(modes["detector_minus_bioclip_macro_f1"]))


if __name__ == "__main__":
    main()
