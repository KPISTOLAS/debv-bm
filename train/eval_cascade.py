"""Detector -> BioCLIP cascade on a test split (Table 6), with the deployed models.

For each test image the YOLO detector runs at the deployed operating point (conf 0.25,
NMS IoU 0.5). Detections are matched class-agnostically to ground-truth boxes (one-to-one,
IoU >= 0.5); the detector's own class is ignored, as in the Tier 2 pipeline. Each matched
instance is classified twice by the BioCLIP head: on its ground-truth crop (Reference) and
on the detector crop (Cascade). Crops are tight boxes passed through the same JPEG round trip
as build_gt_crops.py, so Reference over all ground-truth boxes reproduces Table 2.

Usage:
  python eval_cascade.py --images TEST/images --labels TEST/labels --detector yolo11l.pt \
      --config ../configs/diurnal.yaml --bioclip best.pt --out cascade.json
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


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    names = list(cfg["names"])
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

    detector = YOLO(str(args.detector))
    n_gt = n_det = 0
    gt_y, gt_crops, m_y, m_gt_crops, m_det_crops, m_det_cls = [], [], [], [], [], []
    for img_path in list_images(args.images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = parse_yolo_label(args.labels / f"{img_path.stem}.txt", w, h, len(names))
        n_gt += len(gts)
        for g in gts:
            gt_y.append(g["cls"])
            gt_crops.append(crop_rgb(img, g["bbox"]))
        res = detector.predict(img, conf=args.conf, iou=args.nms_iou, imgsz=640, verbose=False)[0]
        det = res.boxes.xyxy.cpu().numpy()
        det_cls = res.boxes.cls.cpu().numpy().astype(int)
        n_det += len(det)
        if not gts or not len(det):
            continue
        gt_boxes = np.array([g["bbox"] for g in gts], dtype=float)
        for gi, di in greedy_match(iou_matrix(gt_boxes, det), args.match_iou):
            x1, y1, x2, y2 = det[di]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            m_y.append(gts[gi]["cls"])
            m_gt_crops.append(crop_rgb(img, gts[gi]["bbox"]))
            m_det_crops.append(crop_rgb(img, (x1, y1, x2, y2)))
            m_det_cls.append(det_cls[di])

    gt_y, m_y = np.array(gt_y), np.array(m_y)
    det_ok = np.array(m_det_cls) == m_y  # detector's own class on the same matches
    det_own = int(det_ok.sum())
    all_ref = int((classify(gt_crops) == gt_y).sum())
    ref = int((classify(m_gt_crops) == m_y).sum())
    cas_ok = classify(m_det_crops) == m_y
    cas = int(cas_ok.sum())
    n_m = len(m_y)
    # Exact two-sided McNemar test, detector class vs BioCLIP on the same matched instances.
    only_det, only_clip = int((det_ok & ~cas_ok).sum()), int((~det_ok & cas_ok).sum())
    k, n_disc = min(only_det, only_clip), only_det + only_clip
    p_mcnemar = min(1.0, 2 * sum(comb(n_disc, i) for i in range(k + 1)) / 2**n_disc) if n_disc else 1.0
    out = {
        "detector": str(args.detector), "bioclip": str(args.bioclip), "conf": args.conf,
        "n_gt": n_gt, "n_detections": n_det, "n_matched": n_m,
        "loc_recall": round(100 * n_m / n_gt, 2), "loc_recall_ci95": wilson(n_m, n_gt),
        "reference_all_gt": round(100 * all_ref / n_gt, 2),
        "reference": round(100 * ref / n_m, 2), "reference_ci95": wilson(ref, n_m),
        "cascade": round(100 * cas / n_m, 2), "cascade_ci95": wilson(cas, n_m),
        "drop_pp": round(100 * (ref - cas) / n_m, 2),
        "detector_class_acc": round(100 * det_own / n_m, 2), "detector_class_acc_ci95": wilson(det_own, n_m),
        "only_detector_correct": only_det, "only_bioclip_correct": only_clip, "mcnemar_p": round(p_mcnemar, 4),
        "e2e": round(100 * cas / n_gt, 2), "e2e_ci95": wilson(cas, n_gt),
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
