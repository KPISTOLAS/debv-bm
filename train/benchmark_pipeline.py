"""Throughput of the Tier 2 pipeline on the processing-node GPU (Section: Pipeline Performance).

Protocol: test images are decoded into memory first (disk I/O and JPEG decoding excluded).
Images are processed in batches of --batch: YOLO detection (letterbox, inference, NMS at
conf 0.25 / IoU 0.5), then BioCLIP on all detection crops in batches of --batch, then
optionally Grad-CAM++ on every crop for its predicted class (one backward pass per crop
batch, Tier 2 target layer). CUDA is synchronised around every timed stage; the first pass
is a warm-up and is discarded, and the median of --repeats passes is reported.

Usage:
  python benchmark_pipeline.py --images TEST/images --detector yolo11l.pt --config ../configs/nocturnal.yaml \
      --bioclip best.pt --out throughput.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tier2"))
from clip_prompts import prompt_for_class
from evaluate_classifier import list_images
from infer_pipeline import _ClipImageLogits, reshape_transform_vit, vit_target_layer
from train_bioclip import load_open_clip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM Tier 2 throughput benchmark")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--detector", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--bioclip", type=Path, required=True)
    p.add_argument("--pretrained", default="hf-hub:imageomics/bioclip")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    names = list(cfg["names"])
    device = "cuda"
    detector = YOLO(str(args.detector))
    clip, preprocess, tokenizer = load_open_clip("ViT-B-16", args.pretrained, device)
    clip.load_state_dict(torch.load(args.bioclip, map_location=device)["state_dict"])
    clip.eval()
    with torch.no_grad():
        text = F.normalize(clip.encode_text(tokenizer([prompt_for_class(n, str(cfg.get("modality", "rgb"))) for n in names]).to(device)), dim=-1)
    wrapper = _ClipImageLogits(clip, text).module.to(device).eval()
    cam = GradCAMPlusPlus(model=wrapper, target_layers=[vit_target_layer(wrapper.clip)], reshape_transform=reshape_transform_vit)

    images = [img for img in (cv2.imread(str(p)) for p in list_images(args.images)) if img is not None]

    def one_pass() -> dict[str, float]:
        t = {"detect": 0.0, "classify": 0.0, "gradcam": 0.0}
        n_crops = 0
        for i in range(0, len(images), args.batch):
            batch = images[i:i + args.batch]
            sync(); t0 = time.perf_counter()
            results = detector.predict(batch, conf=0.25, iou=0.5, imgsz=640, verbose=False)
            sync(); t["detect"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            crops = []
            for img, res in zip(batch, results):
                for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy().astype(int):
                    if x2 > x1 and y2 > y1:
                        crops.append(preprocess(Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))))
            preds = []
            with torch.no_grad():
                for j in range(0, len(crops), args.batch):
                    x = torch.stack(crops[j:j + args.batch]).to(device)
                    preds.append((F.normalize(clip.encode_image(x), dim=-1) @ text.T).argmax(-1))
            sync(); t["classify"] += time.perf_counter() - t0
            n_crops += len(crops)

            t0 = time.perf_counter()
            for j, p in zip(range(0, len(crops), args.batch), preds):
                x = torch.stack(crops[j:j + args.batch]).to(device)
                cam(input_tensor=x, targets=[ClassifierOutputTarget(int(c)) for c in p])
            sync(); t["gradcam"] += time.perf_counter() - t0
        t["crops"] = n_crops
        return t

    one_pass()  # warm-up
    passes = [one_pass() for _ in range(args.repeats)]
    n = len(images)
    med = {k: statistics.median(p[k] for p in passes) for k in ("detect", "classify", "gradcam")}
    det_cls = med["detect"] + med["classify"]
    full = det_cls + med["gradcam"]
    out = {
        "images": n, "crops_per_pass": passes[0]["crops"], "batch": args.batch, "repeats": args.repeats,
        "gpu": torch.cuda.get_device_name(0),
        "ms_per_image": {k: round(1000 * v / n, 2) for k, v in med.items()},
        "images_per_s": {
            "detect": round(n / med["detect"], 1),
            "detect_classify": round(n / det_cls, 1),
            "detect_classify_gradcam": round(n / full, 1),
        },
        "images_per_h": {
            "detect_classify": round(3600 * n / det_cls),
            "detect_classify_gradcam": round(3600 * n / full),
        },
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
