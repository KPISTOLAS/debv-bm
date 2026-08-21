"""Tier 2: YOLOv11 detect → crop → optional BioCLIP → optional Grad-CAM++."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM Tier 2 inference")
    p.add_argument("--inbox", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--weights", type=Path, required=True, help="YOLOv11 .pt checkpoint")
    p.add_argument("--domain", choices=["plants", "diurnal", "nocturnal"], required=True)
    p.add_argument("--config", type=Path, default=None, help="YAML with names: [...]")
    p.add_argument("--bioclip-weights", type=Path, default=None, help="Optional OpenCLIP/BioCLIP .pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--heatmaps", type=Path, default=None)
    return p.parse_args()


def load_names(args: argparse.Namespace) -> list[str] | None:
    if args.config is None:
        root = Path(__file__).resolve().parents[1] / "configs"
        fallback = {
            "diurnal": root / "diurnal.yaml",
            "nocturnal": root / "nocturnal.yaml",
            "plants": root / "plants.example.yaml",
        }[args.domain]
        path = fallback
    else:
        path = args.config
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(data.get("names", []))


def sidecar_meta(image_path: Path) -> dict:
    js = image_path.with_suffix(".json")
    if js.exists():
        try:
            return json.loads(js.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def classify_crop(crop_bgr, model, preprocess, tokenizer, class_names, device) -> tuple[str, float]:
    import torch
    from PIL import Image

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    im = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
    text = tokenizer([f"a photo of a {n}" for n in class_names]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(im)
        text_features = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        conf, idx = logits[0].max(dim=0)
    return class_names[int(idx)], float(conf)


def maybe_gradcam(crop_bgr, model, out_path: Path) -> bool:
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import torch
        from PIL import Image
        import numpy as np
    except ImportError:
        return False
    # Best-effort: last conv-like module; skip silently if the graph is not CNN-shaped.
    target_layers = []
    for m in model.modules():
        if m.__class__.__name__ in {"Conv2d", "Conv"}:
            target_layers.append(m)
    if not target_layers:
        return False
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    try:
        with GradCAMPlusPlus(model=model, target_layers=[target_layers[-1]]) as cam:
            grayscale = cam(input_tensor=tensor)[0]
        vis = show_cam_on_image(rgb, grayscale, use_rgb=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(vis).save(out_path)
        return True
    except Exception:
        return False


def load_bioclip(weights: Path, device: str):
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        pass
    model.eval()
    return model, preprocess, tokenizer


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    names = load_names(args)
    args.output.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(str(args.weights))
    device = "cpu"
    bioclip = None
    if args.bioclip_weights and args.bioclip_weights.exists() and names:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            bioclip = load_bioclip(args.bioclip_weights, device)
        except Exception as e:
            print(f"BioCLIP not loaded ({e}); writing detector boxes only")

    images = sorted(p for p in args.inbox.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images in {args.inbox}")
        return

    for path in images:
        meta = sidecar_meta(path)
        result = yolo.predict(source=str(path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        im = result.orig_img
        detections = []
        for i, box in enumerate(result.boxes):
            xyxy = [float(x) for x in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            crop = im[y1:y2, x1:x2]
            det = {
                "xyxy": xyxy,
                "detector_class": int(box.cls[0]) if box.cls is not None else None,
                "confidence": float(box.conf[0]) if box.conf is not None else None,
            }
            if bioclip is None or crop.size == 0 or not names:
                det["species"] = None
                det["classifier"] = "classifier_unavailable"
            else:
                model, preprocess, tokenizer = bioclip
                species, cconf = classify_crop(crop, model, preprocess, tokenizer, names, device)
                det["species"] = species
                det["classifier"] = "bioclip"
                det["classifier_confidence"] = cconf
                if args.heatmaps:
                    hp = args.heatmaps / f"{path.stem}_{i}.png"
                    det["gradcam"] = str(hp) if maybe_gradcam(crop, model, hp) else None
            detections.append(det)

        payload = {
            "image": str(path.name),
            "domain": args.domain,
            "timestamp_utc": meta.get("timestamp_utc") or datetime.now(timezone.utc).isoformat(),
            "edge_meta": meta,
            "detections": detections,
        }
        out = args.output / f"{path.stem}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"WROTE {out} n={len(detections)}")


if __name__ == "__main__":
    main()
