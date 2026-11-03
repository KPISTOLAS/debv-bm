"""Tier 2: YOLOv11 detect → crop → optional BioCLIP → optional Grad-CAM++."""

from __future__ import annotations

import argparse
import json
import sys
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
    p.add_argument(
        "--clip-pretrained",
        default="openai-b16",
        help="Must match the paper: bioclip | openai-b16 | openai-b32 | hf-hub:imageomics/bioclip",
    )
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


def unwrap_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint is not a state_dict")
    if any(k.startswith("module.") for k in state):
        state = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}
    return state


def classify_crop(crop_bgr, model, preprocess, tokenizer, class_names, device, modality: str) -> tuple[str, float]:
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
    from clip_prompts import prompt_for_class

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    im = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
    text = tokenizer([prompt_for_class(n, modality) for n in class_names]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(im)
        text_features = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        conf, idx = logits[0].max(dim=0)
    return class_names[int(idx)], float(conf)


def reshape_transform_vit(tensor):
    """Map ViT tokens (B, 1+HW, C) to (B, C, H, W) for pytorch-grad-cam."""
    tokens = tensor[:, 1:, :]
    n = tokens.shape[1]
    side = int(n**0.5)
    if side * side != n:
        raise RuntimeError(f"ViT token count {n} is not a square grid (plus CLS)")
    result = tokens.reshape(tensor.size(0), side, side, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def vit_target_layer(model):
    visual = getattr(model, "visual", None)
    if visual is None:
        return None
    tr = getattr(visual, "transformer", None)
    blocks = getattr(tr, "resblocks", None) if tr is not None else None
    if blocks:
        ln = getattr(blocks[-1], "ln_1", None)
        if ln is not None:
            return ln
    trunk = getattr(visual, "trunk", None)
    tblocks = getattr(trunk, "blocks", None) if trunk is not None else None
    if tblocks:
        return getattr(tblocks[-1], "norm1", None) or getattr(tblocks[-1], "norm", None)
    return None


class _ClipImageLogits:
    """Forward that returns closed-set CLIP logits so Grad-CAM++ has a class score."""

    def __init__(self, clip_model, text_features):
        import torch.nn as nn

        class _M(nn.Module):
            def __init__(self, clip_model, text_features):
                super().__init__()
                self.clip = clip_model
                self.register_buffer("text_features", text_features)

            def forward(self, x):
                feat = self.clip.encode_image(x)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                tf = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                return 100.0 * feat @ tf.T

        self.module = _M(clip_model, text_features)


def maybe_gradcam(
    crop_bgr,
    model,
    preprocess,
    tokenizer,
    class_names: list[str],
    species: str,
    device: str,
    modality: str,
    out_path: Path,
) -> bool:
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import torch
        from PIL import Image
        import numpy as np
    except ImportError:
        print("Grad-CAM++ skipped: pip install grad-cam (tier2/requirements-tier2.txt)")
        return False

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
    from clip_prompts import prompt_for_class

    layer = vit_target_layer(model)
    reshape = reshape_transform_vit if layer is not None else None
    if layer is None:
        convs = [m for m in model.modules() if m.__class__.__name__ in {"Conv2d", "Conv"}]
        if not convs:
            print("Grad-CAM++ skipped: no ViT LayerNorm target and no Conv2d (BioCLIP is ViT-B/16)")
            return False
        layer = convs[-1]
        print("Grad-CAM++: no ViT block found; using last Conv2d (not the paper's classifier path)")

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    texts = tokenizer([prompt_for_class(n, modality) for n in class_names]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    wrapper = _ClipImageLogits(model, text_features).module.to(device)
    wrapper.eval()
    cam_layer = vit_target_layer(wrapper.clip) if reshape is not None else layer
    if cam_layer is None:
        cam_layer = layer
    try:
        idx = class_names.index(species)
    except ValueError:
        idx = 0
    vis_rgb = np.array(Image.fromarray(rgb).resize((tensor.shape[-1], tensor.shape[-2]))).astype("float32") / 255.0
    try:
        with GradCAMPlusPlus(
            model=wrapper,
            target_layers=[cam_layer],
            reshape_transform=reshape,
        ) as cam:
            grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(idx)])[0]
        vis = show_cam_on_image(vis_rgb, grayscale, use_rgb=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(vis).save(out_path)
        return True
    except Exception as e:
        print(f"Grad-CAM++ failed: {e}")
        return False


def load_bioclip(weights: Path, device: str, clip_pretrained: str):
    """Load CLIP/BioCLIP. clip_pretrained must match the paper name (no silent OpenAI fallback)."""
    import open_clip
    import torch

    if clip_pretrained.startswith("hf-hub:"):
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(clip_pretrained)
    else:
        arch = "ViT-B-32" if "b32" in clip_pretrained.replace("-", "").lower() else "ViT-B-16"
        tag = "openai"
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=tag, device=device
        )
        tokenizer = open_clip.get_tokenizer(arch)
    try:
        raw = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        raw = torch.load(weights, map_location=device)
    state = unwrap_state_dict(raw)
    incompatible = model.load_state_dict(state, strict=True)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise RuntimeError(
            f"Classifier weights {weights} do not match backbone {clip_pretrained}: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
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
    if args.bioclip_weights is not None:
        if not args.bioclip_weights.exists():
            raise SystemExit(f"Classifier weights not found: {args.bioclip_weights}")
        if not names:
            raise SystemExit("Classifier weights given but no class names YAML loaded")
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tag = args.clip_pretrained
        if tag == "bioclip":
            tag = "hf-hub:imageomics/bioclip"
        bioclip = load_bioclip(args.bioclip_weights, device, tag)

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
                modality = "thermal" if args.domain == "nocturnal" else "rgb"
                species, cconf = classify_crop(
                    crop, model, preprocess, tokenizer, names, device, modality
                )
                det["species"] = species
                det["classifier"] = args.clip_pretrained
                det["classifier_confidence"] = cconf
                if args.heatmaps:
                    hp = args.heatmaps / f"{path.stem}_{i}.png"
                    det["gradcam"] = (
                        str(hp)
                        if maybe_gradcam(
                            crop,
                            model,
                            preprocess,
                            tokenizer,
                            names,
                            species,
                            device,
                            modality,
                            hp,
                        )
                        else None
                    )
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
