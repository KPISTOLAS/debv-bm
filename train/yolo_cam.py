"""YOLO last-conv Grad-CAM / Grad-CAM++ heatmaps on a GT crop.

Energy backward on captured activations (not a class logit through the YOLO head).
This is the detector CAM used for the paper AD/IC numbers, not the BioCLIP ViT path.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


def find_cam_conv(yolo) -> nn.Module:
    model_module = yolo.model
    excluded = ("detect", "dfl", "head", "cv2", "cv3")
    convs = []
    for name, module in model_module.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        low = name.lower()
        if any(k in low for k in excluded):
            continue
        layer_num = None
        for part in name.split("."):
            if part.isdigit():
                layer_num = int(part)
                break
        if layer_num is None or layer_num < 20:
            convs.append(module)
    if not convs:
        raise RuntimeError("No backbone Conv2d for detector Grad-CAM")
    return convs[-1]


class YoloCam:
    def __init__(self, yolo, method: str = "plusplus"):
        if method not in {"gradcam", "plusplus"}:
            raise ValueError(method)
        self.yolo = yolo
        self.method = method
        self.device = next(yolo.model.parameters()).device
        self.layer = find_cam_conv(yolo)
        self.activations = None
        self._capture = False
        self.layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inp, out):
        if self._capture:
            self.activations = out

    def heatmap_roi(self, image_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        roi = image_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            raise ValueError("empty bbox")
        h, w = roi.shape[:2]
        img = cv2.resize(roi, (640, 640)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.activations = None
        self._capture = True
        with torch.no_grad():
            _ = self.yolo.model(tensor)
        self._capture = False
        if self.activations is None:
            return np.ones((h, w), np.float32) * 0.5
        with torch.enable_grad():
            features = self.activations.detach().float().clone().requires_grad_(True)
            score = features.pow(2).sum()
            score.backward()
            if features.grad is None:
                return np.ones((h, w), np.float32) * 0.5
            gradients = features.grad.detach()[0]
            activations = features.detach()[0]
            if self.method == "gradcam":
                weights = gradients.mean(dim=(1, 2))
                cam = torch.clamp((weights[:, None, None] * activations).sum(dim=0), min=0.0)
            else:
                positive = torch.clamp(gradients, min=0.0)
                denom = positive.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
                alpha = (positive / denom).pow(2) * positive
                cam = torch.clamp((alpha * activations).sum(dim=0), min=0.0)
        hm = cam.cpu().numpy()
        lo, hi = float(hm.min()), float(hm.max())
        hm = (hm - lo) / (hi - lo + 1e-8)
        return cv2.resize(hm, (w, h), interpolation=cv2.INTER_CUBIC)
