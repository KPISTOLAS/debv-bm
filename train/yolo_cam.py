"""Detector CAM heatmaps for a YOLO box (Grad-CAM / Grad-CAM++), not the BioCLIP ViT path.

The map is taken at the last backbone convolution, on the ground-truth box crop resized
to 640x640 (RGB, [0,1], which is what the Ultralytics model expects after its own
BGR->RGB preprocessing).

Targets
  class   (default) backpropagate the detector's score for `target_class`, i.e. the
          maximum class-`target_class` score over all anchors of the crop, through the
          Detect head to the CAM layer. This is the class-discriminative setting of
          Grad-CAM (Selvaraju et al., 2017) and Grad-CAM++ (Chattopadhay et al., 2018).
  energy  legacy: backpropagate sum(A**2) of the detached CAM-layer activations. The
          resulting map is class-agnostic (gradient = 2A) and shows backbone activation
          energy, not a class decision. Kept only to reproduce the v1.3.0 numbers.

Methods
  gradcam    weights = spatial mean of the gradients.
  gradcampp  standard Grad-CAM++ closed form,
             alpha = g^2 / (2 g^2 + sum_ab(A) * g^3),  w_k = sum_ij alpha * relu(g),
             as in Chattopadhay et al. (2018) and pytorch-grad-cam.
  legacy_pp  the custom weighting of v1.3.0, alpha = (relu(g)/sum relu(g))^2 * relu(g).
             This is NOT Grad-CAM++; it is kept only for reproducing earlier numbers.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

METHODS = {"gradcam", "gradcampp", "legacy_pp"}
TARGETS = {"class", "energy"}


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
        raise RuntimeError("No backbone Conv2d for detector CAM")
    return convs[-1]


def _detect_scores(out) -> torch.Tensor:
    """Return the (B, 4+nc, N) decoded prediction tensor of an Ultralytics Detect model."""
    while isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, dict):
        for key in ("one2many", "one2one"):
            if key in out:
                return _detect_scores(out[key])
        raise RuntimeError(f"Unexpected Detect output keys: {list(out)}")
    if not torch.is_tensor(out) or out.dim() != 3:
        raise RuntimeError(f"Unexpected Detect output: {type(out)}")
    return out


def cam_weights(gradients: torch.Tensor, activations: torch.Tensor, method: str) -> torch.Tensor:
    """Channel weights (C,) from gradients and activations of shape (C, H, W)."""
    if method == "gradcam":
        return gradients.mean(dim=(1, 2))
    if method == "gradcampp":
        g2 = gradients.pow(2)
        g3 = gradients.pow(3)
        sum_a = activations.sum(dim=(1, 2), keepdim=True)
        denom = 2.0 * g2 + sum_a * g3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = torch.where(gradients != 0, g2 / denom, torch.zeros_like(g2))
        return (alpha * torch.relu(gradients)).sum(dim=(1, 2))
    if method == "legacy_pp":
        positive = torch.clamp(gradients, min=0.0)
        denom = positive.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        alpha = (positive / denom).pow(2) * positive
        return alpha.sum(dim=(1, 2))
    raise ValueError(method)


class YoloCam:
    def __init__(self, yolo, method: str = "gradcampp", target: str = "class"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}")
        if target not in TARGETS:
            raise ValueError(f"target must be one of {sorted(TARGETS)}")
        self.yolo = yolo
        self.method = method
        self.target = target
        self.device = next(yolo.model.parameters()).device
        self.layer = find_cam_conv(yolo)
        self.activations = None
        self._capture = False
        self._handle = self.layer.register_forward_hook(self._hook)

    def close(self) -> None:
        self._handle.remove()

    def _hook(self, _module, _inp, out):
        if self._capture:
            if out.requires_grad:
                out.retain_grad()
            self.activations = out

    def _blank(self, h, w):
        return np.ones((h, w), np.float32) * 0.5

    def heatmap_roi(self, image_rgb: np.ndarray, bbox: tuple[int, int, int, int], target_class: int | None = None) -> np.ndarray:
        """Heatmap in [0,1] over the bbox crop of an RGB image (shape = crop h x w)."""
        x1, y1, x2, y2 = bbox
        roi = image_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            raise ValueError("empty bbox")
        h, w = roi.shape[:2]
        img = cv2.resize(roi, (640, 640)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.activations = None

        if self.target == "class":
            if target_class is None or target_class < 0:
                raise ValueError("target='class' needs a non-negative target_class")
            tensor.requires_grad_(True)
            self._capture = True
            with torch.enable_grad():
                out = self.yolo.model(tensor)
                self._capture = False
                acts = self.activations
                if acts is None:
                    return self._blank(h, w)
                preds = _detect_scores(out)
                cls_row = 4 + int(target_class)
                if cls_row >= preds.shape[1]:
                    raise ValueError(f"class {target_class} outside model with {preds.shape[1] - 4} classes")
                score = preds[0, cls_row, :].max()
                self.yolo.model.zero_grad(set_to_none=True)
                score.backward()
            if acts.grad is None:
                return self._blank(h, w)
            gradients = acts.grad.detach()[0].float()
            activations = acts.detach()[0].float()
        else:
            self._capture = True
            with torch.no_grad():
                _ = self.yolo.model(tensor)
            self._capture = False
            if self.activations is None:
                return self._blank(h, w)
            with torch.enable_grad():
                features = self.activations.detach().float().clone().requires_grad_(True)
                features.pow(2).sum().backward()
            gradients = features.grad.detach()[0]
            activations = features.detach()[0]

        weights = cam_weights(gradients, activations, self.method)
        cam = torch.clamp((weights[:, None, None] * activations).sum(dim=0), min=0.0)
        hm = cam.cpu().numpy()
        lo, hi = float(hm.min()), float(hm.max())
        hm = (hm - lo) / (hi - lo + 1e-8)
        return cv2.resize(hm, (w, h), interpolation=cv2.INTER_CUBIC)
