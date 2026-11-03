"""Classifier forward-pass latency on the processing-node GPU (classifier latency figure).

ResNet-50, EfficientNet-B0, and ViT-B/16 (torchvision, random weights: timing does not
depend on weight values) and the BioCLIP ViT-B/16 image encoder with closed-set text
matching. Input 224x224, FP32, CUDA synchronised, warm-up discarded, median per image.

Usage:
  python benchmark_classifiers.py --out latency.json --figure latency.png
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM classifier latency benchmark")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--figure", type=Path, default=None)
    p.add_argument("--classes", type=int, default=10, help="Closed-set size for BioCLIP text matching")
    return p.parse_args()


def time_model(fn, batch: int, iters: int, warmup: int = 20) -> float:
    x = torch.randn(batch, 3, 224, 224, device="cuda")
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return 1000 * statistics.median(times) / batch


def main() -> None:
    import open_clip

    args = parse_args()
    models = {
        "ResNet-50": torchvision.models.resnet50(weights=None),
        "EfficientNet-B0": torchvision.models.efficientnet_b0(weights=None),
        "ViT-B/16": torchvision.models.vit_b_16(weights=None),
    }
    fns = {}
    for name, m in models.items():
        m = m.cuda().eval()
        fns[name] = m
    clip, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip", device="cuda")
    clip.eval()
    text = F.normalize(torch.randn(args.classes, 512, device="cuda"), dim=-1)
    fns["BioCLIP (ViT-B/16)"] = lambda x: (F.normalize(clip.encode_image(x), dim=-1) @ text.T).argmax(-1)

    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "input": "224x224 FP32", "ms_per_image": {}}
    for name, fn in fns.items():
        out["ms_per_image"][name] = {
            "batch1": round(time_model(fn, 1, 200), 2),
            "batch32": round(time_model(fn, 32, 50), 2),
        }
        print(name, out["ms_per_image"][name], flush=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(out["ms_per_image"])
        b1 = [out["ms_per_image"][n]["batch1"] for n in names]
        b32 = [out["ms_per_image"][n]["batch32"] for n in names]
        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
        xs = range(len(names))
        w = 0.38
        bars1 = ax.bar([x - w / 2 for x in xs], b1, w, label="Batch 1 (latency)", color="#4C72B0")
        bars2 = ax.bar([x + w / 2 for x in xs], b32, w, label="Batch 32 (per image)", color="#DD8452")
        for bars in (bars1, bars2):
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        ax.set_xticks(list(xs), names, fontsize=8)
        ax.set_ylabel("Inference time (ms image$^{-1}$)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(b1) * 1.18)
        fig.tight_layout()
        fig.savefig(args.figure)
        print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
