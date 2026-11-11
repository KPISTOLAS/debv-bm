"""Paired comparison of two fine-tuned CLIP-family heads on the same image-folder test split.

Both heads label every crop; the script reports each head's accuracy and macro-F1 (mean of
per-class F1) and the paired differences A - B with 95% bootstrap intervals that resample
groups (photographs, forests, or source images; see eval_folder_classifier.py), plus a
group-level sign-flip randomization p-value for the accuracy difference.

Usage:
  python compare_classifiers.py --data DATA --config CFG.yaml \
      --a runs/bioclip.pt --a-pretrained hf-hub:imageomics/bioclip \
      --b runs/clip.pt --b-pretrained openai --group-by marche --out cmp.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_prompts import prompt_for_class
from eval_folder_classifier import GROUP_PATTERNS, confusion, conventional_macro_f1, group_ids
from train_bioclip import FolderCrops, load_open_clip


def predict(weights: Path, pretrained: str, names, modality, root: Path, device: str, model_name: str):
    model, preprocess, tokenizer = load_open_clip(model_name, pretrained, device)
    model.load_state_dict(torch.load(weights, map_location=device)["state_dict"])
    model.to(device).eval()
    ds = FolderCrops(root, {n: i for i, n in enumerate(names)}, preprocess)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    with torch.no_grad():
        text = F.normalize(model.encode_text(tokenizer([prompt_for_class(n, modality) for n in names]).to(device)), dim=-1)
        ys, ps = [], []
        for x, y in loader:
            ps.append((F.normalize(model.encode_image(x.to(device)), dim=-1) @ text.T).argmax(-1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps), [p for p, _ in ds.samples]


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired classifier comparison")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--a-pretrained", default="hf-hub:imageomics/bioclip")
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--b-pretrained", default="openai")
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--group-by", choices=("none", "pureforest", "marche"), default="none")
    ap.add_argument("--group-regex", default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    names = list(cfg["names"])
    k = len(names)
    modality = str(cfg.get("modality", "rgb"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = args.data / args.split
    y, pa, paths = predict(args.a, args.a_pretrained, names, modality, root, device, args.model)
    y2, pb, paths_b = predict(args.b, args.b_pretrained, names, modality, root, device, args.model)
    assert np.array_equal(y, y2) and paths == paths_b

    pattern = args.group_regex or GROUP_PATTERNS.get(args.group_by)
    groups = group_ids(paths, pattern)
    members = [np.flatnonzero(groups == g) for g in range(int(groups.max()) + 1)]

    def stats(idx):
        acc_a, acc_b = (pa[idx] == y[idx]).mean(), (pb[idx] == y[idx]).mean()
        f_a = conventional_macro_f1(confusion(y[idx], pa[idx], k))
        f_b = conventional_macro_f1(confusion(y[idx], pb[idx], k))
        return 100 * acc_a, 100 * acc_b, 100 * (acc_a - acc_b), 100 * f_a, 100 * f_b, 100 * (f_a - f_b)

    point = stats(np.arange(len(y)))
    rng = np.random.default_rng(0)
    boot = np.array([stats(np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))]))
                     for _ in range(args.bootstrap)])
    ci = lambda j: [round(float(np.percentile(boot[:, j], 2.5)), 2), round(float(np.percentile(boot[:, j], 97.5)), 2)]  # noqa: E731
    d = np.array([((pa[m] == y[m]).sum() - (pb[m] == y[m]).sum()) for m in members], dtype=float)
    nz = d[d != 0]
    p_sign = float((np.abs((rng.choice([-1.0, 1.0], size=(20000, len(nz))) * nz).sum(1)) >= abs(nz.sum()) - 1e-9).mean()) if len(nz) else 1.0
    keys = ["acc_a", "acc_b", "acc_diff", "macro_f1_a", "macro_f1_b", "macro_f1_diff"]
    out = {"a": str(args.a), "b": str(args.b), "n": int(len(y)), "n_groups": len(members),
           **{key: {"value": round(float(point[j]), 2), "ci95": ci(j)} for j, key in enumerate(keys)},
           "acc_diff_sign_flip_p": round(p_sign, 4)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
