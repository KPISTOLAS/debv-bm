"""Closed-set BioCLIP / OpenCLIP fine-tune matching the paper recipe.

Image folder layout::

    data/
      train/<species>/*.jpg
      val/<species>/*.jpg   # optional; if missing, a 10% split is taken from train

Hyperparameters (paper): unfrozen backbone, 50 epochs, 224×224, batch 32,
AdamW lr=1e-4, weight decay 0.01, cosine annealing to 1e-6.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_prompts import prompt_for_class


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM BioCLIP fine-tune")
    p.add_argument("--data", type=Path, required=True, help="Root with train/<class>/ images")
    p.add_argument("--config", type=Path, required=True, help="YAML with names: [species, ...]")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--pretrained", default="openai", help="open_clip pretrained tag, or hf-hub:imageomics/bioclip")
    p.add_argument("--out", type=Path, default=Path("runs/bioclip/best.pt"))
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Convenience seed for new fine-tunes (default 0). Paper tables were not pinned to this value.",
    )
    return p.parse_args()


class FolderCrops(Dataset):
    def __init__(self, root: Path, class_to_idx: dict[str, int], preprocess):
        self.preprocess = preprocess
        self.samples: list[tuple[Path, int]] = []
        for name, idx in class_to_idx.items():
            folder = root / name
            if not folder.is_dir():
                continue
            for p in folder.rglob("*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    self.samples.append((p, idx))
        if not self.samples:
            raise SystemExit(f"No images under {root} for classes {list(class_to_idx)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, y = self.samples[i]
        im = Image.open(path).convert("RGB")
        return self.preprocess(im), y


def load_open_clip(model_name: str, pretrained: str, device: str):
    """Load the requested checkpoint. Do not fall back to OpenAI CLIP on failure."""
    import open_clip

    if pretrained.startswith("hf-hub:"):
        model, _, preprocess = open_clip.create_model_and_transforms(
            pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(pretrained)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    print(f"loaded backbone model_name={model_name!r} pretrained={pretrained!r}")
    for p in model.parameters():
        p.requires_grad = True
    return model, preprocess, tokenizer


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    names = list(cfg["names"])
    class_to_idx = {n: i for i, n in enumerate(names)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = load_open_clip(args.model, args.pretrained, device)
    model.to(device)

    train_root = args.data / "train" if (args.data / "train").is_dir() else args.data
    val_root = args.data / "val" if (args.data / "val").is_dir() else None
    train_ds = FolderCrops(train_root, class_to_idx, preprocess)
    if val_root:
        val_ds = FolderCrops(val_root, class_to_idx, preprocess)
    else:
        n_val = max(1, len(train_ds) // 10)
        n_train = len(train_ds) - n_val
        if n_train < 1:
            raise SystemExit("Need more images to split train/val")
        train_ds, val_ds = random_split(
            train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)
    from clip_prompts import prompt_for_class

    modality = str(cfg.get("modality", "rgb"))
    prompts = tokenizer([prompt_for_class(n, modality) for n in names]).to(device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for images, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device)
            y = y.to(device)
            image_features = model.encode_image(images)
            text_features = model.encode_text(prompts)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            logits = 100.0 * image_features @ text_features.T
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss) * images.size(0)
            n += images.size(0)
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, y in val_loader:
                images = images.to(device)
                y = y.to(device)
                image_features = F.normalize(model.encode_image(images), dim=-1)
                text_features = F.normalize(model.encode_text(prompts), dim=-1)
                pred = (100.0 * image_features @ text_features.T).argmax(dim=-1)
                correct += int((pred == y).sum())
                total += int(y.numel())
        acc = 100.0 * correct / max(total, 1)
        print(f"epoch {epoch} train_loss={running / max(n, 1):.4f} val_acc={acc:.2f}%")
        if acc >= best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "names": names, "val_acc": acc}, args.out)
    print(f"best val_acc={best_acc:.2f}% saved {args.out}")


if __name__ == "__main__":
    main()
