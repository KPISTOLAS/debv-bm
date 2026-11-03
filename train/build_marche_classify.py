"""Build the Marche UAV tree-crown classification set (Table 2, plants) from the Zenodo archive.

Source: Nepi et al. 2025, Zenodo 10.5281/zenodo.14988733 (CC BY 4.0), Image_Dataset.zip.
The split is grouped by source drone photo (immagineN), so crowns from one photo never
appear in two splits; a seeded search keeps every class in every split. The paper split
is configs/splits/marche_split_manifest.json (seed 0).

Usage:
  python build_marche_classify.py --archive Image_Dataset.zip --out data/marche
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

PATTERN = re.compile(r"Segmented_Labelled_Images/mascherargb_(immagine\d+)_oggetto_\d+classe-([a-z_]+)\.jpg$")
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
TRIES = 2000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marche UAV crowns -> image-folder classification set")
    p.add_argument("--archive", type=Path, required=True, help="Image_Dataset.zip from Zenodo")
    p.add_argument("--out", type=Path, required=True, help="Writes <out>/{train,val,test}/<class>/ and a manifest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zip", action="store_true", help="Also write <out>.zip (Ultralytics classify upload)")
    return p.parse_args()


def score(assign: dict[str, str], groups: dict[str, Counter], total: Counter) -> float:
    """Lower is better: deviation of each split's class counts from the target fractions."""
    per_split = {s: Counter() for s in FRACTIONS}
    for g, s in assign.items():
        per_split[s].update(groups[g])
    err = 0.0
    for s, frac in FRACTIONS.items():
        for c, n in total.items():
            got = per_split[s][c]
            err += abs(got - frac * n) / n
            if got == 0:
                err += 5.0  # every class should appear in every split
    return err


def main() -> None:
    args = parse_args()
    with zipfile.ZipFile(args.archive) as zf:
        items = [(m.group(1), m.group(2), name) for name in zf.namelist() if (m := PATTERN.search(name))]
        groups: dict[str, Counter] = defaultdict(Counter)
        for photo, cls, _ in items:
            groups[photo][cls] += 1
        total = Counter(cls for _, cls, _ in items)

        rng = random.Random(args.seed)
        photos = sorted(groups)
        best, best_err = None, float("inf")
        for _ in range(TRIES):
            rng.shuffle(photos)
            n = len(photos)
            cut1, cut2 = round(n * FRACTIONS["train"]), round(n * (FRACTIONS["train"] + FRACTIONS["val"]))
            assign = {p: ("train" if i < cut1 else "val" if i < cut2 else "test") for i, p in enumerate(photos)}
            err = score(assign, groups, total)
            if err < best_err:
                best, best_err = assign, err

        if args.out.exists():
            shutil.rmtree(args.out)
        counts = {s: Counter() for s in FRACTIONS}
        for photo, cls, name in items:
            split = best[photo]
            dest = args.out / split / cls / Path(name).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(name))
            counts[split][cls] += 1

    manifest = {
        "source": "Zenodo 10.5281/zenodo.14988733 (CC BY 4.0)",
        "split_unit": "source drone photo",
        "seed": args.seed,
        "photos": {s: sorted(p for p, v in best.items() if v == s) for s in FRACTIONS},
        "counts": {s: dict(sorted(c.items())) for s, c in counts.items()},
    }
    (args.out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.zip:
        zip_path = args.out.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for f in sorted(args.out.rglob("*.jpg")):
                zf.write(f, f.relative_to(args.out).as_posix())
        print(f"wrote {zip_path}")

    print(f"{'class':24s} " + " ".join(f"{s:>6s}" for s in FRACTIONS))
    for c in sorted(total, key=lambda k: -total[k]):
        print(f"{c:24s} " + " ".join(f"{counts[s][c]:6d}" for s in FRACTIONS))
    print("photos:", {s: len(manifest["photos"][s]) for s in FRACTIONS},
          "crowns:", {s: sum(counts[s].values()) for s in FRACTIONS})


if __name__ == "__main__":
    main()
