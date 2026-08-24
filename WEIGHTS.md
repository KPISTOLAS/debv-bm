# Checkpoints (not in git)

Fine-tuned YOLO and BioCLIP weights are excluded from this repository (see `.gitignore`). Table 2, the cascade table, and AD/IC numbers need those files plus the public Roboflow splits.

## What to archive on Zenodo

1. Push git tag `v1.0.0` and create a GitHub Release from it (Zenodo–GitHub can mint a DOI for that tag).
2. Upload the **same** tagged source plus the `.pt` files used for the paper tables as a second Zenodo version, or as files on that release:
   - YOLOv11-l plants / diurnal / nocturnal fine-tunes
   - BioCLIP (or OpenAI CLIP) 50-epoch unfrozen fine-tunes for each domain
3. Paste the DOI into `CITATION.cff` (`identifiers:`) and the paper Data Availability sentence next to `https://github.com/KPISTOLAS/debv-bm`.

Until that DOI exists, weights remain “on reasonable request” and a clone cannot turnkey-regenerate the numeric tables. The scripts in `train/` are the recipes, not the numbers.
