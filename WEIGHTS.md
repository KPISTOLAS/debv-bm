# Checkpoints

The trained weights of the paper are archived on Zenodo (CC BY 4.0), concept DOI
[10.5281/zenodo.23920497](https://doi.org/10.5281/zenodo.23920497) (resolves to the latest
version). Checkpoints are not stored in git.

| Weights version | DOI | Contents |
|---|---|---|
| 8 Nov 2026 | [10.5281/zenodo.24058509](https://doi.org/10.5281/zenodo.24058509) | 16 YOLO detectors (YOLOv8n/l and YOLOv11n/l for plants, the earlier WAID derivative, the official WAID split, and FIR) and 5 fine-tuned BioCLIP heads (PureForest, Marche, WAID derivative, official WAID, FIR), with a model card and `SHA256SUMS.txt` |
| next version (with release v1.4.0) | see the concept DOI | adds the models of the three-seed and leave-one-recording-out experiments listed below |

Models added for release v1.4.0 (paper Tables 9 and the BioCLIP-versus-CLIP comparison):

- FIR recording-level split: YOLOv8n detectors, seeds 0, 1, 2 (`firrec_yolov8n*.pt`)
- FIR recording-level split: BioCLIP heads and OpenAI-CLIP heads, seeds 0, 1, 2
- Marche: BioCLIP heads, seeds 1 and 2 (seed 0 is `bioclip_marche.pt`), and OpenAI-CLIP heads, seeds 0, 1, 2
- FIR leave-one-recording-out: five YOLOv8n detectors (`loro1`--`loro5`)

Verify downloads with `sha256sum -c SHA256SUMS.txt`.

[`weights/paper/`](weights/paper/) keeps only the three R2 detector checkpoints (tag **v1.0.2**);
see that README.
