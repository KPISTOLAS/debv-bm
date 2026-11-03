# R2 checkpoints (v1.0.2)

**The revision R3 paper weights are on Zenodo, not here:** DOI
[10.5281/zenodo.23920498](https://doi.org/10.5281/zenodo.23920498). That record holds the
12 detectors of Table 1 (YOLOv8n/l and YOLOv11n/l for plants, WAID and FIR), the 4
fine-tuned BioCLIP heads of Table 2 (PureForest, Marche, WAID, FIR), a model card with the
test-split score of every file, and `SHA256SUMS.txt`.

The three files below are the R2 checkpoints, kept for traceability of the R2 numbers only:

| File | Used in R2 for | Train start (`args.yaml`) |
|---|---|---|
| `diurnal_wildlife_yolo_aerial_optimized4.pt` | WAID cascade / diurnal detect | `yolov8l.pt` |
| `nocturnal_yolo11m.pt` | FIR cascade / nocturnal detect | `yolo11m.pt` |
| `diurnal_wildlife_yolo_balanced.pt` | AD/IC Grad-CAM++ (R3 reruns give AD 22.58% with the released script) | `yolov8m.pt` |

To reproduce R3 results, download the Zenodo record and run the scripts in `train/`
(see `train/README.md`, R3 pipeline section), e.g.

```bash
python train/evaluate_faithfulness.py \
  --weights detector_diurnal_yolo11l.pt \
  --images path/to/WAID/images/test --labels path/to/WAID/labels/test
```
