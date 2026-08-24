# Paper checkpoints (v1.0.2)

These are the detector `.pt` files the local eval scripts actually loaded. They live in `weights/paper/`. BioCLIP 50-epoch fine-tunes and the plant `exp-2.pt` detector were not on this machine and are **not** in this tag.

| File | Used for | Train start (`args.yaml`) |
|---|---|---|
| `diurnal_wildlife_yolo_aerial_optimized4.pt` | WAID cascade / diurnal detect | `yolov8l.pt` |
| `nocturnal_yolo11m.pt` | FIR cascade / nocturnal detect | `yolo11m.pt` |
| `diurnal_wildlife_yolo_balanced.pt` | AD/IC Grad-CAM++ | `yolov8m.pt` |

The paper text reports YOLOv11l. Archive what was run, not a renamed architecture.

```bash
python train/evaluate_faithfulness.py \
  --weights weights/paper/diurnal_wildlife_yolo_balanced.pt \
  --images path/to/WAID/images/test --labels path/to/WAID/labels/test
```

Plant detector: recover `exp-2.pt` (YOLOv11n Tree) from the training PC if you still have it. Classifier Table 2 still needs a BioCLIP overlay `.pt` or `hf-hub:imageomics/bioclip`.
