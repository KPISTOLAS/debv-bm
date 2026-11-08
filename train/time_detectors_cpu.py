"""CPU forward-pass time per image (Table 1 protocol: batch 16, 640 px, processing-node CPU).

Runs Ultralytics val on the CPU and reports speed['inference'] (forward pass only, per image,
excluding pre- and post-processing). The old platform weights are timed in the same run so
the new and old rows share one protocol.
"""
import json
from pathlib import Path
import os
import sys

from ultralytics import YOLO

ROOT = os.environ.get("DEBV_ROOT", Path(__file__).resolve().parents[2].as_posix())  # folder holding debv-bm/, data/, AI_Plant/
W = ROOT + "/AI_Plant/zenodo_weights"
DATA = ROOT + "/data/waid_official/data_dedup.yaml"


def main():
    out = {}
    for name in ["yolov8n", "yolo11n", "yolov8l", "yolo11l"]:
        for tag, w in (("official", f"{W}/detector_diurnal_waidofficial_{name}.pt"), ("platform", f"{W}/detector_diurnal_{name}.pt")):
            try:
                m = YOLO(w).val(data=DATA, split="test", imgsz=640, batch=16, conf=0.001, iou=0.5, device="cpu",
                            plots=False, verbose=False, project=ROOT + "/debv-bm/train/runs/detect", name="cpu_tmp", exist_ok=True)
            except Exception as e:  # old 9-label weights may not validate on the 6-class set
                print(tag, name, "skip", e, flush=True)
                continue
            out[f"{tag}_{name}"] = round(float(m.speed["inference"]), 1)
            print(tag, name, out[f"{tag}_{name}"], flush=True)
    json.dump(out, open(ROOT + "/data/waid_official/cpu_times.json", "w"), indent=2)


if __name__ == "__main__":
    main()
