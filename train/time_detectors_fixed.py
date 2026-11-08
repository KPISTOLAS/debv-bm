"""Data-independent CPU forward-pass time for every Table 1 detector (one protocol).

Each fused model runs a fixed 16x3x640x640 tensor on the CPU (torch threads = all cores),
5 warm-up passes, then 20 timed passes; reported value = median pass time / 16 (ms per image).
Pre- and post-processing (letterbox, NMS) are excluded, as in Table 1.
"""
import json
from pathlib import Path
import os
import statistics
import time

import torch
from ultralytics import YOLO

ROOT = os.environ.get("DEBV_ROOT", Path(__file__).resolve().parents[2].as_posix())  # folder holding debv-bm/, data/, AI_Plant/
W = ROOT + "/AI_Plant/zenodo_weights"
NAMES = [f"detector_{d}_{m}" for d in ("plants", "diurnal_waidofficial", "nocturnal", "diurnal")
         for m in ("yolov8n", "yolo11n", "yolov8l", "yolo11l")]


def main():
    torch.manual_seed(0)
    x = torch.rand(16, 3, 640, 640)
    out = {"protocol": "CPU, fused, fp32, batch 16, 640x640, median of 20 passes after 5 warm-up, ms per image",
           "threads": torch.get_num_threads(), "ms_per_image": {}}
    for n in NAMES:
        m = YOLO(f"{W}/{n}.pt").model.fuse().eval().float()
        with torch.inference_mode():
            for _ in range(5):
                m(x)
            t = []
            for _ in range(20):
                t0 = time.perf_counter(); m(x); t.append(time.perf_counter() - t0)
        out["ms_per_image"][n] = round(statistics.median(t) * 1000 / 16, 1)
        print(n, out["ms_per_image"][n], flush=True)
    json.dump(out, open(ROOT + "/data/waid_official/cpu_fixed_times.json", "w"), indent=2)


if __name__ == "__main__":
    main()
