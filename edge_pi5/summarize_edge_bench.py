"""Aggregate bench_edge.py CSVs from one or more nodes into the system-measurement table.

Reports, per mode (edge vs baseline) and over all nodes: frames, QC rejections, median and
95th-percentile per-frame edge time, bytes sent per captured frame and the reduction against
the raw frames, median / 95th-percentile delivery delay, frames per second per node and
aggregate, and for outage runs the buffer peak (files, MB), flush time, and the delivery
delay of buffered records.

Usage:
  python summarize_edge_bench.py bench_*.csv --out edge_bench_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def pct(v, q):
    return round(float(np.percentile(v, q)), 3) if len(v) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize edge benchmark CSVs")
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for f in args.csvs:
        with f.open(encoding="utf-8") as fh:
            by_mode_rows = list(csv.DictReader(fh))
        for r in by_mode_rows:
            by_mode[r["mode"]].append(r)
    out = {}
    for mode, rows in by_mode.items():
        frames = [r for r in rows if r["source"] != "FLUSH"]
        flushes = [r for r in rows if r["source"] == "FLUSH"]
        nodes = sorted({r["node"] for r in frames})
        pre = [float(r["pre_ms"]) for r in frames if r["pre_ms"]]
        raw = sum(int(r["bytes_raw"]) for r in frames if r["bytes_raw"])
        sent = sum(int(r["bytes_sent"] or 0) for r in frames)
        live = [float(r["deliver_delay_s"]) for r in frames if r["state"] == "uploaded" and r["deliver_delay_s"]]
        late = [float(r["deliver_delay_s"]) for r in frames if r["state"] == "flushed" and r["deliver_delay_s"]]
        span = defaultdict(float)
        for r in frames:
            span[r["node"]] = max(span[r["node"]], float(r["t_capture"]))
        fps_node = [sum(1 for r in frames if r["node"] == n) / span[n] for n in nodes if span[n] > 0]
        out[mode] = {
            "nodes": nodes, "frames": len(frames),
            "qc_rejected": sum(r["state"] == "rejected" for r in frames),
            "edge_ms_median": pct(pre, 50), "edge_ms_p95": pct(pre, 95),
            "bytes_raw_per_frame": round(raw / max(len(frames), 1)),
            "bytes_sent_per_frame": round(sent / max(len(frames), 1)),
            "transfer_reduction_pct": round(100 * (1 - sent / raw), 1) if raw else None,
            "delivery_s_median": pct(live, 50), "delivery_s_p95": pct(live, 95),
            "fps_per_node_mean": round(float(np.mean(fps_node)), 3) if fps_node else None,
            "fps_aggregate": round(float(np.sum(fps_node)), 3) if fps_node else None,
            "outage": {
                "backlog_peak_files": max((int(r["pending_files"] or 0) for r in frames), default=0),
                "backlog_peak_mb": round(max((int(r["pending_bytes"] or 0) for r in frames), default=0) / 1e6, 2),
                "local_store_final_mb": round(max((int(r["local_bytes"] or 0) for r in frames), default=0) / 1e6, 2),
                "flushes": len(flushes),
                "flush_ms": [float(r["upload_ms"]) for r in flushes],
                "buffered_delivery_s_median": pct(late, 50), "buffered_delivery_s_max": pct(late, 100),
            },
        }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
