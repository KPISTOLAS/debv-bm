"""Edge-tier benchmark: per-frame cost, transferred bytes, delivery latency, outage buffering.

Runs the released edge path (capture_preprocess.preprocess -> qc_ok -> write_record ->
upload) on a folder of raw camera frames and logs one CSV row per frame. Run it on each
Raspberry Pi 5 with the same --outbox (the share mounted from the processing node); start
all five nodes together for the multi-node run, then aggregate with summarize_edge_bench.py.

Modes
  edge      the DEBV-BM arrangement: preprocess + QC gate on the node, upload the JPEG.
  baseline  upload every raw frame unchanged (preprocessing would happen centrally).

Outage simulation
  --outage-start S --outage-frames N marks frames S..S+N-1 as "link down": their upload is
  skipped and they stay in --buffer. When the link returns, the buffer is flushed before the
  next frame (as capture_preprocess.py --retry-buffer does), and the flush time, buffer
  peak (files, bytes), and per-record delivery delays are logged.

Columns: node, mode, frame, source, t_capture (s since start), pre_ms, qc, reason,
bytes_raw, bytes_sent, upload_ms, state (uploaded/buffered/rejected/flushed),
deliver_delay_s (capture -> arrival in outbox), pending_files / pending_bytes (records waiting
for the link: the outage backlog), local_files / local_bytes (everything kept on the node;
the released edge code copies accepted frames to the outbox and keeps the local copy).

Usage (on each Pi):
  python bench_edge.py --input raw_frames/ --outbox /mnt/tier2/inbox --mode edge \
      --node-id pi-03 --out bench_pi-03_edge.csv [--thermal] [--interval 2.0] \
      [--outage-start 100 --outage-frames 60]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import socket
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_preprocess import list_images, preprocess, qc_ok, upload, write_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM edge-tier benchmark")
    p.add_argument("--input", type=Path, required=True, help="Folder of raw camera frames")
    p.add_argument("--outbox", type=Path, required=True, help="Processing-node inbox (mounted share)")
    p.add_argument("--buffer", type=Path, default=Path("bench_buffer"))
    p.add_argument("--mode", choices=["edge", "baseline"], required=True)
    p.add_argument("--node-id", default=socket.gethostname())
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--thermal", action="store_true")
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--blur-thr", type=float, default=60.0)
    p.add_argument("--dark-thr", type=float, default=12.0)
    p.add_argument("--bright-thr", type=float, default=245.0)
    p.add_argument("--interval", type=float, default=0.0, help="Seconds between captures (0 = as fast as possible)")
    p.add_argument("--outage-start", type=int, default=-1)
    p.add_argument("--outage-frames", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def buffer_state(buf: Path) -> tuple[int, int]:
    files = [f for f in buf.glob("*") if f.is_file() and f.suffix in (".jpg", ".json", ".raw")]
    return sum(f.suffix != ".json" for f in files), sum(f.stat().st_size for f in files)


def main() -> None:
    args = parse_args()
    frames = list_images(args.input)
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        raise SystemExit(f"no frames in {args.input}")
    args.buffer.mkdir(parents=True, exist_ok=True)
    args.outbox.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    pending: list[dict] = []  # buffered records waiting for the link
    rows: list[dict] = []
    fields = ["node", "mode", "frame", "source", "t_capture", "pre_ms", "qc", "reason", "bytes_raw", "bytes_sent",
              "upload_ms", "state", "deliver_delay_s", "pending_files", "pending_bytes", "local_files", "local_bytes"]

    def pending_state() -> tuple[int, int]:
        return len(pending), sum(int(p["row"].get("bytes_sent") or 0) for p in pending)

    def send(stem: str, ext: str) -> bool:
        if ext == ".raw":
            try:
                shutil.copy2(args.buffer / f"{stem}{ext}", args.outbox / f"{stem}.jpg")
                return True
            except OSError:
                return False
        return upload(args.buffer, args.outbox, stem)

    for i, path in enumerate(frames):
        if args.interval > 0:
            wait = t0 + i * args.interval - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        link_down = args.outage_start >= 0 and args.outage_start <= i < args.outage_start + args.outage_frames
        if not link_down and pending:  # link back: flush the buffer first
            t_flush = time.perf_counter()
            for rec in pending:
                ok = send(rec["stem"], rec["ext"])
                rec_row = rec["row"]
                rec_row["state"] = "flushed" if ok else "flush_failed"
                rec_row["deliver_delay_s"] = round(time.perf_counter() - t0 - rec_row["t_capture"], 3)
            nb, bb = buffer_state(args.buffer)
            pf, pb = pending_state()
            rows.append({"node": args.node_id, "mode": args.mode, "frame": i, "source": "FLUSH",
                         "t_capture": round(t_flush - t0, 3), "upload_ms": round(1000 * (time.perf_counter() - t_flush), 1),
                         "state": f"flush_{len(pending)}", "pending_files": pf, "pending_bytes": pb,
                         "local_files": nb, "local_bytes": bb})
            pending.clear()
        t_cap = time.perf_counter() - t0
        raw_bytes = path.stat().st_size
        row = {"node": args.node_id, "mode": args.mode, "frame": i, "source": path.name, "t_capture": round(t_cap, 3),
               "bytes_raw": raw_bytes}
        stem = f"{args.node_id}_{i:06d}"
        ts = time.perf_counter()
        if args.mode == "edge":
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            bgr = preprocess(bgr, args.max_side, thermal=args.thermal, colour=False)
            ok, reason, _ = qc_ok(bgr, args.blur_thr, args.dark_thr, args.bright_thr)
            row.update(qc=ok, reason=reason)
            if not ok:
                row.update(pre_ms=round(1000 * (time.perf_counter() - ts), 1), bytes_sent=0, state="rejected")
                rows.append(row)
                continue
            jpg = write_record(bgr, stem, args.buffer, {"node_id": args.node_id, "source": path.name})
            ext = ".jpg"
            row["bytes_sent"] = jpg.stat().st_size + (args.buffer / f"{stem}.json").stat().st_size
        else:
            shutil.copy2(path, args.buffer / f"{stem}.raw")
            ext = ".raw"
            row.update(qc="", reason="", bytes_sent=raw_bytes)
        row["pre_ms"] = round(1000 * (time.perf_counter() - ts), 1)
        if link_down:
            row["state"] = "buffered"
            pending.append({"stem": stem, "ext": ext, "row": row})
        else:
            tu = time.perf_counter()
            ok = send(stem, ext)
            row["upload_ms"] = round(1000 * (time.perf_counter() - tu), 1)
            row["state"] = "uploaded" if ok else "upload_failed"
            row["deliver_delay_s"] = round(time.perf_counter() - t0 - t_cap, 3)
        row["pending_files"], row["pending_bytes"] = pending_state()
        row["local_files"], row["local_bytes"] = buffer_state(args.buffer)
        rows.append(row)

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
