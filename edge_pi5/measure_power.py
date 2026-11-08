"""Log Raspberry Pi 5 board power from its PMIC for a fixed window (idle / active power check).

Reads `vcgencmd pmic_read_adc` every --interval seconds, multiplies each rail's current by its
voltage, and sums the rails. This is the power delivered by the Pi 5 PMIC to the board's own
rails (SoC, RAM, 3.3 V/1.8 V peripherals incl. the CSI camera); it excludes PMIC conversion
losses and anything powered from the USB ports (e.g. a 4G modem), so it is a lower bound on
node power. For the complete node, put a USB-C power meter between battery and Pi as well and
note its reading; the script prints both side by side if --meter-wh is given.

Usage (on the Pi, node in the state to measure, e.g. idle with camera standby):
  python3 measure_power.py --minutes 60 --interval 5 --out idle_power.csv
  python3 measure_power.py --minutes 60 --out idle_power.csv --meter-wh 1.9   # with USB meter reading
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import time
from pathlib import Path

LINE = re.compile(r"^\s*(\S+)_(A|V)\s+(?:current|volt)\(\d+\)=([0-9.]+)(?:A|V)\s*$")


def parse_adc(text: str) -> float:
    """Sum of current x voltage over the rails reported by `vcgencmd pmic_read_adc`, in watts."""
    amps, volts = {}, {}
    for line in text.splitlines():
        m = LINE.match(line)
        if m:
            (amps if m.group(2) == "A" else volts)[m.group(1)] = float(m.group(3))
    return sum(a * volts[r] for r, a in amps.items() if r in volts)


def read_power() -> float:
    out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, check=True).stdout
    return parse_adc(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Pi 5 PMIC power logger")
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--interval", type=float, default=5.0, help="Seconds between samples")
    p.add_argument("--out", type=Path, default=Path("power_log.csv"))
    p.add_argument("--meter-wh", type=float, default=None, help="Energy shown by a USB-C meter over the same window")
    args = p.parse_args()

    end = time.time() + 60 * args.minutes
    samples = []
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unix_time", "pmic_rail_power_w"])
        while time.time() < end:
            watts = read_power()
            samples.append(watts)
            w.writerow([f"{time.time():.1f}", f"{watts:.4f}"])
            f.flush()
            time.sleep(args.interval)
    hours = args.minutes / 60
    print(f"{len(samples)} samples over {args.minutes:.0f} min")
    print(f"PMIC rail power: mean {statistics.mean(samples):.3f} W, median {statistics.median(samples):.3f} W, "
          f"min {min(samples):.3f} W, max {max(samples):.3f} W (lower bound on node power)")
    if args.meter_wh is not None:
        print(f"USB meter: {args.meter_wh:.3f} Wh in {hours:.2f} h = {args.meter_wh / hours:.3f} W mean (complete node)")
    print(f"log: {args.out}")


if __name__ == "__main__":
    main()
