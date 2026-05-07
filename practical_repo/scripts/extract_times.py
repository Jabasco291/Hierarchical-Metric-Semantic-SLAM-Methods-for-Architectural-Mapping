#!/usr/bin/env python3
"""
parse_sgraphs_timings.py
------------------------
Parses an S-Graphs stdout log file and extracts backend optimization
timings, outputting a CSV in the same format as Hydra/Clio timing CSVs.

Usage
-----
python3 parse_sgraphs_timings.py --log sgraphs.log --output sgraphs_timings.csv

Options
-------
--log PATH      Path to the S-Graphs stdout log file (default: sgraphs.log)
--output PATH   Output CSV file (default: sgraphs_timings.csv)
"""

import argparse
import csv
import re
import statistics
from pathlib import Path


def parse_timings(log_path: Path) -> list[float]:
    """
    Extract per-cycle optimization wall-clock times from S-Graphs log.
    Matches lines like:
      [s_graphs_node-4] time: 0.012[sec]
    """
    pattern = re.compile(r"time:\s*([\d.eE+\-]+)\[sec\]")
    timings = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                timings.append(float(m.group(1)))
    return timings


def write_csv(timings: list[float], output_path: Path, log_name: str, append: bool = False):
    if not timings:
        return

    mean = statistics.mean(timings)
    minimum = min(timings)
    maximum = max(timings)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0

    # Use "a" if append is True, otherwise "w"
    mode = "a" if append else "w"
    file_exists = output_path.exists()

    with open(output_path, mode, newline="") as f:
        writer = csv.writer(f)
        
        # Only write header if we are starting a new file OR the file is empty
        if mode == "w" or not file_exists:
            writer.writerow(["name", "mean[s]", "min[s]", "max[s]", "std-dev[s]"])
        
        # Use the log_name (e.g., "run_1") as the row label
        writer.writerow([log_name, mean, minimum, maximum, std])


def main():
    base_dir = Path("../../graphs/s-graphs/sgraphs_eval_20260426_1657")
    combined_output = base_dir / "all_runs_timings.csv"

    for i in range(1, 6):
        log_path = base_dir / f"run_{i}_sgraph.log"
        
        if not log_path.exists():
            print(f"Skipping: {log_path} not found.")
            continue

        timings = parse_timings(log_path)
        
        # If it's the first run (i=1), we overwrite/create. 
        # For runs 2-5, we append.
        should_append = (i > 1)
        
        write_csv(
            timings, 
            combined_output, 
            log_name=f"run_{i}", 
            append=should_append
        )
        
    print(f"\nDone! All results saved to: {combined_output}")


if __name__ == "__main__":
    main()