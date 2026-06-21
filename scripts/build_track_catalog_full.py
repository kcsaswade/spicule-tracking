"""
Build a catalog of reconstructed spicule tracks.

Input:
    outputs/50G/envelope_detections.json

Outputs:
    outputs/50G/tracks_full.csv
    outputs/50G/tracks_full.json
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np

from src.tracker import (
    load_detections,
    run_tracker,
)

# ==========================================================
# Configuration
# ==========================================================

DETECTION_FILE = "outputs/50G/envelope_detections.json"

OUTPUT_CSV = "outputs/50G/tracks_full.csv"
OUTPUT_JSON = "outputs/50G/tracks_full.json"

# Tracker parameters
# (must match those used in animate_tracker.py)
MAX_COST = 3.0
MAX_MISSED = 2

# ==========================================================
# Build tracks
# ==========================================================

print("Loading detections...")
detections = load_detections(DETECTION_FILE)

print("Running tracker...")
tracks = run_tracker(
    detections,
    max_cost=MAX_COST,
    max_missed=MAX_MISSED,
)

print(f"Found {len(tracks)} tracks.")

# ==========================================================
# Build catalog
# ==========================================================

catalog = []

for tr in tracks:

    lifetime_frames = len(tr.frames)

    duration = (
        tr.times[-1] - tr.times[0]
        if lifetime_frames > 1
        else 0.0
    )

    total_missed = (
        tr.frames[-1]
        - tr.frames[0]
        + 1
        - lifetime_frames
    )

    # row = {
    #     # identifiers
    #     "track_id": tr.track_id,

    #     # frame information
    #     "start_frame": tr.frames[0],
    #     "end_frame": tr.frames[-1],
    #     "lifetime_frames": lifetime_frames,

    #     # physical time
    #     "start_time": tr.times[0],
    #     "end_time": tr.times[-1],
    #     "duration": duration,

    #     # starting location
    #     "start_x": tr.xs[0],
    #     "start_z": tr.zs[0],

    #     # ending location
    #     "end_x": tr.xs[-1],
    #     "end_z": tr.zs[-1],

    #     # extrema
    #     "min_z": float(np.min(tr.zs)),
    #     "max_z": float(np.max(tr.zs)),

    #     # derived quantities
    #     "horizontal_drift": tr.xs[-1] - tr.xs[0],
    #     "vertical_range": np.max(tr.zs) - np.min(tr.zs),

    #     # tracker diagnostics
    #     "assigned_detections": lifetime_frames,
    #     "missed_frames": total_missed,
    # }
    row = {
        "track_id": tr.track_id,
        "frames": tr.frames,
        "times": tr.times,
        "xs": tr.xs,
        "zs": tr.zs,
        "start_frame": tr.frames[0],
        "end_frame": tr.frames[-1],
        "lifetime_frames": len(tr.frames),
        "start_time": tr.times[0],
        "end_time": tr.times[-1],
        "duration": tr.times[-1] - tr.times[0],
        "start_x": tr.xs[0],
        "start_z": tr.zs[0],
        "end_x": tr.xs[-1],
        "end_z": tr.zs[-1],
        "min_z": min(tr.zs),
        "max_z": max(tr.zs),
        "horizontal_drift": tr.xs[-1] - tr.xs[0],
        "vertical_range": max(tr.zs) - min(tr.zs),
        "assigned_detections": len(tr.frames),
        "missed_frames": tr.missed_frames,
    }

    catalog.append(row)

# ==========================================================
# Save outputs
# ==========================================================

df = pd.DataFrame(catalog)
df = df.sort_values("track_id")

df.to_csv(
    OUTPUT_CSV,
    index=False,
)

with open(OUTPUT_JSON, "w") as f:
    json.dump(catalog, f, indent=2)

print()
print(f"Saved CSV : {OUTPUT_CSV}")
print(f"Saved JSON: {OUTPUT_JSON}")

# ==========================================================
# Baseline statistics
# ==========================================================

lengths = df["lifetime_frames"]

print()
print("========== TRACK CATALOG SUMMARY ==========")
print(f"Total tracks           : {len(df)}")
print(f"Mean lifetime          : {lengths.mean():.2f}")
print(f"Median lifetime        : {lengths.median():.2f}")
print(f"Min lifetime           : {lengths.min()}")
print(f"Max lifetime           : {lengths.max()}")

print()
print("Track length counts:")
print(f"  1 frame  : {(lengths == 1).sum()}")
print(f"  2 frames : {(lengths == 2).sum()}")
print(f"  3 frames : {(lengths == 3).sum()}")

print()
print("Long-lived tracks:")
print(f"  >=  5 frames : {(lengths >= 5).sum()}")
print(f"  >= 10 frames : {(lengths >= 10).sum()}")
print(f"  >= 20 frames : {(lengths >= 20).sum()}")
print(f"  >= 50 frames : {(lengths >= 50).sum()}")