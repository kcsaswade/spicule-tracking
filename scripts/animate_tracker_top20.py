"""
Animate reconstructed spicule tracks.

This script is intentionally independent of the detector.
It uses:
    - original HDF5 file (background movie)
    - outputs/10G/envelope_detections.json (detector output)
    - src.tracker.run_tracker() (trajectory reconstruction)

Detector improvements only require regenerating
envelope_detections.json.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

from src.io import SpiculeDataset
from src.tracker_v2 import (
    load_detections,
    run_tracker,
)

# ==========================================================
# Configuration
# ==========================================================

DATAFILE = "data/raw/spicules_temperature_final.h5"
DETECTION_FILE = "outputs/10G/envelope_detections.json"
OUTPUT_MOVIE = "outputs/10G/movies/tracker_overlay_top20_v2.mp4"

FPS = 10
TAIL_LENGTH = 15          # previous detections to show

# tracker parameters (must match those used previously)
MAX_COST = 3.0
MAX_MISSED = 2

# display parameters
LOGT_MIN = 3.57
LOGT_MAX = 6.00

# ==========================================================
# Load dataset and run tracker
# ==========================================================

print("Loading dataset...")
dataset = SpiculeDataset(DATAFILE)

print("Loading detections...")
detections_by_frame = load_detections(DETECTION_FILE)

print("Running tracker...")
tracks = run_tracker(
    detections_by_frame,
    max_cost=MAX_COST,
    max_missed=MAX_MISSED,
)
all_tracks = sorted(
    tracks,
    key=lambda tr: len(tr.frames),
    reverse=True,
)

display_tracks = all_tracks[:20]
display_ids = {tr.track_id for tr in display_tracks}

print(f"Loaded top {len(display_tracks)} tracks.")

# ----------------------------------------------------------
# Assign a persistent color to every track
# ----------------------------------------------------------

cmap = cm.get_cmap("tab20")

track_colors = {
    tr.track_id: cmap(tr.track_id % 20)
    for tr in display_tracks
}

# ==========================================================
# Figure setup
# ==========================================================

fig, ax = plt.subplots(
    figsize=(8, 10),
    constrained_layout=True,
)

extent = [
    dataset.x.min(),
    dataset.x.max(),
    dataset.z.min(),
    dataset.z.max(),
]

# ==========================================================
# Animation callback
# ==========================================================

def update(frame_idx):

    ax.clear()

    # ------------------------------------------------------
    # Background image (same representation as original movie)
    # ------------------------------------------------------

    T = dataset.get_frame(frame_idx)
    logT = np.clip(
        np.log10(T),
        LOGT_MIN,
        LOGT_MAX,
    )

    ax.imshow(
        logT,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gist_rainbow",
        vmin=LOGT_MIN,
        vmax=LOGT_MAX,
    )

    # ------------------------------------------------------
    # Draw detections actually used by the tracker
    # ------------------------------------------------------

    current_detections = detections_by_frame.get(
        frame_idx,
        [],
    )

    x_det = [d["x"] for d in current_detections]
    z_det = [d["z"] for d in current_detections]

    ax.scatter(
        x_det,
        z_det,
        s=20,
        c="red",
        edgecolors="black",
        linewidths=0.3,
        zorder=10,
        label="Detections",
    )

    # ------------------------------------------------------
    # Draw track tails
    # ------------------------------------------------------

    for tr in display_tracks:

        # Track not yet born
        if frame_idx < tr.frames[0]:
            continue

        # Track already dead
        if frame_idx > tr.frames[-1]:
            continue

        # If the tracker missed this frame, simply skip drawing
        if frame_idx not in tr.frames:
            continue

        idx = tr.frames.index(frame_idx)

        start = max(
            0,
            idx - TAIL_LENGTH + 1,
        )

        xs = tr.xs[start:idx + 1]
        zs = tr.zs[start:idx + 1]

        color = track_colors[tr.track_id]

        # trajectory tail
        ax.plot(
            xs,
            zs,
            "-o",
            color=color,
            lw=2.5,
            ms=3,
            alpha=0.9,
            zorder=8,
        )

        # current track position
        ax.scatter(
            xs[-1],
            zs[-1],
            s=35,
            color=color,
            edgecolors="black",
            linewidths=0.4,
            zorder=11,
        )

        # track ID label
        ax.text(
            xs[-1],
            zs[-1] + 0.15,
            str(tr.track_id),
            fontsize=8,
            color=color,
            ha="center",
            zorder=12,
        )

    # ------------------------------------------------------
    # Cosmetics
    # ------------------------------------------------------

    ax.set_xlim(dataset.x.min(), dataset.x.max())
    ax.set_ylim(dataset.z.min(), dataset.z.max())

    ax.set_xlabel("x (Mm)")
    ax.set_ylabel("z (Mm)")

    current_time = dataset.time[frame_idx]

    ax.set_title(
        f"Spicule Tracker"
        f"\nFrame {frame_idx:03d}/{dataset.n_frames - 1:03d}"
        f"    t = {current_time:.2f}"
    )

# ==========================================================
# Generate animation
# ==========================================================

print("Rendering animation...")

anim = FuncAnimation(
    fig,
    update,
    frames=dataset.n_frames,
    interval=1000 // FPS,
    blit=False,
)

# Requires ffmpeg to be installed and available on PATH.
anim.save(
    OUTPUT_MOVIE,
    writer="ffmpeg",
    fps=FPS,
    dpi=150,
)

plt.close(fig)

print()
print("Done.")
print(f"Saved animation to: {OUTPUT_MOVIE}")