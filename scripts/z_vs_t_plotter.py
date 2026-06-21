"""
Plot z(t) for all reconstructed spicule tracks.

Input:
    outputs/50G/v2/tracks_full.json

Outputs:
    outputs/50G/analysis/v2/z_vs_t_all_tracks.png
    outputs/50G/analysis/v2/z_vs_t_top20.png
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ==========================================================
# Configuration
# ==========================================================

TRACK_FILE = Path("outputs/50G/v2/tracks_full.json")
OUTPUT_DIR = Path("outputs/50G/analysis/v2")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load tracks
# ==========================================================

print("Loading tracks...")

with open(TRACK_FILE, "r") as f:
    tracks = json.load(f)

print(f"Loaded {len(tracks)} tracks.")

# ==========================================================
# Plot 1: All tracks (spaghetti plot)
# ==========================================================

plt.figure(figsize=(12, 8))

for tr in tracks:

    plt.plot(
        tr["times"],
        tr["zs"],
        lw=1.0,
        alpha=0.25,
    )

plt.xlabel("Simulation time")
plt.ylabel("Apex height z (Mm)")
plt.title("Reconstructed spicule apex trajectories: $z(t)$")

plt.grid(alpha=0.3)

plt.tight_layout()

outfile = OUTPUT_DIR / "z_vs_t_all_tracks.png"
plt.savefig(outfile, dpi=200)
plt.show()
plt.close()

print(f"Saved: {outfile}")

# ==========================================================
# Plot 2: Top 20 longest-lived tracks
# ==========================================================

tracks_sorted = sorted(
    tracks,
    key=lambda tr: tr["lifetime_frames"],
    reverse=True,
)

top_tracks = tracks_sorted[:20]

fig, ax = plt.subplots(figsize=(12, 8))

cmap = cm.get_cmap("tab20", len(top_tracks))

for i, tr in enumerate(top_tracks):

    t = tr["times"]
    z = tr["zs"]

    ax.plot(
        t,
        z,
        "-o",
        lw=2,
        ms=2,
        color=cmap(i),
        alpha=0.9,
        label=f'ID {tr["track_id"]}',
    )

    # Label the end of the trajectory
    ax.text(
        t[-1],
        z[-1] + 0.08,
        str(tr["track_id"]),
        fontsize=8,
        color=cmap(i),
    )

ax.set_xlabel("Simulation time")
ax.set_ylabel("Apex height z (Mm)")
ax.set_title("Top 20 longest-lived spicule trajectories: $z(t)$")

ax.grid(alpha=0.3)

plt.tight_layout()

outfile = OUTPUT_DIR / "z_vs_t_top20.png"
plt.savefig(outfile, dpi=200)
plt.show()
plt.close()

print(f"Saved: {outfile}")