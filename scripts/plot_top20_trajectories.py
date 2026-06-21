from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.cm as cm

TRACK_FILE = Path("outputs/10G/v2/tracks_full.json")
OUTPUT_FILE = Path("outputs/10G/analysis/v2/top20_trajectories.png")

# ----------------------------------------------------
# Load full track catalog
# ----------------------------------------------------

with open(TRACK_FILE, "r") as f:
    tracks = json.load(f)

# Sort by lifetime
tracks = sorted(
    tracks,
    key=lambda tr: tr["lifetime_frames"],
    reverse=True,
)

top_tracks = tracks[:20]

print("\nTop 20 longest-lived tracks:")
print("-" * 60)
for tr in top_tracks:
    print(
        f"ID={tr['track_id']:3d}   "
        f"lifetime={tr['lifetime_frames']:3d} frames   "
        f"duration={tr['duration']:6.2f} s   "
        f"max_z={tr['max_z']:5.2f} Mm"
    )

# ----------------------------------------------------
# Plot
# ----------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 8))

cmap = cm.get_cmap("tab20", len(top_tracks))

for i, tr in enumerate(top_tracks):

    xs = tr["xs"]
    zs = tr["zs"]

    color = cmap(i)

    ax.plot(
        xs,
        zs,
        "-o",
        lw=2,
        ms=2,
        color=color,
        alpha=0.9,
    )

    # mark starting point
    ax.scatter(
        xs[0],
        zs[0],
        marker="s",
        s=30,
        color=color,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )

    # mark ending point
    ax.scatter(
        xs[-1],
        zs[-1],
        marker="o",
        s=35,
        color=color,
        edgecolors="black",
        linewidths=0.5,
        zorder=6,
    )

    # label by track ID
    ax.text(
        xs[-1],
        zs[-1] + 0.15,
        str(tr["track_id"]),
        fontsize=8,
        color=color,
        weight="bold",
    )

ax.set_xlabel("x (Mm)")
ax.set_ylabel("Apex height z (Mm)")
ax.set_title("Twenty longest-lived reconstructed spicule trajectories")

ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200)
plt.show()

print(f"\nSaved figure to: {OUTPUT_FILE}")