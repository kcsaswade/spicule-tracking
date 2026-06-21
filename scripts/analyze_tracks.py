"""
Analyze reconstructed spicule trajectories.

Input:
    outputs/50G/v2/track_catalog.csv

Outputs:
    outputs/50G/analysis/v2/
        lifetime_histogram.png
        max_height_histogram.png
        horizontal_drift_histogram.png
        lifetime_vs_height.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.ndimage import gaussian_filter1d

# ==========================================================
# Configuration
# ==========================================================

CATALOG_FILE = "outputs/50G/v2/track_catalog.csv"
OUTPUT_DIR = Path("outputs/50G/analysis/v2/")
TRACK_JSON = "outputs/50G/v2/tracks_full.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load catalog
# ==========================================================

print("Loading track catalog...")

df = pd.read_csv(CATALOG_FILE)
with open(TRACK_JSON, "r") as f:
    track_data = json.load(f)

print(f"Loaded {len(df)} tracks.\n")

# ==========================================================
# Turning-point analysis
# ==========================================================

def count_turning_points(zs, sigma=1.0):
    """
    Count the number of + -> - transitions
    in the smoothed apex trajectory.

    Parameters
    ----------
    zs : list
        Apex heights.
    sigma : float
        Gaussian smoothing width.

    Returns
    -------
    int
        Number of turning points.
    """

    if len(zs) < 5:
        return 0

    zs = np.asarray(zs, dtype=float)

    # suppress 1-frame jitter
    zs_smooth = gaussian_filter1d(
        zs,
        sigma=sigma,
        mode="nearest",
    )

    vz = np.diff(zs_smooth)

    # remove tiny fluctuations
    eps = 0.02
    sign = np.zeros_like(vz)

    sign[vz > eps] = 1
    sign[vz < -eps] = -1

    # remove zero entries
    sign = sign[sign != 0]

    if len(sign) < 2:
        return 0

    turning_points = 0

    for i in range(len(sign) - 1):
        if sign[i] > 0 and sign[i + 1] < 0:
            turning_points += 1

    return turning_points

# ==========================================================
# Summary statistics
# ==========================================================

print("========== TRACK SUMMARY ==========")
print(f"Total tracks           : {len(df)}")
print(f"Mean lifetime (frames) : {df['lifetime_frames'].mean():.2f}")
print(f"Median lifetime        : {df['lifetime_frames'].median():.2f}")
print(f"Min lifetime           : {df['lifetime_frames'].min()}")
print(f"Max lifetime           : {df['lifetime_frames'].max()}")

print()
print("Lifetime breakdown:")
print(f"  1 frame  : {(df['lifetime_frames'] == 1).sum()}")
print(f"  2 frames : {(df['lifetime_frames'] == 2).sum()}")
print(f"  3 frames : {(df['lifetime_frames'] == 3).sum()}")
print(f"  >= 5     : {(df['lifetime_frames'] >= 5).sum()}")
print(f"  >= 10    : {(df['lifetime_frames'] >= 10).sum()}")
print(f"  >= 20    : {(df['lifetime_frames'] >= 20).sum()}")
print(f"  >= 50    : {(df['lifetime_frames'] >= 50).sum()}")

print()
print("Maximum height:")
print(f"  Mean max(z): {df['max_z'].mean():.3f}")
print(f"  Median    : {df['max_z'].median():.3f}")
print(f"  Max       : {df['max_z'].max():.3f}")

print()
print("Horizontal drift:")
print(f"  Mean Δx : {df['horizontal_drift'].mean():.3f}")
print(f"  Std(Δx) : {df['horizontal_drift'].std():.3f}")

print()
print("Missed frames:")
print(f"  Mean missed : {df['missed_frames'].mean():.2f}")
print(f"  Max missed  : {df['missed_frames'].max()}")

print()
print("Turning-point analysis:")

turning_counts = []

for tr in track_data:
    nturn = count_turning_points(tr["zs"])
    turning_counts.append(nturn)

turning_counts = np.asarray(turning_counts)
df["turning_points"] = turning_counts

print(f"  Mean turning points : {turning_counts.mean():.2f}")
print(f"  Max turning points  : {turning_counts.max()}")

print()
print("Turning-point breakdown:")
print(f"  0 turning points : {(turning_counts == 0).sum()}")
print(f"  1 turning point  : {(turning_counts == 1).sum()}")
print(f"  2 turning points : {(turning_counts == 2).sum()}")
print(f"  >=3 turning pts  : {(turning_counts >= 3).sum()}")

# ==========================================================
# Plot 1: Lifetime histogram
# ==========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["lifetime_frames"],
    bins=20,
    edgecolor="black",
)

plt.xlabel("Track lifetime (frames)")
plt.ylabel("Count")
plt.title("Distribution of spicule lifetimes")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "lifetime_histogram.png",
    dpi=150,
)
plt.close()

# ==========================================================
# Plot 2: Maximum height histogram
# ==========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["max_z"],
    bins=20,
    edgecolor="black",
)

plt.xlabel("Maximum apex height (Mm)")
plt.ylabel("Count")
plt.title("Distribution of maximum apex heights")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "max_height_histogram.png",
    dpi=150,
)
plt.close()

# ==========================================================
# Plot 3: Horizontal drift histogram
# ==========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["horizontal_drift"],
    bins=20,
    edgecolor="black",
)

plt.xlabel(r"$\Delta x$ (Mm)")
plt.ylabel("Count")
plt.title("Distribution of horizontal drift")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "horizontal_drift_histogram.png",
    dpi=150,
)
plt.close()

# ==========================================================
# Plot 4: Lifetime vs maximum height
# ==========================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    df["lifetime_frames"],
    df["max_z"],
    s=20,
    alpha=0.7,
)

plt.xlabel("Lifetime (frames)")
plt.ylabel("Maximum apex height (Mm)")
plt.title("Lifetime vs maximum height")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "lifetime_vs_height.png",
    dpi=150,
)
plt.close()

# ==========================================================
# Bonus: Top longest-lived tracks
# ==========================================================

print("\n========== LONGEST-LIVED TRACKS ==========")

cols = [
    "track_id",
    "lifetime_frames",
    "duration",
    "max_z",
    "horizontal_drift",
    "missed_frames",
]

print(
    df.sort_values(
        "lifetime_frames",
        ascending=False,
    )[cols].head(20).to_string(index=False)
)

print("\n========== POSSIBLE STITCHED TRACKS ==========")

stitched = []

for tr, nturn in zip(track_data, turning_counts):
    if nturn >= 2:
        stitched.append(
            (
                tr["track_id"],
                tr["lifetime_frames"],
                nturn,
                tr["max_z"],
            )
        )

if len(stitched) == 0:
    print("None found.")
else:
    stitched.sort(key=lambda x: (-x[2], -x[1]))

    print(
        f"{'ID':>4} "
        f"{'Lifetime':>10} "
        f"{'Turns':>8} "
        f"{'Max_z':>8}"
    )

    for row in stitched[:20]:
        print(
            f"{row[0]:4d} "
            f"{row[1]:10d} "
            f"{row[2]:8d} "
            f"{row[3]:8.2f}"
        )

# ==========================================================
# Plot 5: Turning-point histogram
# ==========================================================

plt.figure(figsize=(8, 5))

bins = np.arange(
    turning_counts.max() + 2
) - 0.5

plt.hist(
    turning_counts,
    bins=bins,
    edgecolor="black",
)

plt.xlabel("Number of turning points")
plt.ylabel("Count")
plt.title("Distribution of trajectory turning points")

plt.xticks(
    np.arange(turning_counts.max() + 1)
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "turning_point_histogram.png",
    dpi=150,
)
plt.close()

# ==========================================================
# Plot 6: Lifetime vs turning points
# ==========================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    df["lifetime_frames"],
    df["turning_points"],
    s=25,
    alpha=0.7,
)

plt.xlabel("Lifetime (frames)")
plt.ylabel("Number of turning points")
plt.title("Lifetime vs trajectory complexity")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "lifetime_vs_turning_points.png",
    dpi=150,
)
plt.close()

print()
print(f"Saved analysis figures to: {OUTPUT_DIR}")