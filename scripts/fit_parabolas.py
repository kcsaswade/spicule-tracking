"""
Fit parabolic trajectories to reconstructed spicule apex tracks.

Input:
    outputs/v2/track_catalog.json

Output:
    outputs/v2/parabolic_fit_catalog.csv

Model:
    z(t) = A t^2 + B t + C

where
    initial_velocity      = B
    effective_acceleration = -2 A

Only sufficiently long, single-turn trajectories are accepted
for fitting.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

TRACK_FILE = Path("outputs/v2/track_catalog.json")
OUTPUT_FILE = Path("outputs/v2/parabolic_fit_catalog.csv")

MIN_LIFETIME = 8          # frames
MIN_POINTS = 8            # stored trajectory samples
REQUIRE_ONE_TURN = True

# ----------------------------------------------------------
# Utility functions
# ----------------------------------------------------------

def count_turning_points(zs):
    """
    Count turning points in a z(t) trajectory.

    A turning point is defined as a sign change of dz/dt.
    Flat segments are ignored.
    """

    zs = np.asarray(zs, dtype=float)

    if len(zs) < 3:
        return 0

    dz = np.diff(zs)
    signs = np.sign(dz)

    # propagate nonzero sign through flat regions
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]

    # backwards pass for leading zeros
    for i in range(len(signs) - 2, -1, -1):
        if signs[i] == 0:
            signs[i] = signs[i + 1]

    turns = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            turns += 1

    return turns


def fit_track(times, zs):
    """
    Perform quadratic fit.

    Returns
    -------
    dict containing fit parameters.
    """

    times = np.asarray(times, dtype=float)
    zs = np.asarray(zs, dtype=float)

    # shift origin for numerical stability
    tau = times - times[0]

    # quadratic fit
    A, B, C = np.polyfit(tau, zs, deg=2)

    z_fit = np.polyval([A, B, C], tau)

    residuals = zs - z_fit

    rss = np.sum(residuals ** 2)
    tss = np.sum((zs - np.mean(zs)) ** 2)

    if tss > 0:
        r2 = 1.0 - rss / tss
    else:
        r2 = np.nan

    rmse = np.sqrt(np.mean(residuals ** 2))

    # physical interpretation
    initial_velocity = B
    effective_acceleration = -2.0 * A

    if np.abs(A) > 1e-12:
        apex_time = -B / (2.0 * A)
        apex_height = np.polyval([A, B, C], apex_time)
    else:
        apex_time = np.nan
        apex_height = np.nan

    return {
        "A": A,
        "B": B,
        "C": C,
        "initial_velocity": initial_velocity,
        "effective_acceleration": effective_acceleration,
        "fitted_apex_time": apex_time,
        "fitted_apex_height": apex_height,
        "r2": r2,
        "rmse": rmse,
    }


# ==========================================================
# Load tracks
# ==========================================================

print("Loading track catalog...")

with open(TRACK_FILE, "r") as f:
    tracks = json.load(f)

print(f"Loaded {len(tracks)} tracks.\n")

# ==========================================================
# Main fitting loop
# ==========================================================

rows = []

n_attempted = 0
n_accepted = 0

for tr in tracks:

    lifetime = tr["lifetime_frames"]

    times = tr.get("times", [])
    xs = tr.get("xs", [])
    zs = tr.get("zs", [])

    n_points = len(zs)

    turning_points = count_turning_points(zs)

    accepted = True

    if lifetime < MIN_LIFETIME:
        accepted = False

    if n_points < MIN_POINTS:
        accepted = False

    if REQUIRE_ONE_TURN and turning_points != 1:
        accepted = False

    row = {
        "track_id": tr["track_id"],
        "lifetime_frames": lifetime,
        "duration": tr["duration"],
        "n_points": n_points,
        "turning_points": turning_points,
        "observed_max_height": tr["max_z"],
        "observed_min_height": tr["min_z"],
        "horizontal_drift": tr["horizontal_drift"],
        "accepted": accepted,
    }

    if accepted:

        n_attempted += 1

        try:
            fit = fit_track(times, zs)

            row.update(fit)

            n_accepted += 1

        except Exception as exc:

            print(
                f"Warning: fit failed for "
                f"track {tr['track_id']} ({exc})"
            )

            row.update({
                "A": np.nan,
                "B": np.nan,
                "C": np.nan,
                "initial_velocity": np.nan,
                "effective_acceleration": np.nan,
                "fitted_apex_time": np.nan,
                "fitted_apex_height": np.nan,
                "r2": np.nan,
                "rmse": np.nan,
            })

            row["accepted"] = False

    else:

        row.update({
            "A": np.nan,
            "B": np.nan,
            "C": np.nan,
            "initial_velocity": np.nan,
            "effective_acceleration": np.nan,
            "fitted_apex_time": np.nan,
            "fitted_apex_height": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
        })

    rows.append(row)

# ==========================================================
# Save catalog
# ==========================================================

df = pd.DataFrame(rows)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

# ==========================================================
# Summary
# ==========================================================

print("========== PARABOLIC FIT SUMMARY ==========")
print(f"Total tracks              : {len(df)}")
print(f"Fit candidates            : {n_attempted}")
print(f"Successful fits           : {n_accepted}")

good = df[df["accepted"] == True]

if len(good) > 0:

    print()
    print(f"Mean R²                   : {good['r2'].mean():.3f}")
    print(f"Median R²                 : {good['r2'].median():.3f}")
    print(f"Min R²                    : {good['r2'].min():.3f}")

    print()
    print("Initial velocity (Mm/s):")
    print(f"  Mean                    : {good['initial_velocity'].mean():.3f}")
    print(f"  Median                  : {good['initial_velocity'].median():.3f}")

    print()
    print("Effective acceleration (Mm/s²):")
    print(
        f"  Mean                    : "
        f"{good['effective_acceleration'].mean():.3f}"
    )
    print(
        f"  Median                  : "
        f"{good['effective_acceleration'].median():.3f}"
    )

    print()
    print("RMSE (Mm):")
    print(f"  Mean                    : {good['rmse'].mean():.3f}")
    print(f"  Median                  : {good['rmse'].median():.3f}")

    print()
    print("========== BEST FITS ==========")

    cols = [
        "track_id",
        "lifetime_frames",
        "r2",
        "rmse",
        "observed_max_height",
        "fitted_apex_height",
    ]

    print(
        good.sort_values(
            "r2",
            ascending=False,
        )[cols].head(15).to_string(index=False)
    )

print()
print(f"Saved parabolic fit catalog to:")
print(f"  {OUTPUT_FILE}")