"""
Classify reconstructed spicule trajectories according to how
well they resemble parabolic motion.

Input
-----
outputs/50G/v2/tracks_full.json

Outputs
-------
outputs/50G/parabola_classification/

    classification_catalog.csv
    classification_catalog.json
    summary.txt

    class_A_perfect/
        overview.png
        track_XXX.png

    class_B_partial/
        overview.png
        track_XXX.png

    class_C_endpoint_artifacts/
        overview.png
        track_XXX.png

    class_D_kinky/
        overview.png
        track_XXX.png

    class_E_reject/
        overview.png
        track_XXX.png
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

# ==========================================================
# Configuration
# ==========================================================

TRACK_FILE = Path("outputs/50G/v2/tracks_full.json")

OUTPUT_DIR = Path("outputs/50G/parabola_classification")

CLASS_INFO = {
    "A": "class_A_perfect",
    "B": "class_B_partial",
    "C": "class_C_endpoint_artifacts",
    "D": "class_D_kinky",
    "E": "class_E_reject",
}

MIN_POINTS = 5

# Classification thresholds
R2_GOOD = 0.995
R2_ACCEPT = 0.98

NRMSE_GOOD = 0.03
ENDPOINT_RATIO_LIMIT = 2.0

# ==========================================================
# Create output directories
# ==========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for dirname in CLASS_INFO.values():
    (OUTPUT_DIR / dirname).mkdir(
        parents=True,
        exist_ok=True,
    )

# ==========================================================
# Helper functions
# ==========================================================


def smooth_signal(z):
    """
    Mild smoothing for turning-point counting only.
    """

    if len(z) < 5:
        return z.copy()

    window = min(5, len(z))
    if window % 2 == 0:
        window -= 1

    window = max(window, 3)

    return savgol_filter(
        z,
        window_length=window,
        polyorder=2,
        mode="interp",
    )


def count_turning_points(z):

    z_s = smooth_signal(z)

    dz = np.diff(z_s)

    signs = np.sign(dz)
    signs = signs[signs != 0]

    if len(signs) < 2:
        return 0

    turns = np.sum(
        signs[:-1] * signs[1:] < 0
    )

    return int(turns)


def quadratic_fit(tau, z):
    """
    Robust quadratic fit.
    Returns NaNs if the fit cannot be performed.
    """

    tau = np.asarray(tau, dtype=float)
    z = np.asarray(z, dtype=float)

    # ---------------------------------------------
    # Remove invalid values
    # ---------------------------------------------

    valid = (
        np.isfinite(tau)
        & np.isfinite(z)
    )

    tau = tau[valid]
    z = z[valid]

    # Need at least 3 points
    if len(tau) < 3:
        return (
            np.nan, np.nan, np.nan,
            np.full(len(tau), np.nan),
            np.full(len(tau), np.nan),
            np.nan, np.nan, np.nan,
        )

    # Require at least 3 distinct times
    if len(np.unique(tau)) < 3:
        return (
            np.nan, np.nan, np.nan,
            np.full(len(tau), np.nan),
            np.full(len(tau), np.nan),
            np.nan, np.nan, np.nan,
        )
    
    if (
        not np.all(np.isfinite(tau))
        or not np.all(np.isfinite(z))
    ):
        print(
            f"Track {track_id} contains "
            f"NaN/Inf values."
        )
        print("tau =", tau)
        print("z   =", z)

    if len(np.unique(tau)) < 3:
        print(
            f"Track {track_id} has "
            f"insufficient distinct time values."
        )

    try:
        coeff = np.polyfit(tau, z, 2)

    except np.linalg.LinAlgError:
        print(
            f"WARNING: polyfit failed "
            f"(n={len(tau)})"
        )
        print("tau =", tau)
        print("z   =", z)

        return (
            np.nan, np.nan, np.nan,
            np.full(len(tau), np.nan),
            np.full(len(tau), np.nan),
            np.nan, np.nan, np.nan,
        )

    A, B, C = coeff

    z_fit = np.polyval(coeff, tau)

    residual = z - z_fit

    rmse = np.sqrt(
        np.mean(residual**2)
    )

    ss_res = np.sum(residual**2)
    ss_tot = np.sum(
        (z - np.mean(z))**2
    )

    if ss_tot < 1e-12:
        r2 = 1.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    vertical_range = max(
        z.max() - z.min(),
        1e-8,
    )

    nrmse = rmse / vertical_range

    return (
        A,
        B,
        C,
        z_fit,
        residual,
        rmse,
        nrmse,
        r2,
    )


def endpoint_ratio(residual):

    n = len(residual)

    if n < 6:
        return 1.0

    n_edge = min(2, n // 4)

    edge_idx = (
        list(range(n_edge))
        +
        list(range(n - n_edge, n))
    )

    middle_idx = [
        i for i in range(n)
        if i not in edge_idx
    ]

    if len(middle_idx) == 0:
        return 1.0

    edge_rms = np.sqrt(
        np.mean(
            residual[edge_idx] ** 2
        )
    )

    middle_rms = np.sqrt(
        np.mean(
            residual[middle_idx] ** 2
        )
    )

    return edge_rms / (middle_rms + 1e-8)


def classify_track(
    n_points,
    turning_points,
    r2,
    nrmse,
    endpoint_ratio_value,
):

    if n_points < MIN_POINTS:
        return "E"

    if turning_points >= 3:
        return "E"

    if turning_points == 0:
        return "B"

    if turning_points == 1:

        if (
            r2 > R2_GOOD
            and nrmse < NRMSE_GOOD
        ):
            return "A"

        if (
            endpoint_ratio_value
            > ENDPOINT_RATIO_LIMIT
        ):
            return "C"

        return "D"

    if turning_points == 2:

        if r2 > R2_ACCEPT:
            return "D"
        else:
            return "E"

    return "E"


# ==========================================================
# Load tracks
# ==========================================================

print("Loading tracks...")

with open(TRACK_FILE, "r") as f:
    tracks = json.load(f)

print(f"Loaded {len(tracks)} tracks.\n")

# ==========================================================
# Main analysis loop
# ==========================================================

catalog = []

class_members = {
    "A": [],
    "B": [],
    "C": [],
    "D": [],
    "E": [],
}

for track in tracks:

    track_id = track["track_id"]

    t = np.array(track["times"])
    z = np.array(track["zs"])

    tau = t - t[0]

    n_points = len(z)

    turns = count_turning_points(z)

    (
        A,
        B,
        C,
        z_fit,
        residual,
        rmse,
        nrmse,
        r2,
    ) = quadratic_fit(
        tau,
        z,
    )
    if np.isnan(r2):
        print(
            f"Skipping track "
            f"{track_id}: invalid fit."
        )
        continue

    ep_ratio = endpoint_ratio(
        residual
    )

    tau_apex = -B / (2 * A) if A != 0 else np.nan

    apex_inside = (
        tau_apex >= tau[0]
        and tau_apex <= tau[-1]
    )

    label = classify_track(
        n_points=n_points,
        turning_points=turns,
        r2=r2,
        nrmse=nrmse,
        endpoint_ratio_value=ep_ratio,
    )

    class_members[label].append(track)

    catalog.append({
        "track_id": track_id,
        "class": label,
        "n_points": n_points,
        "lifetime_frames": track["lifetime_frames"],
        "turning_points": turns,
        "A": A,
        "B": B,
        "C": C,
        "rmse": rmse,
        "normalized_rmse": nrmse,
        "r2": r2,
        "endpoint_ratio": ep_ratio,
        "tau_apex": tau_apex,
        "apex_inside": apex_inside,
    })

    # ------------------------------------------------------
    # Individual diagnostic figure
    # ------------------------------------------------------

    plt.figure(figsize=(7, 5))

    plt.plot(
        tau,
        z,
        "o-",
        ms=3,
        label="Trajectory",
    )

    plt.plot(
        tau,
        z_fit,
        "--",
        lw=2,
        label="Quadratic fit",
    )

    plt.xlabel(
        r"Time since birth $\tau$ (s)"
    )
    plt.ylabel(
        "Apex height z (Mm)"
    )

    plt.title(
        f"Track {track_id}   "
        f"Class {label}\n"
        f"$R^2$={r2:.4f}, "
        f"turns={turns}"
    )

    plt.legend()
    plt.tight_layout()

    outfile = (
        OUTPUT_DIR
        / CLASS_INFO[label]
        / f"track_{track_id:03d}.png"
    )

    plt.savefig(
        outfile,
        dpi=150,
    )
    plt.close()

# ==========================================================
# Save classification catalog
# ==========================================================

df = pd.DataFrame(catalog)

df.to_csv(
    OUTPUT_DIR / "classification_catalog.csv",
    index=False,
)

# with open(
#     OUTPUT_DIR / "classification_catalog.json",
#     "w",
# ) as f:
#     json.dump(
#         catalog,
#         f,
#         indent=2,
#     )
df.to_json(
    OUTPUT_DIR / "classification_catalog.json", 
    orient="records", 
    indent=2
)

# ==========================================================
# Create overview plots
# ==========================================================

print("Creating overview plots...")

for class_label, dirname in CLASS_INFO.items():

    members = class_members[class_label]

    plt.figure(figsize=(10, 7))

    for tr in members:

        tau = (
            np.array(tr["times"])
            - tr["times"][0]
        )

        plt.plot(
            tau,
            tr["zs"],
            alpha=0.30,
            lw=1.2,
        )

    plt.xlabel(
        r"Time since birth $\tau$ (s)"
    )
    plt.ylabel(
        "Apex height z (Mm)"
    )

    titles = {
        "A": "Class A: complete parabolas",
        "B": "Class B: partial parabolas",
        "C": "Class C: endpoint artifacts",
        "D": "Class D: kinky / imperfect parabolas",
        "E": "Class E: rejected trajectories",
    }

    plt.title(
        f"{titles[class_label]}\n"
        f"({len(members)} tracks)"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / dirname
        / "overview.png",
        dpi=200,
    )

    plt.close()

# ==========================================================
# Summary
# ==========================================================

summary_lines = []

summary_lines.append(
    "========== PARABOLA CLASSIFICATION ==========\n"
)

summary_lines.append(
    f"Total tracks : {len(df)}\n"
)

for cls in ["A", "B", "C", "D", "E"]:

    n = (df["class"] == cls).sum()

    summary_lines.append(
        f"Class {cls} : {n:4d} tracks"
    )

summary_lines.append("\n")

summary_lines.append(
    "Mean fit quality:\n"
)

summary_lines.append(
    f"Mean R²     : {df['r2'].mean():.4f}"
)

summary_lines.append(
    f"Mean NRMSE  : {df['normalized_rmse'].mean():.4f}"
)

summary_lines.append(
    f"Mean turns  : {df['turning_points'].mean():.2f}"
)

summary_text = "\n".join(summary_lines)

print(summary_text)

with open(
    OUTPUT_DIR / "summary.txt",
    "w",
) as f:
    f.write(summary_text)

print()
print(
    f"Saved outputs to: {OUTPUT_DIR}"
)