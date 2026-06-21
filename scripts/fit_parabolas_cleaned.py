"""
Fit quadratic trajectories to reconstructed spicule apex tracks.

Inputs
------
outputs/50G/v2/tracks_full.json
outputs/50G/parabola_classification/classification_catalog.csv

Outputs
-------
outputs/50G/parabola_fits/

    parabola_catalog.csv
    parabola_catalog.json

    acceleration_histogram.png
    initial_velocity_histogram.png
    apex_height_histogram.png
    rmse_histogram.png
    r2_histogram.png

    accepted_fits.png
    rejected_fits.png

    fit_examples/
        track_XXXX.png
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Configuration
# ==========================================================

TRACK_FILE = Path("outputs/50G/v2/tracks_full.json")
CLASS_FILE = Path(
    "outputs/50G/parabola_classification/classification_catalog.csv"
)

OUTPUT_DIR = Path("outputs/50G/parabola_fits")
FIT_EXAMPLE_DIR = OUTPUT_DIR / "fit_examples"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIT_EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

MIN_POINTS = 5

# endpoint trimming threshold
TRIM_SIGMA = 2.5

# robust fit threshold
ROBUST_SIGMA = 2.5
ROBUST_MAX_ITER = 3

# ==========================================================
# Physical units
# ==========================================================

SIM_TIME_TO_SEC = 100.0
MM_TO_KM = 1000.0

# ==========================================================
# Utilities
# ==========================================================

def compute_fit_metrics(t, z, coeff):
    """
    Compute fit, RMSE and R².
    """

    z_fit = np.polyval(coeff, t)

    residual = z - z_fit

    rmse = np.sqrt(np.mean(residual ** 2))

    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((z - np.mean(z)) ** 2)

    if ss_tot < 1e-12:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    return z_fit, residual, rmse, r2


# ----------------------------------------------------------

def direct_fit(t, z):
    """
    Standard quadratic least-squares fit.
    """

    coeff = np.polyfit(t, z, 2)

    z_fit, residual, rmse, r2 = compute_fit_metrics(
        t,
        z,
        coeff,
    )

    return {
        "coeff": coeff,
        "t": t,
        "z": z,
        "z_fit": z_fit,
        "rmse": rmse,
        "r2": r2,
        "n_trimmed": 0,
        "n_outliers_removed": 0,
    }


# ----------------------------------------------------------

def trimmed_fit(t, z):
    """
    Remove endpoint outliers and refit.
    """

    coeff = np.polyfit(t, z, 2)

    z_fit, residual, rmse, r2 = compute_fit_metrics(
        t,
        z,
        coeff,
    )

    keep = np.ones(len(z), dtype=bool)

    if abs(residual[0]) > TRIM_SIGMA * rmse:
        keep[0] = False

    if abs(residual[-1]) > TRIM_SIGMA * rmse:
        keep[-1] = False

    if keep.sum() < MIN_POINTS:
        keep[:] = True

    coeff = np.polyfit(
        t[keep],
        z[keep],
        2,
    )

    z_fit, residual, rmse, r2 = compute_fit_metrics(
        t[keep],
        z[keep],
        coeff,
    )

    return {
        "coeff": coeff,
        "t": t[keep],
        "z": z[keep],
        "z_fit": z_fit,
        "rmse": rmse,
        "r2": r2,
        "n_trimmed": len(z) - keep.sum(),
        "n_outliers_removed": 0,
    }


# ----------------------------------------------------------

def robust_fit(t, z):
    """
    Iterative sigma-clipped quadratic fit.
    """

    mask = np.ones(len(z), dtype=bool)

    for _ in range(ROBUST_MAX_ITER):

        if mask.sum() < MIN_POINTS:
            break

        coeff = np.polyfit(
            t[mask],
            z[mask],
            2,
        )

        z_fit_all = np.polyval(coeff, t)

        residual = z - z_fit_all

        sigma = residual[mask].std()

        if sigma < 1e-10:
            break

        new_mask = np.abs(residual) < ROBUST_SIGMA * sigma

        if np.all(new_mask == mask):
            break

        mask = new_mask

    if mask.sum() < MIN_POINTS:
        mask[:] = True

    coeff = np.polyfit(
        t[mask],
        z[mask],
        2,
    )

    z_fit, residual, rmse, r2 = compute_fit_metrics(
        t[mask],
        z[mask],
        coeff,
    )

    return {
        "coeff": coeff,
        "t": t[mask],
        "z": z[mask],
        "z_fit": z_fit,
        "rmse": rmse,
        "r2": r2,
        "n_trimmed": 0,
        "n_outliers_removed": len(z) - mask.sum(),
    }


# ==========================================================
# Load inputs
# ==========================================================

print("Loading tracks...")

with open(TRACK_FILE, "r") as f:
    tracks = json.load(f)

track_lookup = {
    tr["track_id"]: tr
    for tr in tracks
}

classification = pd.read_csv(CLASS_FILE)

print(f"Loaded {len(track_lookup)} tracks.")
print(
    f"Loaded {len(classification)} classifications.\n"
)

# ==========================================================
# Main fitting loop
# ==========================================================

catalog = []

accepted_examples = []
rejected_examples = []

for _, row in classification.iterrows():

    track_id = int(row["track_id"])
    cls = str(row["class"]).strip().upper()

    tr = track_lookup.get(track_id)

    if tr is None:
        continue

    t = np.asarray(tr["times"], dtype=float)
    z = np.asarray(tr["zs"], dtype=float)

    if len(t) < MIN_POINTS:
        continue

    # ----------------------------------------------------------
    # Physical coordinates
    # ----------------------------------------------------------

    tau = (t - t[0]) * SIM_TIME_TO_SEC      # seconds

    z = z * MM_TO_KM                        # km

    # ------------------------------------------
    # Reject everything except A and B
    # ------------------------------------------

    if cls not in ["A", "B"]:
        rejected_examples.append((tau, z))

        catalog.append(
            {
                "track_id": track_id,
                "class": cls,
                "fit_status": "rejected",
            }
        )

        continue

    try:

        result = direct_fit(tau, z)

        if cls == "A":
            fit_method = "direct"

        else:
            fit_method = "partial"

    except Exception:
        print(
            f"Skipping track {track_id}: fit failed."
        )
        continue

    coeff = result["coeff"]

    A, B, C = coeff

    # ---------------------------------------------------
    # Physical parameters
    # ---------------------------------------------------

    initial_velocity = B            # km/s

    acceleration = 2.0*A            # km/s²

    deceleration = -2.0*A           # km/s²

    # upward-opening parabola
    if acceleration > 0:

        rejected_examples.append((tau, z))

        catalog.append(
            {
                "track_id": track_id,
                "class": cls,
                "fit_status": "rejected",
                "reject_reason": "positive_acceleration"
            }
        )

        continue

    # ---------------------------------------
    # Reject descending trajectories
    # ---------------------------------------

    if initial_velocity < 0:

        rejected_examples.append((tau, z))

        catalog.append(
            {
                "track_id": track_id,
                "class": cls,
                "fit_status": "rejected",
                "reject_reason": "negative_v0"
            }
        )

        continue

    if abs(A) > 1e-14:

        t_apex_sec = -B/(2*A)

        z_apex_km = np.polyval(
            coeff,
            t_apex_sec,
        )

    else:

        t_apex_sec = np.nan
        z_apex_km = np.nan

    z_apex_Mm = z_apex_km / MM_TO_KM
    t_apex = t_apex_sec / SIM_TIME_TO_SEC

    if cls == "B":
        apex_inside = (
            -0.2 <= t_apex <= tau[-1] + 0.2
        )
    else:
        apex_inside = True

    rmse_Mm = result["rmse"] / MM_TO_KM

    catalog.append(
        {
            "track_id": track_id,
            "class": cls,
            "fit_status": "accepted",
            "fit_method": fit_method,
            "n_points": len(z),
            "n_used": len(result["z"]),
            "A": A,
            "B": B,
            "C": C,
            # physical quantities
            "initial_velocity_km_s":
                initial_velocity,

            "acceleration_km_s2":
                acceleration,

            "deceleration_km_s2":
                deceleration,

            "apex_time_sec":
                t_apex_sec,

            "apex_height_Mm":
                z_apex_Mm,

            "rmse_Mm":
                rmse_Mm,
            "r2": result["r2"],
            "apex_inside_window": apex_inside,
            "n_trimmed": result["n_trimmed"],
            "n_outliers_removed":
                result["n_outliers_removed"],
        }
    )

    accepted_examples.append(
        (
            result["t"],
            result["z"],
            result["z_fit"],
        )
    )

    # --------------------------------------------------
    # Save individual figure
    # --------------------------------------------------

    plt.figure(figsize=(6, 4))

    plt.plot(
        result["t"],
        result["z"],
        "o",
        ms=4,
        label="Data",
    )

    t_dense = np.linspace(
        result["t"].min(),
        result["t"].max(),
        200,
    )

    plt.plot(
        t_dense,
        np.polyval(coeff, t_dense),
        lw=2,
        label="Quadratic fit",
    )

    plt.xlabel(
        r"Time since birth $\tau$ (s)"
    )
    plt.ylabel("Apex height z (Mm)")

    plt.title(
        f"Track {track_id}\n"
        f"Class {cls} ({fit_method})\n"
        f"R²={result['r2']:.3f}   "
        f"RMSE={result['rmse']:.3f}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIT_EXAMPLE_DIR
        / f"track_{track_id:04d}.png",
        dpi=120,
    )

    plt.close()

# ==========================================================
# Save catalog
# ==========================================================

catalog_df = pd.DataFrame(catalog)

catalog_df.to_csv(
    OUTPUT_DIR / "parabola_catalog.csv",
    index=False,
)

# with open(
#     OUTPUT_DIR / "parabola_catalog.json",
#     "w",
# ) as f:
#     json.dump(
#         catalog,
#         f,
#         indent=2,
#     )
catalog_df.to_json(
    OUTPUT_DIR / "parabola_catalog.json", 
    orient="records", 
    indent=2
)

# ==========================================================
# Diagnostic histograms
# ==========================================================

accepted = catalog_df[
    catalog_df["fit_status"] == "accepted"
]

histograms = [
    (
        "initial_velocity_km_s",
        "Initial velocity (km/s)",
        "initial_velocity_histogram.png"
    ),
    (
        "acceleration_km_s2",
        "Signed acceleration (km/s²)",
        "acceleration_histogram.png"
    ),
    (
        "deceleration_km_s2",
        "Deceleration magnitude (km/s²)",
        "deceleration_histogram.png"
    ),
    (
        "apex_height_Mm",
        "Fitted apex height (Mm)",
        "apex_height_histogram.png",
    ),
    (
        "rmse_Mm",
        "RMSE (Mm)",
        "rmse_histogram.png"
    ),
    (
        "r2",
        r"$R^2$",
        "r2_histogram.png",
    ),
]

for col, xlabel, fname in histograms:

    plt.figure(figsize=(7, 5))

    plt.hist(
        accepted[col].dropna(),
        bins=20,
        edgecolor="black",
    )

    plt.xlabel(xlabel)
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / fname,
        dpi=150,
    )

    plt.close()

# ==========================================================
# Overview plot: accepted fits
# ==========================================================

plt.figure(figsize=(10, 7))

for t, z, fit in accepted_examples[:30]:
    plt.plot(
        t,
        z,
        alpha=0.4,
        lw=1,
    )

plt.xlabel(
    r"Time since birth $\tau$ (s)"
)
plt.ylabel("Apex height z (Mm)")
plt.title("Accepted fitted trajectories")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "accepted_fits.png",
    dpi=150,
)

plt.close()

# ==========================================================
# Overview plot: rejected fits
# ==========================================================

plt.figure(figsize=(10, 7))

for t, z in rejected_examples[:30]:
    plt.plot(
        t,
        z,
        alpha=0.5,
        lw=1,
    )

plt.xlabel(
    r"Time since birth $\tau$ (s)"
)
plt.ylabel("Apex height z (Mm)")
plt.title("Rejected trajectories")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rejected_fits.png",
    dpi=150,
)

plt.close()

# ==========================================================
# Summary
# ==========================================================

print("========== PARABOLA FIT SUMMARY ==========\n")

print(
    f"Accepted fits : "
    f"{(catalog_df['fit_status']=='accepted').sum()}"
)

print(
    f"Rejected      : "
    f"{(catalog_df['fit_status']=='rejected').sum()}"
)

print()

if len(accepted) > 0:

    print(
        f"Mean R²     : "
        f"{accepted['r2'].mean():.4f}"
    )

    print(
        f"Mean RMSE   : "
        f"{accepted['rmse_Mm'].mean():.4f}"
    )

    print(
        f"Mean v0     : "
        f"{accepted['initial_velocity_km_s'].mean():.4f}"
    )

    print(
    f"Mean acceleration : "
    f"{accepted['acceleration_km_s2'].mean():.4f}"
    )

    print(
        f"Mean deceleration : "
        f"{accepted['deceleration_km_s2'].mean():.4f}"
    )

print()
print(
    f"Saved results to: {OUTPUT_DIR}"
)