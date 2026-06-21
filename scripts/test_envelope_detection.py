from src.io import SpiculeDataset
from src.bandpass import (
    prepare_logT_frame,
    bandpass_mask,
)
from src.segment import clean_mask
from src.envelope import (
    extract_upper_envelope,
    smooth_envelope,
    detect_tip_peaks,
)

import matplotlib.pyplot as plt
import numpy as np

DATAFILE = "data/raw/spicules_temperature_final_50G.h5"


def main():

    ds = SpiculeDataset(DATAFILE)

    frame = 100

    logT = prepare_logT_frame(ds, frame)

    mask = bandpass_mask(
        logT,
        ds.z,
        Tmin=3.9,
        Tmax=4.9,
        zmin=0.0,
        zmax=15.0,
    )

    mask = clean_mask(
        mask,
        min_size=20,
        closing_radius=1,
    )

    envelope = extract_upper_envelope(
        mask,
        ds.z,
    )

    smooth = smooth_envelope(
        envelope,
        sigma=2.0,
    )

    peaks = detect_tip_peaks(
        smooth,
        prominence=0.5,
        distance=15,
    )

    extent = [
        ds.x.min(),
        ds.x.max(),
        ds.z.min(),
        ds.z.max(),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 8),
        constrained_layout=True,
    )

    # -----------------------------------------------------------------
    # Panel 1: Original movie-like image
    # -----------------------------------------------------------------

    im = axes[0].imshow(
        logT,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gist_rainbow",
        vmin=3.57,
        vmax=6.00,
    )

    axes[0].set_title("Global log10(T)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")

    # -----------------------------------------------------------------
    # Panel 2: Canopy overlay + envelope
    # -----------------------------------------------------------------

    axes[1].imshow(
        logT,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
        vmin=3.57,
        vmax=6.00,
    )

    axes[1].contour(
        mask.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=0.8,
        origin="lower",
        extent=extent,
    )

    valid = np.isfinite(smooth)

    axes[1].plot(
        ds.x[valid],
        smooth[valid],
        color="cyan",
        linewidth=2,
        label="Upper envelope",
    )

    axes[1].set_title("Canopy + upper envelope")
    axes[1].set_xlabel("x")

    # -----------------------------------------------------------------
    # Panel 3: Detected tip peaks
    # -----------------------------------------------------------------

    axes[2].imshow(
        logT,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
        vmin=3.57,
        vmax=6.00,
    )

    axes[2].plot(
        ds.x[valid],
        smooth[valid],
        color="cyan",
        linewidth=2,
    )

    axes[2].plot(
        ds.x[peaks],
        smooth[peaks],
        "ro",
        markersize=6,
    )

    axes[2].set_title(
        f"Detected tips ({len(peaks)})"
    )
    axes[2].set_xlabel("x")

    fig.colorbar(
        im,
        ax=axes[0],
        shrink=0.8,
        label=r"$\log_{10}T$",
    )

    plt.show()

    print()
    print("Detected peak count:", len(peaks))
    print("Peak x positions (Mm):")
    print(ds.x[peaks])


if __name__ == "__main__":
    main()