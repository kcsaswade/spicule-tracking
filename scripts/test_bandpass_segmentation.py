from src.io import SpiculeDataset
from src.bandpass import (
    prepare_logT_frame,
    bandpass_mask,
)
from src.segment import clean_mask

import matplotlib.pyplot as plt

DATAFILE = "data/raw/spicules_temperature_final.h5"


def main():

    ds = SpiculeDataset(DATAFILE)

    frame = 100

    logT = prepare_logT_frame(ds, frame)

    # -------- PARAMETERS TO EXPLORE --------

    Tmin = 3.9
    Tmax = 4.9
    mask = bandpass_mask(
        logT,
        ds.z,
        Tmin=Tmin,
        Tmax=Tmax,
        zmin=0.0,
        zmax=15.0,
    )

    cleaned = clean_mask(
        mask,
        min_size=20,
        closing_radius=1,
    )

    extent = [
        ds.x.min(),
        ds.x.max(),
        ds.z.min(),
        ds.z.max(),
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 8),
        constrained_layout=True,
    )

    # Original movie-like representation
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

    # Raw bandpass mask
    axes[1].imshow(
        mask,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
    )
    axes[1].set_title(
        f"{Tmin:.1f} ≤ log10(T) ≤ {Tmax:.1f}"
    )

    # Cleaned mask
    axes[2].imshow(
        cleaned,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
    )
    axes[2].set_title("Cleaned mask")

    # Overlay
    axes[3].imshow(
        logT,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
        vmin=3.57,
        vmax=6.00,
    )

    axes[3].contour(
        cleaned.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=0.8,
        origin="lower",
        extent=extent,
    )

    axes[3].set_title("Segmentation overlay")

    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("z")

    fig.colorbar(
        im,
        ax=axes[0],
        shrink=0.8,
        label=r"$\log_{10} T$",
    )

    plt.show()


if __name__ == "__main__":
    main()