from src.io import SpiculeDataset
from src.normalize import (
    BackgroundNormalizer,
    preprocess_frame,
)
from src.segment import (
    threshold_anomaly,
    clean_mask,
)

import matplotlib.pyplot as plt

DATAFILE = "data/raw/spicules_temperature_final.h5"
BACKGROUND_FILE = "data/processed/background_profile.npz"


def main():

    ds = SpiculeDataset(DATAFILE)

    normalizer = BackgroundNormalizer()
    normalizer.load(BACKGROUND_FILE)

    frame = 100

    img = preprocess_frame(
        ds,
        frame_idx=frame,
        normalizer=normalizer,
    )

    threshold = -0.4

    mask = threshold_anomaly(
        img,
        threshold=threshold,
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

    # Normalized anomaly
    im = axes[0].imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-4,
        vmax=4,
    )
    axes[0].set_title("Normalized anomaly")

    # Raw threshold
    axes[1].imshow(
        mask,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
    )
    axes[1].set_title(
        f"Threshold (A ≤ {threshold})"
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
        img,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
        vmin=-4,
        vmax=4,
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
    )

    plt.show()


if __name__ == "__main__":
    main()