from src.io import SpiculeDataset
from src.normalize import (
    BackgroundNormalizer,
    preprocess_frame,
)
from src.ridge import compute_ridge_image
from src.segment import (
    threshold_ridge_image,
    clean_binary_mask,
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

    ridge = compute_ridge_image(img)

    # Initial guess; we'll tune this if necessary.
    threshold = 0.2

    mask = threshold_ridge_image(
        ridge,
        threshold=threshold,
    )

    cleaned = clean_binary_mask(
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
        5,
        figsize=(20, 8),
        constrained_layout=True,
    )

    im0 = axes[0].imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-4,
        vmax=4,
    )
    axes[0].set_title("Normalized anomaly")

    im1 = axes[1].imshow(
        ridge,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    axes[1].set_title("Sato ridge response")

    axes[2].imshow(
        mask,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
    )
    axes[2].set_title(
        f"Thresholded (τ={threshold})"
    )

    axes[3].imshow(
        cleaned,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
    )
    axes[3].set_title("Cleaned mask")

    # Overlay: normalized anomaly + cleaned segmentation mask

    axes[4].imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
        vmin=-4,
        vmax=4,
    )

    axes[4].contour(
        cleaned.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=0.8,
        origin="lower",
        extent=extent,
    )

    axes[4].set_title("Segmentation overlay")
    axes[4].set_xlabel("x")

    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("z")

    fig.colorbar(
        im0,
        ax=axes[0],
        shrink=0.8,
    )
    fig.colorbar(
        im1,
        ax=axes[1],
        shrink=0.8,
    )

    plt.show()


if __name__ == "__main__":
    main()