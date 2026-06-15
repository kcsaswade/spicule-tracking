from src.io import SpiculeDataset
from src.normalize import (
    BackgroundNormalizer,
    preprocess_frame,
)
from src.ridge import (
    frangi_ridge,
    sato_ridge,
)

import matplotlib.pyplot as plt
import numpy as np

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

    frangi_img = frangi_ridge(img)
    sato_img = sato_ridge(img)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 8),
        constrained_layout=True,
    )

    extent = [
        ds.x.min(),
        ds.x.max(),
        ds.z.min(),
        ds.z.max(),
    ]

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
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")

    # im1 = axes[1].imshow(
    #     frangi_img,
    #     origin="lower",
    #     aspect="auto",
    #     extent=extent,
    #     cmap="viridis",
    # )
    # axes[1].set_title("Frangi ridge response")
    # axes[1].set_xlabel("x")

    im2 = axes[2].imshow(
        sato_img,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    axes[2].set_title("Sato ridge response")
    axes[2].set_xlabel("x")

    alpha = 0.5

    axes[3].imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
    )

    axes[3].imshow(
        sato_img,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        alpha=alpha,
    )

    axes[3].set_title("Sato overlay")

    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    # fig.colorbar(im1, ax=axes[1], shrink=0.8)
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.show()


if __name__ == "__main__":
    main()