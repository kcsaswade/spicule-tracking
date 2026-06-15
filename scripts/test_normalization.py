from src.io import SpiculeDataset
from src.normalize import (
    log_temperature,
    normalize_frame,
)

import matplotlib.pyplot as plt

DATAFILE = "data/raw/spicules_temperature_final.h5"

 
def main():
    ds = SpiculeDataset(DATAFILE)

    frame = 100

    T = ds.get_frame(frame)
    logT = log_temperature(T)

    normalized, mean_profile, std_profile = normalize_frame(logT)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 8),
        constrained_layout=True
    )

    # Original
    im0 = axes[0].imshow(
        logT,
        origin="lower",
        aspect="auto",
        extent=[
            ds.x.min(), ds.x.max(),
            ds.z.min(), ds.z.max()
        ],
        cmap="magma",
    )
    axes[0].set_title("log10(T)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")

    # Normalized
    im1 = axes[1].imshow(
        normalized,
        origin="lower",
        aspect="auto",
        extent=[
            ds.x.min(), ds.x.max(),
            ds.z.min(), ds.z.max()
        ],
        cmap="coolwarm",
    )
    axes[1].set_title("Normalized anomaly")
    axes[1].set_xlabel("x")

    # Vertical profile
    axes[2].plot(mean_profile, ds.z, label="Mean")
    axes[2].fill_betweenx(
        ds.z,
        mean_profile - std_profile,
        mean_profile + std_profile,
        alpha=0.3,
        label=r"$\pm1\sigma$"
    )
    axes[2].set_xlabel(r"$\log_{10}(T)$")
    axes[2].set_ylabel("z")
    axes[2].set_title("Vertical background profile")
    axes[2].legend()

    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.show()


if __name__ == "__main__":
    main()