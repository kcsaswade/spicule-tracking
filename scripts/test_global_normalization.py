from src.io import SpiculeDataset
from src.normalize import (
    log_temperature,
    BackgroundNormalizer,
)

import matplotlib.pyplot as plt

DATAFILE = "data/raw/spicules_temperature_final_50G.h5"


def main():
    ds = SpiculeDataset(DATAFILE)

    # Build global background model
    normalizer = BackgroundNormalizer()
    normalizer.fit(ds)
    normalizer.save("data/processed/background_profile.npz")

    # Test on one representative frame
    frame = 100

    T = ds.get_frame(frame)
    logT = log_temperature(T)
    normalized = normalizer.transform(logT)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 8),
        constrained_layout=True,
    )

    # Original log(T)
    im0 = axes[0].imshow(
        logT,
        origin="lower",
        aspect="auto",
        extent=[
            ds.x.min(),
            ds.x.max(),
            ds.z.min(),
            ds.z.max(),
        ],
        cmap="magma",
    )
    axes[0].set_title("log10(T)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")

    # Globally normalized image
    im1 = axes[1].imshow(
        normalized,
        origin="lower",
        aspect="auto",
        extent=[
            ds.x.min(),
            ds.x.max(),
            ds.z.min(),
            ds.z.max(),
        ],
        cmap="coolwarm",
        vmin=-4,
        vmax=4,  # fixed limits help compare frames
    )
    axes[1].set_title("Global normalized anomaly")
    axes[1].set_xlabel("x")

    # Learned background profile
    axes[2].plot(
        normalizer.mean_profile,
        ds.z,
        label="Global mean",
    )
    axes[2].fill_betweenx(
        ds.z,
        normalizer.mean_profile - normalizer.std_profile,
        normalizer.mean_profile + normalizer.std_profile,
        alpha=0.3,
        label=r"$\pm1\sigma$",
    )

    axes[2].set_title("Global background profile")
    axes[2].set_xlabel(r"$\log_{10}(T)$")
    axes[2].set_ylabel("z")
    axes[2].legend()

    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.show()


if __name__ == "__main__":
    main()