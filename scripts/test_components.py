from src.io import SpiculeDataset
from src.normalize import (
    BackgroundNormalizer,
    preprocess_frame,
)
from src.segment import (
    threshold_anomaly,
    clean_mask,
)
from src.components import (
    extract_components,
    component_summary,
    find_apex_pixel,
    is_candidate_spicule,
    find_body_apex,
)

import matplotlib.pyplot as plt
import numpy as np

DATAFILE = "data/raw/spicules_temperature_final_50G.h5"
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

    # Current best segmentation
    mask = threshold_anomaly(
        img,
        threshold=-0.5,
    )

    mask = clean_mask(
        mask,
        min_size=20,
        closing_radius=1,
    )
    # Remove everything below z = 0
    mask[ds.z < 0.0, :] = False

    labels, props = extract_components(mask)

    extent = [
        ds.x.min(),
        ds.x.max(),
        ds.z.min(),
        ds.z.max(),
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(24, 8),
        constrained_layout=True,
    )

    # Original normalized anomaly
    axes[0].imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-4,
        vmax=4,
    )
    axes[0].set_title("Normalized anomaly")

    # Label image
    axes[1].imshow(
        labels,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="nipy_spectral",
    )
    axes[1].set_title(
        f"Connected components ({len(props)})"
    )

    # Overlay with apexes
    axes[3].imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
        vmin=-4,
        vmax=4,
    )
    axes[3].set_title("Filtered body apexes")

    filtered_props = []

    for prop in props:

        if not is_candidate_spicule(
            prop,
            ds.z,
            min_height_pixels=20,
            min_apex_z=5.0,
            max_apex_z=15.0,
            min_aspect_ratio=2.0,
        ):
            continue

        filtered_props.append(prop)

        apex_row, apex_col = find_body_apex(
            prop,
            width_fraction=0.25,
        )

        apex_x = ds.x[apex_col]
        apex_z = ds.z[apex_row]

        axes[3].plot(
            apex_x,
            apex_z,
            "ro",
            markersize=4,
        )

    print(f"Total components   : {len(props)}")
    print(f"Filtered candidates: {len(filtered_props)}")

    filtered_mask = np.zeros_like(labels, dtype=bool)

    for prop in filtered_props:
        rr, cc = prop.coords[:, 0], prop.coords[:, 1]
        filtered_mask[rr, cc] = True

    axes[2].imshow(
        filtered_mask,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
    )
    axes[2].set_title(
        f"Filtered candidates ({len(filtered_props)})"
    )

    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("z")

    print("\nFirst few accepted candidates:")
    for prop in filtered_props[:10]:
        min_row, min_col, max_row, max_col = prop.bbox
        h = max_row - min_row
        w = max_col - min_col
        ar = h / max(w, 1)

        apex_row, apex_col = find_body_apex(prop)
        apex_z = ds.z[apex_row]

        print(
            f"ID={prop.label:3d} | "
            f"area={prop.area:4f} | "
            f"h={h:3d} | w={w:3d} | "
            f"AR={ar:4.2f} | "
            f"apex_z={apex_z:5.2f}"
        )

    plt.show()



if __name__ == "__main__":
    main()