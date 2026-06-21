from src.tracker_v2 import (
    load_detections,
    run_tracker,
)

import matplotlib.pyplot as plt
import numpy as np


JSON_FILE = "outputs/10G/envelope_detections.json"


def main():

    detections = load_detections(JSON_FILE)

    tracks = run_tracker(detections)

    print()
    print(f"Total tracks: {len(tracks)}")

    # lengths = [len(t.frames) for t in tracks]
    lengths = np.array([len(t.frames) for t in tracks])

    print(f"Total tracks           : {len(tracks)}")
    print(f"Mean lifetime (frames) : {lengths.mean():.2f}")
    print(f"Median lifetime        : {np.median(lengths):.2f}")
    print(f"Min lifetime           : {lengths.min()}")
    print(f"Max lifetime           : {lengths.max()}")

    print()
    print("Tracks with:")
    print(f"  >=  5 frames : {(lengths >= 5).sum()}")
    print(f"  >= 10 frames : {(lengths >= 10).sum()}")
    print(f"  >= 20 frames : {(lengths >= 20).sum()}")
    print(f"  >= 50 frames : {(lengths >= 50).sum()}")

    print(
        f"Mean lifetime (frames): "
        f"{np.mean(lengths):.2f}"
    )

    fig, ax = plt.subplots(
        figsize=(12, 8),
        constrained_layout=True,
    )

    #cmap = plt.cm.get_cmap("tab20", len(tracks))
    cmap = plt.get_cmap("tab20", len(tracks))

    for i, tr in enumerate(tracks):

        if len(tr.frames) < 3:
            continue

        ax.plot(
            tr.xs,
            tr.zs,
            "-o",
            markersize=2,
            linewidth=1.5,
            color=cmap(i),
            alpha=0.8,
        )

    ax.set_xlabel("x (Mm)")
    ax.set_ylabel("Apex height z (Mm)")
    ax.set_title("Reconstructed spicule trajectories")

    plt.show()


if __name__ == "__main__":
    main()