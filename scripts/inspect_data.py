# scripts/inspect_data.py

from src.io import SpiculeDataset
from src.visualize import plot_frame, animate_frames

DATAFILE = "data/raw/spicules_temperature_final_50G.h5"


def main():
    ds = SpiculeDataset(DATAFILE)

    print()
    print("=== Dataset Summary ===")
    print(f"Frames     : {ds.n_frames}")
    print(f"Shape      : {ds.shape}")
    print(f"x range    : {ds.x.min():.3f} -> {ds.x.max():.3f}")
    print(f"z range    : {ds.z.min():.3f} -> {ds.z.max():.3f}")
    print(f"time range : {ds.time[0]:.3f} -> {ds.time[-1]:.3f}")
    print()

    plot_frame(ds, frame_idx=0, logscale=True)
    plot_frame(ds, frame_idx=100, logscale=True)

    # Uncomment when needed:
    # anim = animate_frames(
    #     ds,
    #     logscale=True,
    #     save_path="outputs/50G/animations/raw_temperature.mp4"
    # )

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()