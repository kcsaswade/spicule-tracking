from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
from src.visualize import plot_envelope_detection
import json


DATAFILE = "data/raw/spicules_temperature_final.h5"

# ---- Segmentation parameters ----
TMIN = 3.9
TMAX = 4.9
ZMIN = 0.0
ZMAX = 15.0

# ---- Envelope parameters ----
SMOOTH_SIGMA = 2.0

# ---- Peak detection parameters ----
PEAK_PROMINENCE = 0.5
PEAK_DISTANCE = 15

OUTPUT_DIR = Path("outputs/envelope_frames")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = SpiculeDataset(DATAFILE)

    peak_counts = []
    all_detections = []

    for frame in range(ds.n_frames):

        logT = prepare_logT_frame(ds, frame)

        mask = bandpass_mask(
            logT,
            ds.z,
            Tmin=TMIN,
            Tmax=TMAX,
            zmin=ZMIN,
            zmax=ZMAX,
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

        envelope = smooth_envelope(
            envelope,
            sigma=SMOOTH_SIGMA,
        )

        peaks = detect_tip_peaks(
            envelope,
            prominence=PEAK_PROMINENCE,
            distance=PEAK_DISTANCE,
        )

        peak_counts.append(len(peaks))
        # Save detections for later tracking
        for k, peak_idx in enumerate(peaks):
            all_detections.append({
                "frame": int(frame),
                "time": float(ds.time[frame]),
                "detection_id": int(k),          # index within this frame
                "x": float(ds.x[peak_idx]),
                "z": float(envelope[peak_idx]),
            })

        fig, ax = plt.subplots(
            figsize=(8, 10),
            constrained_layout=True,
        )

        plot_envelope_detection(
            ax=ax,
            logT=logT,
            x=ds.x,
            z=ds.z,
            mask=mask,
            envelope=envelope,
            peaks=peaks,
        )

        sim_time = ds.time[frame]

        ax.set_title(
            f"Frame {frame:03d}   "
            f"t = {sim_time:.2f}   "
            f"Tips = {len(peaks)}"
        )

        outfile = OUTPUT_DIR / f"frame_{frame:03d}.png"

        plt.savefig(
            outfile,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"[{frame+1:3d}/{ds.n_frames}] "
            f"tips = {len(peaks):2d}"
        )

    print()
    print("Finished.")
    # Save all detections to JSON
    with open("outputs/envelope_detections.json", "w") as f:
        json.dump(all_detections, f, indent=2)

    print("Saved detections to outputs/envelope_detections.json")
    print(
        f"Mean detected tips/frame: "
        f"{np.mean(peak_counts):.2f}"
    )
    print(
        f"Min/Max tips/frame: "
        f"{np.min(peak_counts)} / {np.max(peak_counts)}"
    )


if __name__ == "__main__":
    main()