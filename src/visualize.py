# src/visualize.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_frame(dataset,
               frame_idx,
               logscale=True,
               ax=None,
               cmap="magma",
               title=True,
               colorbar=True):
    """
    Plot a single simulation frame.

    Parameters
    ----------
    dataset : SpiculeDataset
    frame_idx : int
    logscale : bool
        If True, display log10(T).
    ax : matplotlib axis or None
    """

    T = dataset.get_frame(frame_idx)

    if logscale:
        image = np.log10(T)
        cbar_label = "log10(Temperature)"
    else:
        image = T
        cbar_label = "Temperature"

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = ax.figure

    im = ax.imshow(
        image,
        origin="lower",
        aspect="auto",
        extent=[
            dataset.x.min(),
            dataset.x.max(),
            dataset.z.min(),
            dataset.z.max(),
        ],
        cmap=cmap,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("z")

    if title:
        t = dataset.time[frame_idx]
        ax.set_title(f"Frame {frame_idx}   t={t:.2f}")

    if colorbar:
        fig.colorbar(im, ax=ax, label=cbar_label)

    return fig, ax, im

def animate_frames(dataset,
                   logscale=True,
                   interval=100,
                   save_path=None):
    """
    Create an animation of the temperature field.
    """

    fig, ax = plt.subplots(figsize=(6, 8))

    first = dataset.get_frame(0)
    if logscale:
        first = np.log10(first)

    im = ax.imshow(
        first,
        origin="lower",
        aspect="auto",
        extent=[
            dataset.x.min(),
            dataset.x.max(),
            dataset.z.min(),
            dataset.z.max(),
        ],
        cmap="magma",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("z")

    def update(frame):
        img = dataset.get_frame(frame)
        if logscale:
            img = np.log10(img)

        im.set_data(img)
        ax.set_title(
            f"Frame {frame}    t={dataset.time[frame]:.2f}"
        )
        return [im]

    anim = FuncAnimation(
        fig,
        update,
        frames=dataset.n_frames,
        interval=interval,
        blit=False,
    )

    if save_path is not None:
        anim.save(save_path, dpi=150)

    return anim

def plot_envelope_detection(
    ax,
    logT,
    x,
    z,
    mask,
    envelope,
    peaks,
):
    """
    Draw one annotated frame showing:
      - movie-style log10(T)
      - canopy contour
      - upper envelope
      - detected apexes
    """

    extent = [x.min(), x.max(), z.min(), z.max()]

    ax.imshow(
        logT,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gist_rainbow",
        vmin=3.57,
        vmax=6.00,
    )

    ax.contour(
        mask.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=0.6,
        origin="lower",
        extent=extent,
    )

    valid = np.isfinite(envelope)

    ax.plot(
        x[valid],
        envelope[valid],
        color="cyan",
        linewidth=2,
    )

    if len(peaks) > 0:
        ax.plot(
            x[peaks],
            envelope[peaks],
            "ro",
            markersize=5,
        )

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(z.min(), z.max())
    ax.set_xlabel("x (Mm)")
    ax.set_ylabel("z (Mm)")