import numpy as np


def prepare_logT_frame(dataset, frame_idx,
                       clip_min=3.57,
                       clip_max=6.00):
    """
    Produce the same globally-scaled log10(T) field
    used in the original Pencil Code contour plots.
    """
    T = dataset.get_frame(frame_idx)
    logT = np.log10(T)

    return np.clip(logT, clip_min, clip_max)


def bandpass_mask(
    logT,
    z,
    Tmin,
    Tmax,
    zmin=0.0,
    zmax=15.0,
):
    """
    Segment a temperature interval from the globally
    clipped log10(T) field.

    Parameters
    ----------
    Tmin, Tmax :
        Temperature interval to retain.
    zmin, zmax :
        Physical height limits (Mm).
    """

    mask = (logT >= Tmin) & (logT <= Tmax)

    # remove unwanted height regions
    zmask = (z >= zmin) & (z <= zmax)
    mask &= zmask[:, None]

    return mask