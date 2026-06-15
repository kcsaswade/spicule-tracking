# src/envelope.py

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def extract_upper_envelope(mask, z):
    """
    Compute the upper envelope of a binary segmentation mask.

    Parameters
    ----------
    mask : (nz, nx) bool array
        Segmented spicule canopy.
    z : (nz,) array
        Physical z coordinates.

    Returns
    -------
    envelope : (nx,) array
        Highest segmented z-value for each x-column.
        Columns containing no segmented pixels are set to NaN.
    """

    nz, nx = mask.shape
    envelope = np.full(nx, np.nan)

    for j in range(nx):
        rows = np.where(mask[:, j])[0]

        if len(rows) > 0:
            envelope[j] = z[rows.max()]

    return envelope


def smooth_envelope(envelope, sigma=2.0):
    """
    Smooth the upper envelope while respecting NaNs.
    """

    valid = np.isfinite(envelope)

    if valid.sum() == 0:
        return envelope.copy()

    filled = envelope.copy()
    filled[~valid] = np.interp(
        np.flatnonzero(~valid),
        np.flatnonzero(valid),
        envelope[valid]
    )

    smoothed = gaussian_filter1d(
        filled,
        sigma=sigma,
        mode="nearest"
    )

    smoothed[~valid] = np.nan

    return smoothed


def detect_tip_peaks(
    envelope,
    prominence=0.4,
    distance=15,
):
    """
    Detect local maxima of the smoothed envelope.

    Parameters
    ----------
    prominence : float
        Required peak prominence (Mm).
    distance : int
        Minimum separation between peaks (pixels).

    Returns
    -------
    peak_indices : ndarray
        Column indices of detected peaks.
    """

    valid = np.isfinite(envelope)

    if valid.sum() == 0:
        return np.array([], dtype=int)

    work = envelope.copy()
    work[~valid] = -np.inf

    peaks, _ = find_peaks(
        work,
        prominence=prominence,
        distance=distance,
    )

    return peaks