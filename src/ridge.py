# spicule_tracking/ridge.py

import numpy as np
from skimage.filters import frangi, sato


def frangi_ridge(
    image,
    sigmas=(1, 2, 3),
    black_ridges=False,
):
    """
    Multi-scale Frangi vesselness filter.

    Parameters
    ----------
    image : ndarray
        Normalized anomaly image.
    sigmas : iterable
        Spatial scales to probe.
    black_ridges : bool
        False -> bright ridges on dark background.
    """
    return frangi(
        image,
        sigmas=sigmas,
        black_ridges=black_ridges,
    )


def sato_ridge(
    image,
    sigmas=(1, 2, 3, 4),
    black_ridges=False,
):
    """
    Multi-scale Sato tubeness filter.
    """
    return sato(
        image,
        sigmas=sigmas,
        black_ridges=black_ridges,
    )

def compute_ridge_image(image, sigmas=(1, 2, 3, 4)):
    positive = np.clip(image, 0.0, None)

    return sato(
        positive,
        sigmas=sigmas,
        black_ridges=False,
    )

# spicule_tracking/ridge.py

def prepare_for_ridge(image):
    """
    Keep only positive thermal anomalies before ridge filtering.
    """
    return np.clip(image, 0.0, None)