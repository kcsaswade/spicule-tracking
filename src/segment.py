# # spicule_tracking/segment.py

# import numpy as np
# from skimage.morphology import (
#     remove_small_objects,
#     binary_closing,
#     disk,
# )


# def threshold_ridge_image(
#     ridge_image,
#     threshold,
# ):
#     """
#     Convert a ridge-response image into a binary mask.
#     """
#     return ridge_image >= threshold


# def clean_binary_mask(
#     mask,
#     min_size=20,
#     closing_radius=1,
# ):
#     """
#     Simple morphological cleanup.

#     Parameters
#     ----------
#     mask : bool ndarray
#     min_size : int
#         Remove connected components smaller than this.
#     closing_radius : int
#         Radius of binary closing operation.
#     """

#     cleaned = remove_small_objects(
#         mask,
#         min_size=min_size,
#     )

#     if closing_radius > 0:
#         cleaned = binary_closing(
#             cleaned,
#             footprint=disk(closing_radius),
#         )

#     return cleaned

# spicule_tracking/segment.py

import numpy as np
from skimage.morphology import (
    binary_closing,
    remove_small_objects,
    disk,
)


def threshold_anomaly(
    image,
    threshold=-0.8,
):
    """
    Segment cold spicule bodies directly from the
    normalized anomaly image.

    Pixels with values <= threshold are considered
    part of a candidate spicule.
    """
    return image <= threshold

def apply_height_mask(mask, z, z_min=0.5):
    """
    Remove all segmented pixels below a physical height.
    """
    mask = mask.copy()
    mask[z[:, None] < z_min] = False
    return mask


def clean_mask(
    mask,
    min_size=20,
    closing_radius=1,
):
    """
    Morphological cleanup.

    1. Bridge tiny gaps.
    2. Remove tiny isolated objects.
    """

    if closing_radius > 0:
        mask = binary_closing(
            mask,
            footprint=disk(closing_radius),
        )

    mask = remove_small_objects(
        mask,
        min_size=min_size,
    )

    return mask