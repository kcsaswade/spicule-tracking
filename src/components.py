# spicule_tracking/components.py

import numpy as np
from skimage.measure import label, regionprops
from collections import defaultdict


def extract_components(mask, connectivity=2):
    """
    Label connected components in a binary mask.

    Parameters
    ----------
    mask : bool ndarray
    connectivity : int
        1 -> 4-connectivity
        2 -> 8-connectivity (recommended)

    Returns
    -------
    labels : int ndarray
        Label image.
    props : list
        List of skimage RegionProperties objects.
    """

    labels = label(
        mask,
        connectivity=connectivity,
    )

    props = regionprops(labels)

    return labels, props


def component_summary(prop):
    """
    Extract useful properties from a RegionProperties object.
    """

    min_row, min_col, max_row, max_col = prop.bbox

    return {
        "label": prop.label,
        "area": prop.area,
        "bbox": prop.bbox,
        "min_z_idx": min_row,
        "max_z_idx": max_row - 1,
        "min_x_idx": min_col,
        "max_x_idx": max_col - 1,
        "height_pixels": max_row - min_row,
        "width_pixels": max_col - min_col,
    }


def find_apex_pixel(prop):
    """
    Candidate apex = highest pixel in the component.

    Since image coordinates increase upward in z,
    the apex corresponds to the largest row index.
    """

    coords = prop.coords  # (row, col)

    apex_idx = np.argmax(coords[:, 0])

    apex_row, apex_col = coords[apex_idx]

    return apex_row, apex_col


def find_body_apex(
    prop,
    width_fraction=0.25,
):
    """
    Estimate the apex as the highest row where the
    component still retains a significant fraction
    of its maximum width.
    """

    coords = prop.coords

    # Count number of pixels in each row
    row_dict = defaultdict(list)

    for row, col in coords:
        row_dict[row].append(col)

    rows = sorted(row_dict.keys())

    row_widths = {
        row: len(cols)
        for row, cols in row_dict.items()
    }

    max_width = max(row_widths.values())

    threshold_width = width_fraction * max_width

    # Walk downward from the top of the component
    for row in reversed(rows):

        if row_widths[row] >= threshold_width:

            cols = row_dict[row]
            apex_col = int(np.round(np.mean(cols)))

            return row, apex_col

    # Fallback
    apex_row, apex_col = coords[np.argmax(coords[:, 0])]
    return apex_row, apex_col


def is_candidate_spicule(
    prop,
    z_coords,
    min_height_pixels=20,
    min_apex_z=5.0,
    max_apex_z=15.0,
    min_aspect_ratio=2.0,
):
    """
    Physics-based filtering of connected components.
    """

    min_row, min_col, max_row, max_col = prop.bbox

    height = max_row - min_row
    width = max_col - min_col

    aspect_ratio = height / max(width, 1)

    coords = prop.coords
    apex_row = coords[:, 0].max()
    apex_z = z_coords[apex_row]

    if height < min_height_pixels:
        return False

    if aspect_ratio < min_aspect_ratio:
        return False

    if apex_z < min_apex_z:
        return False

    if apex_z > max_apex_z:
        return False

    return True