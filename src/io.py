# src/io.py

import h5py
import numpy as np


class SpiculeDataset:
    """
    Wrapper around the HDF5 temperature cube.
    """

    def __init__(self, filename):
        self.filename = filename

        with h5py.File(filename, "r") as f:
            self.x = f["x"][...]
            self.z = f["z"][...]
            self.time = f["time"][...]
            self.ivar = f["ivar"][...]
            self.shape = f["temperature"].shape

    def get_frame(self, frame_idx):
        with h5py.File(self.filename, "r") as f:
            return f["temperature"][frame_idx]

    @property
    def n_frames(self):
        return self.shape[0]