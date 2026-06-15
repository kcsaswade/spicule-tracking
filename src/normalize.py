# spicule_tracking/normalize.py

import numpy as np


def log_temperature(T):
    """Safely compute log10(T)."""
    return np.log10(T)

def preprocess_frame(dataset, frame_idx, normalizer):
    """
    Convenience function:
    dataset + frame index -> normalized anomaly field.
    """
    T = dataset.get_frame(frame_idx)
    logT = log_temperature(T)
    return normalizer.transform(logT)

class BackgroundNormalizer:
    """
    Learns a global vertical background profile from the
    entire simulation and applies z-dependent normalization.
    """

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.mean_profile = None
        self.std_profile = None
        self.fitted = False

    def fit(self, dataset):
        """
        Compute μ(z) and σ(z) from all frames.

        Parameters
        ----------
        dataset : SpiculeDataset
        """

        n_frames = dataset.n_frames
        nz = dataset.shape[1]

        sum_profile = np.zeros(nz)
        sumsq_profile = np.zeros(nz)
        total_samples = 0

        print("Computing global background profile...")

        for k in range(n_frames):
            logT = log_temperature(dataset.get_frame(k))

            # Mean over x for each z
            sum_profile += np.sum(logT, axis=1)
            sumsq_profile += np.sum(logT**2, axis=1)

            total_samples += logT.shape[1]  # nx

            if (k + 1) % 25 == 0 or (k + 1) == n_frames:
                print(f"  processed {k+1}/{n_frames} frames")

        self.mean_profile = sum_profile / total_samples

        variance = (
            sumsq_profile / total_samples
            - self.mean_profile**2
        )

        # Numerical safety
        variance = np.maximum(variance, 0.0)

        self.std_profile = np.sqrt(variance)

        self.fitted = True

    def transform(self, logT):
        """
        Normalize a single log-temperature frame.
        """

        if not self.fitted:
            raise RuntimeError(
                "BackgroundNormalizer.fit() must be called first."
            )

        return (
            logT - self.mean_profile[:, np.newaxis]
        ) / (
            self.std_profile[:, np.newaxis] + self.eps
        )
    def save(self, filename):
        np.savez(
            filename,
            mean_profile=self.mean_profile,
            std_profile=self.std_profile,
        )

    def load(self, filename):
        data = np.load(filename)
        self.mean_profile = data["mean_profile"]
        self.std_profile = data["std_profile"]
        self.fitted = True