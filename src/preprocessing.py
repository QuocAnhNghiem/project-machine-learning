import numpy as np


class MinMaxNormalizer:
    """Scale each feature to [0, 1] using training-set min and max."""

    def __init__(self):
        self.x_min = None
        self.x_max = None

    def fit(self, x: np.ndarray) -> None:
        """Store per-feature minima and maxima from training data."""
        self.x_min = np.min(x, axis=0)
        self.x_max = np.max(x, axis=0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply Min-Max scaling: x_new = (x - x_min) / (x_max - x_min)."""
        if self.x_min is None or self.x_max is None:
            raise ValueError("MinMaxNormalizer is not fitted yet.")

        denominator = self.x_max - self.x_min
        denominator = np.where(denominator == 0, 1, denominator)
        return (x - self.x_min) / denominator

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit scaler then apply Min-Max transform on the same matrix."""
        self.fit(x)
        return self.transform(x)
