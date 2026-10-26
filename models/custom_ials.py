import numpy as np

from .ials import ImplicitALS


class RatingTargetIALS(ImplicitALS):
    """ALS with p_ui = clip((rating - 3) / 2, 0, 1)."""

    def _targets(self, interactions):
        rating = interactions["rating"].to_numpy(dtype=np.float64)
        return np.clip((rating - 3.0) / 2.0, 0.0, 1.0)
