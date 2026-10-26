import numpy as np


class TopKMixin:
    """Rank all catalog items, optionally excluding a CSR history matrix."""

    def recommend(self, users, k=10, exclude=None):
        users = np.asarray(users, dtype=np.int32)
        scores = np.asarray(self.score(users), dtype=np.float64).copy()
        if exclude is not None:
            for row, user in enumerate(users):
                start, end = exclude.indptr[user : user + 2]
                scores[row, exclude.indices[start:end]] = -np.inf
        return np.argsort(-scores, axis=1, kind="stable")[:, :k]
