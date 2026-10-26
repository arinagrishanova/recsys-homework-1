import numpy as np
from scipy import sparse
from .common import TopKMixin


class ItemKNN(TopKMixin):
    """Item cosine similarity with shrinkage and a fixed neighborhood size."""

    def __init__(self, neighbors=100, shrinkage=10.0):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, interactions, n_users, n_items):
        users = interactions["user"].to_numpy(dtype=np.int32)
        items = interactions["item"].to_numpy(dtype=np.int32)
        self.history = sparse.csr_matrix(
            (np.ones(len(users), dtype=np.float32), (users, items)),
            shape=(n_users, n_items),
        )
        self.history.data[:] = 1.0

        cooccurrence = (self.history.T @ self.history).toarray()
        popularity = np.diag(cooccurrence).copy()
        scale = np.sqrt(np.outer(popularity, popularity)) + self.shrinkage
        similarity = np.divide(cooccurrence, scale, out=np.zeros_like(cooccurrence), where=scale > 0)
        np.fill_diagonal(similarity, 0.0)

        k = min(self.neighbors, n_items - 1)
        if k < n_items - 1:
            for item in range(n_items):
                keep = np.argpartition(similarity[:, item], -k)[-k:]
                mask = np.ones(n_items, dtype=bool)
                mask[keep] = False
                similarity[mask, item] = 0.0
        self.similarity = similarity.astype(np.float32)
        return self

    def score(self, users):
        return np.asarray(self.history[np.asarray(users)] @ self.similarity)
