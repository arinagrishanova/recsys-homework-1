import numpy as np
from scipy import sparse
from scipy.linalg import solve
from .common import TopKMixin


class EASE(TopKMixin):
    """Closed-form item-to-item linear reconstruction with zero diagonal."""

    def __init__(self, regularization=300.0):
        self.regularization = regularization

    def fit(self, interactions, n_users, n_items):
        users = interactions["user"].to_numpy(dtype=np.int32)
        items = interactions["item"].to_numpy(dtype=np.int32)
        self.history = sparse.csr_matrix(
            (np.ones(len(users), dtype=np.float64), (users, items)),
            shape=(n_users, n_items),
        )
        self.history.data[:] = 1.0
        gram = (self.history.T @ self.history).toarray()
        gram.flat[:: n_items + 1] += self.regularization
        precision = solve(gram, np.eye(n_items), assume_a="pos", check_finite=False)
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0.0)
        return self

    def score(self, users):
        return np.asarray(self.history[np.asarray(users)] @ self.weights)
