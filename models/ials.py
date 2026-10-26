import numpy as np
from scipy import sparse
from scipy.linalg import solve
from .common import TopKMixin


class ImplicitALS(TopKMixin):
    """Implicit-feedback ALS with confidence 1 + alpha on observed pairs."""

    def __init__(self, factors=16, alpha=20.0, regularization=0.1, iterations=4, seed=42):
        self.factors = factors
        self.alpha = alpha
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed

    def _targets(self, interactions):
        return np.ones(len(interactions), dtype=np.float64)

    def _update(self, fixed, observed, preferences):
        gram = fixed.T @ fixed
        ridge = self.regularization * np.eye(self.factors)
        result = np.zeros((observed.shape[0], self.factors), dtype=np.float64)
        for row in range(observed.shape[0]):
            start, end = observed.indptr[row : row + 2]
            ids = observed.indices[start:end]
            if len(ids) == 0:
                continue
            vectors = fixed[ids]
            target = preferences.data[start:end]
            lhs = gram + self.alpha * (vectors.T @ vectors) + ridge
            rhs = (1.0 + self.alpha) * (vectors.T @ target)
            result[row] = solve(lhs, rhs, assume_a="pos", check_finite=False)
        return result

    def fit(self, interactions, n_users, n_items):
        user = interactions["user"].to_numpy(dtype=np.int32)
        item = interactions["item"].to_numpy(dtype=np.int32)
        target = self._targets(interactions)
        order = np.lexsort((item, user))
        user, item, target = user[order], item[order], target[order]
        indptr = np.r_[0, np.bincount(user, minlength=n_users).cumsum()]
        observed = sparse.csr_matrix(
            (np.ones(len(item)), item, indptr), shape=(n_users, n_items)
        )
        preferences = sparse.csr_matrix((target, item, indptr), shape=(n_users, n_items))
        observed_t = observed.T.tocsr()
        preferences_t = preferences.T.tocsr()

        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.1, (n_items, self.factors))
        for _ in range(self.iterations):
            self.user_factors = self._update(self.item_factors, observed, preferences)
            self.item_factors = self._update(self.user_factors, observed_t, preferences_t)
        return self

    def score(self, users):
        return self.user_factors[np.asarray(users)] @ self.item_factors.T
