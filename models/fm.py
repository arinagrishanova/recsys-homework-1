import re
from collections import Counter

import numpy as np
from scipy import sparse
from .common import TopKMixin


STOPWORDS = {"the", "and", "for", "with", "from", "app", "free", "android", "download"}


def title_tokens(title):
    return {word for word in re.findall(r"[a-z0-9]+", str(title).lower()) if len(word) >= 3 and word not in STOPWORDS}


class FactorizationMachine(TopKMixin):
    """Second-order FM trained with manually implemented mini-batch Adam."""

    def __init__(self, factors=16, learning_rate=0.02, regularization=0.001,
                 epochs=5, batch_size=4096, use_metadata=True, seed=42):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_metadata = use_metadata
        self.seed = seed

    def _encode_metadata(self, item_metadata, n_items):
        if not self.use_metadata:
            return sparse.csr_matrix((n_items, 0), dtype=np.float32)
        rows = item_metadata.set_index("item").reindex(range(n_items)).fillna("")
        titles = [title_tokens(value) for value in rows["title"]]
        token_counts = Counter(token for words in titles for token in words)
        tokens = [key for key, count in sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))
                  if count >= 5][:200]
        stores = [str(value).strip().lower() for value in rows["store"]]
        store_counts = Counter(stores)
        common_stores = [key for key, count in sorted(store_counts.items(), key=lambda x: (-x[1], x[0]))
                         if key and count >= 3][:80]
        categories = [str(value).strip().lower() for value in rows["main_category"]]
        common_categories = sorted(set(categories) - {""})
        columns = {("title", value): index for index, value in enumerate(tokens)}
        columns.update({("store", value): len(columns) + index for index, value in enumerate(common_stores)})
        columns.update({("category", value): len(columns) + index for index, value in enumerate(common_categories)})
        rr, cc = [], []
        for item in range(n_items):
            features = [("title", word) for word in titles[item]]
            features += [("store", stores[item]), ("category", categories[item])]
            for feature in features:
                if feature in columns:
                    rr.append(item)
                    cc.append(columns[feature])
        return sparse.csr_matrix(
            (np.ones(len(rr), dtype=np.float32), (rr, cc)),
            shape=(n_items, len(columns)),
        )

    def _pair_features(self, users, items):
        count = len(users)
        row = np.arange(count)
        user_part = sparse.csr_matrix(
            (np.ones(count, dtype=np.float32), (row, users)),
            shape=(count, self.n_users),
        )
        item_part = sparse.csr_matrix(
            (np.ones(count, dtype=np.float32), (row, items)),
            shape=(count, self.n_items),
        )
        return sparse.hstack((user_part, item_part, self.item_metadata[items]), format="csr")

    def fit(self, interactions, n_users, n_items, item_metadata):
        self.n_users, self.n_items = n_users, n_items
        self.item_metadata = self._encode_metadata(item_metadata, n_items)
        users = interactions["user"].to_numpy(dtype=np.int32)
        items = interactions["item"].to_numpy(dtype=np.int32)
        labels = (interactions["rating"].to_numpy() >= 4).astype(np.float32)
        rng = np.random.default_rng(self.seed)
        # One sampled unknown item per positive review gives a simple ranking signal.
        # Unknown pairs are only training proxies, not confirmed dislikes.
        seen = [set() for _ in range(n_users)]
        for user, item in zip(users, items):
            seen[user].add(item)
        negative_users = users[labels == 1]
        negative_items = np.empty(len(negative_users), dtype=np.int32)
        for row, user in enumerate(negative_users):
            sampled = int(rng.integers(n_items))
            while sampled in seen[user]:
                sampled = int(rng.integers(n_items))
            negative_items[row] = sampled
        users = np.concatenate([users, negative_users])
        items = np.concatenate([items, negative_items])
        labels = np.concatenate([labels, np.zeros(len(negative_users), dtype=np.float32)])
        features = self._pair_features(users, items)
        n_features = features.shape[1]
        self.bias = 0.0
        self.linear = np.zeros(n_features, dtype=np.float32)
        self.vectors = rng.normal(0, 0.02, (n_features, self.factors)).astype(np.float32)
        m_linear = np.zeros_like(self.linear)
        v_linear = np.zeros_like(self.linear)
        m_vectors = np.zeros_like(self.vectors)
        v_vectors = np.zeros_like(self.vectors)
        m_bias = v_bias = 0.0
        step = 0
        self.training_loss = []
        for _ in range(self.epochs):
            order = rng.permutation(len(labels))
            losses = []
            for start in range(0, len(labels), self.batch_size):
                batch = order[start : start + self.batch_size]
                x = features[batch]
                y = labels[batch]
                projection = np.asarray(x @ self.vectors)
                squared = np.asarray(x @ (self.vectors * self.vectors))
                logits = self.bias + np.asarray(x @ self.linear).ravel()
                logits += 0.5 * np.sum(projection * projection - squared, axis=1)
                losses.append(float(np.mean(np.logaddexp(0, logits) - y * logits)))
                error = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30))) - y
                g_bias = float(np.mean(error))
                g_linear = np.asarray(x.T @ error).ravel() / len(batch)
                g_linear += self.regularization * self.linear
                second = np.asarray(x.multiply(x).T @ error).reshape(-1, 1)
                g_vectors = np.asarray(x.T @ (error[:, None] * projection))
                g_vectors = (g_vectors - second * self.vectors) / len(batch)
                g_vectors += self.regularization * self.vectors
                step += 1
                m_bias = 0.9 * m_bias + 0.1 * g_bias
                v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
                self.bias -= self.learning_rate * (m_bias / (1 - 0.9 ** step)) / (
                    np.sqrt(v_bias / (1 - 0.999 ** step)) + 1e-8
                )
                m_linear = 0.9 * m_linear + 0.1 * g_linear
                v_linear = 0.999 * v_linear + 0.001 * g_linear * g_linear
                self.linear -= self.learning_rate * (m_linear / (1 - 0.9 ** step)) / (
                    np.sqrt(v_linear / (1 - 0.999 ** step)) + 1e-8
                )
                m_vectors = 0.9 * m_vectors + 0.1 * g_vectors
                v_vectors = 0.999 * v_vectors + 0.001 * g_vectors * g_vectors
                self.vectors -= self.learning_rate * (m_vectors / (1 - 0.9 ** step)) / (
                    np.sqrt(v_vectors / (1 - 0.999 ** step)) + 1e-8
                )
            self.training_loss.append(float(np.mean(losses)))
        self._build_item_cache()
        return self

    def _build_item_cache(self):
        item_features = sparse.hstack(
            (sparse.eye(self.n_items, format="csr", dtype=np.float32), self.item_metadata),
            format="csr",
        )
        item_vectors = self.vectors[self.n_users :]
        item_linear = self.linear[self.n_users :]
        self.item_projection = np.asarray(item_features @ item_vectors)
        squared = np.asarray(item_features @ (item_vectors * item_vectors))
        self.item_base = np.asarray(item_features @ item_linear).ravel()
        self.item_base += 0.5 * np.sum(self.item_projection ** 2 - squared, axis=1)

    def score(self, users):
        users = np.asarray(users, dtype=np.int32)
        return (self.bias + self.linear[users, None] + self.item_base[None, :]
                + self.vectors[users] @ self.item_projection.T)
