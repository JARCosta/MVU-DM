import numpy as np
from sklearn.metrics import pairwise_distances

from models.spectral import Spectral


class KPCA(Spectral):
    def __init__(self, model_args:dict):
        super().__init__(model_args)

    def _compute_gamma(self, X:np.ndarray, D2:np.ndarray) -> float:
        tri = D2[np.triu_indices_from(D2, k=1)]
        median_sq = np.median(tri[tri > 0]) if np.any(tri > 0) else np.median(tri)
        if median_sq <= 0:
            median_sq = np.mean(tri) if tri.size > 0 else 1.0
        return 1.0 / (2.0 * median_sq)

    def _fit(self, X: np.ndarray):
        """Compute centered RBF kernel matrix for Kernel PCA."""
        # Compute squared Euclidean distances
        D2 = pairwise_distances(X, metric='euclidean', squared=True, n_jobs=-1)

        gamma = self._compute_gamma(X, D2)
        K = np.exp(-gamma * D2)

        # Center the kernel
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        Kc = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        self.kernel_ = Kc
        return self


