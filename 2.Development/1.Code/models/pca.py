import numpy as np
from models.spectral import Spectral

class PCA(Spectral):
    def __init__(self, model_args):
        super().__init__(model_args)

    def _fit(self, X: np.ndarray):
        """Computes the kernel matrix (default: similarity matrix)."""
        K = X @ X.T

        n = K.shape[0]
        one_n = np.ones((n, n)) / n

        # self.kernel_ = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        H = np.eye(n) - one_n
        self.kernel_ = H @ K @ H 

        return self
