# models/laplacian.py

import numpy as np

import models
from utils import stamp
import utils

class LaplacianEigenmaps(models.Neighbourhood):
    """
    Laplacian Eigenmaps for dimensionality reduction.

    Parameters
    ----------
    model_args : dict
        Dictionary of model arguments, including 'preview', 'overlap', 'verbose', '#neighs', 'model'.
    n_neighbors : int
        Number of neighbors to use for building the adjacency graph.
    sigma : float, optional (default=1.0)
        Width parameter for the weight matrix.
    """
    def __init__(self, model_args: dict, n_neighbors: int, sigma: float = 1.0):
        super().__init__(model_args, n_neighbors)
        self.sigma = sigma
        self.laplacian_ = None
        self.degree_matrix_ = None
        # Configure for LE-specific behavior
        self.use_smallest_eigenvalues = True  # LE needs smallest eigenvalues
        self.apply_sqrt_scaling = False  # LE uses eigenvectors directly

    def _neigh_matrix(self, X: np.ndarray):
        """
        Computes the k-nearest neighbor graph and returns the distance matrix.
        Uses bidirectional connections.
        """
        neigh_matrix = utils.neigh_matrix(X, self.n_neighbors, bidirectional=True)
        # neigh_matrix = neigh_matrix + neigh_matrix.T # no need for symmetry here?, Weight matrix will be made symmetric
        return neigh_matrix

    def _fit(self, X: np.ndarray):
        """
        Compute the graph Laplacian matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.
        """
        from scipy.sparse import csr_matrix

        n_samples = X.shape[0]
        
        neigh_dist_matrix = self.NM # Get distances for neighbors
        
        W = np.zeros((n_samples, n_samples))
        non_zero_indices = np.where(neigh_dist_matrix > 0)

        # Gaussian kernel function: W_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
        dists_sq = neigh_dist_matrix[non_zero_indices]**2
        self.sigma = np.std(dists_sq)
        W[non_zero_indices] = np.exp(-dists_sq / (2 * self.sigma**2))
        # Make W symmetric
        W = np.maximum(W, W.T)
        self.kernel_ = W # Store the weight matrix if needed, though LE uses Laplacian

        # 2. Compute Degree Matrix D
        D = np.diag(np.sum(W, axis=1))
        self.degree_matrix_ = D # Store degree matrix for generalized eigenproblem

        # 3. Compute Graph Laplacian L = D - W
        L = D - W
        self.laplacian_ = csr_matrix(L) # Store sparse Laplacian

        # Note: The base Spectral class assumes self.kernel_ is the matrix
        # for the standard eigenvalue problem. For LE, we need L v = lambda D v.
        # We will handle this in the _transform method.
        # Setting self.kernel_ = L might work if D is identity, but not generally.
        # We'll store L and D separately.

        return self

    def _eigenvalue_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform eigenvalue decomposition for LaplacianEigenmaps by solving the generalized
        eigenvalue problem L v = lambda D v.
        """
        from scipy.linalg import eig, eigh
        if self.laplacian_ is None or self.degree_matrix_ is None:
            raise ValueError("Laplacian matrix is not initialized. Run fit(X) first.")

        if self.model_args['verbose']:
            print(f"Solving generalized eigenvalue problem: L v = lambda D v")

        # Solve generalized eigenvalue problem L v = lambda D v
        try:
            eigenvalues, eigenvectors = eigh(self.laplacian_.toarray(), self.degree_matrix_)
        except Exception as e:
            print(f"Error solving generalized eigenvalue problem: {e}")
            eigenvalues, eigenvectors = eig(self.laplacian_.toarray(), self.degree_matrix_)
            eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real


        # Discard the first eigenpair (smallest ~0) - this is the constant eigenvector
        nontrivial_eigenvalues = eigenvalues[1:]
        nontrivial_eigenvectors = eigenvectors[:, 1:]

        if self.model_args['verbose']:
            print(f"Computed Eigenvalues (smallest available):", eigenvalues[:min(len(eigenvalues), 10)])
            print(f"Discarded first eigenvalue (~0):", eigenvalues[0])
            print(f"Remaining eigenvalues:", nontrivial_eigenvalues[:min(len(nontrivial_eigenvalues), 10)])

        return nontrivial_eigenvalues, nontrivial_eigenvectors

class ENG(models.extensions.ENG, LaplacianEigenmaps):
    pass