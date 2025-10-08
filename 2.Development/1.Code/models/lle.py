# models/lle.py

import numpy as np
from scipy.linalg import eigh, solve, LinAlgError, pinv
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, eye as sparse_eye

import models
from utils import stamp
from plot import plot
import utils

class LocallyLinearEmbedding(models.Neighbourhood):
    """
    Locally Linear Embedding (LLE) for dimensionality reduction.

    Parameters
    ----------
    model_args : dict
        Dictionary of model arguments, including 'preview', 'overlap', 'verbose', '#neighs', 'model'.
    n_neighbors : int
        Number of neighbors to use for reconstruction.
    reg : float, optional (default=1e-3)
        Regularization parameter for solving the weight matrix.
    """
    def __init__(self, model_args: dict, n_neighbors: int, reg: float = 1e-3):
        super().__init__(model_args, n_neighbors)
        self.reg = reg
        self.weights_ = None
        self.embedding_matrix_M_ = None # The matrix M = (I-W)^T (I-W)
        # Configure for LLE-specific behavior
        self.use_smallest_eigenvalues = True  # LLE needs smallest eigenvalues
        self.apply_sqrt_scaling = False  # LLE uses eigenvectors directly

    def _neigh_matrix(self, X: np.ndarray):
        """
        Computes the k-nearest neighbor graph (non-symmetric).
        LLE uses the direct neighbors for reconstruction.
        """
        # Use the default neigh_matrix which finds k closest neighbors for each point
        neigh_matrix = utils.neigh_matrix(X, self.n_neighbors)
        return neigh_matrix # Returns distances, we only need indices later

    def _fit(self, X: np.ndarray):
        """
        Compute the reconstruction weights and the LLE embedding matrix M.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.
        """
        n_samples, n_features = X.shape
        if self.n_neighbors >= n_samples:
             raise ValueError("n_neighbors must be less than n_samples.")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")

        stamp.set()

        # 2. Compute Reconstruction Weights W
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            # print(f"{i}/{n_samples}", end="\r")
            neighbors_i = np.where(self.NM[i] != 0)[0].tolist() # Indices of neighbors for point i
            X_i = X[i]
            X_neighbors = X[neighbors_i] # Coordinates of neighbors

            # Center the neighborhood points relative to X_i
            # We want to solve for w in: X_i approx sum(w_j * X_j)
            # Equivalent to minimizing || sum(w_j * (X_j - X_i)) ||^2
            # Let Z = X_neighbors - X_i (shape: n_neighbors, n_features)
            Z = X_neighbors - X_i
            
            # Compute local covariance matrix C = Z @ Z.T (shape: n_neighbors, n_neighbors)
            C = Z @ Z.T
            
            # Add regularization following sklearn's approach exactly
            trace = np.trace(C)
            if trace > 0:
                R = self.reg * trace
            else:
                R = self.reg
            C.flat[::len(neighbors_i) + 1] += R  # Add regularization to diagonal

            # Solve C w = 1 for weights w (vector of size n_neighbors)
            ones_vec = np.ones(len(neighbors_i))
            
            try:
                w = solve(C, ones_vec, assume_a='pos')
            except LinAlgError:
                # Fallback to pseudo-inverse if solve fails
                w = pinv(C) @ ones_vec

            # Normalize weights to sum to 1
            w /= np.sum(w)
            
            # Store weights in the main matrix W
            W[i, neighbors_i] = w

            # except LinAlgError:
            #     print(f"Warning: Singular matrix encountered for point {i}. Setting weights to zero.")
            #     print(C)
            #     breakpoint()
            #     # Handle cases where C is singular - could assign equal weights or zero
            #     W[i, neighbors_i] = 1.0 / len(neighbors_i) # Fallback: equal weights
            # except Exception as e:
            #      print(f"Warning: Error solving for weights for point {i}: {e}")
            #      W[i, neighbors_i] = 1.0 / len(neighbors_i) # Fallback

        self.weights_ = csr_matrix(W)
        stamp.print(f"*\t {self.model_args['model']}\t Computed Weights")

        # 3. Compute Embedding Matrix M = (I - W)^T (I - W)
        I = sparse_eye(n_samples)
        M = (I - self.weights_).T @ (I - self.weights_)
        self.embedding_matrix_M_ = M # Store sparse M

        # The base Spectral class sets self.kernel_ for transformation.
        # Here, M is the matrix whose eigenvectors we need.
        self.kernel_ = self.embedding_matrix_M_ # Assign M to kernel_ for transform

        return self

    def _eigenvalue_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform eigenvalue decomposition for LLE by finding the bottom eigenvectors of M = (I-W)^T(I-W).
        """
        if self.kernel_ is None: # M matrix stored in kernel_
            raise ValueError("Embedding matrix M is not initialized. Run fit(X) first.")

        # Use dense solver for LLE (M matrix is typically not too large)
        eigenvalues, eigenvectors = eigh(self.kernel_.toarray()) # eigh sorts eigenvalues in ascending order

        # Discard the first eigenpair (smallest ~0) - this is the constant eigenvector
        nontrivial_eigenvalues = eigenvalues[1:]
        nontrivial_eigenvectors = eigenvectors[:, 1:]

        if self.model_args['verbose']:
            print(f"Computed Eigenvalues (smallest available):", eigenvalues[:min(len(eigenvalues), 10)])
            print(f"Discarded first eigenvalue (~0):", eigenvalues[0])
            print(f"Remaining eigenvalues:", nontrivial_eigenvalues[:min(len(nontrivial_eigenvalues), 10)])

        return nontrivial_eigenvalues, nontrivial_eigenvectors


class ENG(models.extensions.ENG, LocallyLinearEmbedding):
    pass