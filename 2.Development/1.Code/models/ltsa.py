# models/ltsa.py

import numpy as np
from scipy.linalg import svd, eigh
from scipy.sparse import csr_matrix, lil_matrix, eye as sparse_eye
from scipy.sparse.linalg import eigsh

import models
from utils import stamp
from plot import plot
import utils

class LTSA(models.Neighbourhood):
    """
    Local Tangent Space Alignment (LTSA) for dimensionality reduction.

    Parameters
    ----------
    model_args : dict
        Dictionary of model arguments, including 'preview', 'overlap', 'verbose', '#neighs', 'model'.
    n_neighbors : int
        Number of neighbors to use for constructing local tangent spaces.
    n_components : int
        Number of dimensions for the embedded space.
    """
    def __init__(self, model_args: dict, n_neighbors: int):
        super().__init__(model_args, n_neighbors)
        self.alignment_matrix_ = None # The global alignment matrix (often denoted B or M)
        if self.model_args['intrinsic'] is None:
            self.model_args['intrinsic'] = 2  # Default intrinsic dimensionality
            utils.warning(f"Setting intrinsic dimensionality to 2 for LTSA as it was not specified.")
        self.n_components = self.model_args['intrinsic']
        # Configure for LTSA-specific behavior
        self.use_smallest_eigenvalues = True  # LTSA needs smallest eigenvalues (null space)
        self.apply_sqrt_scaling = False  # LTSA uses eigenvectors directly

    def _neigh_matrix(self, X: np.ndarray):
        """
        Computes the k-nearest neighbor graph (non-symmetric).
        LTSA uses the direct neighbors for tangent space estimation.
        """
        # Use the default neigh_matrix which finds k closest neighbors for each point
        neigh_matrix = utils.neigh_matrix(X, self.n_neighbors)
        return neigh_matrix # We only need indices later

    def _fit(self, X: np.ndarray):
        """
        Compute the LTSA alignment matrix M (following scikit-learn implementation).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.
        """
        n_samples, n_features = X.shape
        if self.n_neighbors >= n_samples:
             raise ValueError("n_neighbors must be less than n_samples.")
        if self.n_neighbors <= self.n_components:
            utils.hard_warning(f"n_neighbors ({self.n_neighbors}) must be greater than n_components ({self.n_components}) for SVD.")

        stamp.set()
        print("Computing LTSA local tangent spaces and alignment matrix M...")

        # 1. Find Neighbors (following sklearn approach)
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(X)
        neighbors_with_self = knn.kneighbors(X, return_distance=False)
        # Remove the first column (the point itself) to get just the neighbors
        neighbors = neighbors_with_self[:, 1:]  # Shape: (n_samples, n_neighbors)

        # Initialize the matrix M (following sklearn pattern)
        from scipy.sparse import lil_matrix
        M = lil_matrix((n_samples, n_samples), dtype=np.float64)

        # Determine whether to use SVD or eigendecomposition
        use_svd = self.n_neighbors > n_features

        # 2. Compute Local Information for each point
        for i in range(n_samples):
            # Get neighborhood data and center it
            Xi = X[neighbors[i]]  # Shape: (n_neighbors, n_features)
            Xi = Xi - Xi.mean(0)  # Center the neighborhood

            # Compute n_components largest eigenvectors of Xi * Xi^T
            if use_svd:
                # Use SVD: Xi = U @ S @ Vh, we want U (left singular vectors)
                v = svd(Xi, full_matrices=True)[0]  # Shape: (n_neighbors, n_neighbors)
            else:
                # Use eigendecomposition of covariance matrix
                Ci = np.dot(Xi, Xi.T)  # Shape: (n_neighbors, n_neighbors)
                eigenvals, eigenvecs = eigh(Ci)
                # eigh returns eigenvalues in ascending order, reverse to get largest first
                v = eigenvecs[:, ::-1]  # Shape: (n_neighbors, n_neighbors)

            # Construct the Gi matrix: [constant_vector, top_n_components_eigenvectors]
            Gi = np.zeros((self.n_neighbors, self.n_components + 1))
            Gi[:, 1:] = v[:, :self.n_components]  # Top n_components eigenvectors
            Gi[:, 0] = 1.0 / np.sqrt(self.n_neighbors)  # Constant vector (normalized)

            # Compute Gi @ Gi^T
            GiGiT = np.dot(Gi, Gi.T)  # Shape: (n_neighbors, n_neighbors)

            # Update matrix M: M[neighbors[i], neighbors[i]] -= GiGiT
            # This implements: M -= sum_i (S_i @ Gi @ Gi^T @ S_i^T)
            # where S_i is the selection matrix for neighborhood i
            for r, row_idx in enumerate(neighbors[i]):
                for c, col_idx in enumerate(neighbors[i]):
                    M[row_idx, col_idx] -= GiGiT[r, c]

            # Add identity: M[neighbors[i], neighbors[i]] += I
            for idx in neighbors[i]:
                M[idx, idx] += 1.0

        # 3. Convert to CSR format and store
        self.kernel_ = M.tocsr()
        return self

    def _eigenvalue_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform eigenvalue decomposition for LTSA by finding the null space 
        (smallest eigenvalues) of the alignment matrix M.
        """
        if self.kernel_ is None:
            raise ValueError("LTSA matrix M is not initialized. Run fit(X) first.")

        if self.model_args['verbose']:
            print(f"Solving eigenvalue problem for LTSA matrix M (finding null space)...")

        # LTSA needs the SMALLEST eigenvalues of the matrix M (null space approach)
        # Try using sparse solver first
        try:
            # Request more eigenvalues than needed as a buffer, but focus on smallest
            from scipy.sparse.linalg import eigsh
            k_request = min(self.n_components + 2, self.kernel_.shape[0] - 1)
            
            # Use 'SM' (Smallest Magnitude) to find the null space
            eigenvalues, eigenvectors = eigsh(self.kernel_, k=k_request, which='SM', tol=1e-9)

            # eigsh returns eigenvalues in ascending order for 'SM'
            # Keep them in ascending order (smallest first)

            if self.model_args['verbose']:
                print(f"Computed Eigenvalues (smallest {k_request}) via sparse solver:", eigenvalues)

        except Exception as e:
            if self.model_args['verbose']:
                print(f"Sparse solver failed ({e}), attempting dense solver...")
            
            # Fallback to dense solver
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(self.kernel_.toarray())  # eigh sorts ascending
            
            # Keep ascending order (smallest first)

            if self.model_args['verbose']:
                print(f"Computed Eigenvalues (all, showing smallest 10) via dense solver:", eigenvalues[:min(len(eigenvalues), 10)])

        return eigenvalues, eigenvectors
