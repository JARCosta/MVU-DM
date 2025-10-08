import numpy as np
from scipy.linalg import eigh, solve, svd, qr
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, eye as sparse_eye, lil_matrix

import models
from utils import stamp
import utils
from plot import plot

class HessianLLE(models.Neighbourhood):
    """
    Hessian Locally Linear Embedding (Hessian LLE) for dimensionality reduction.

    Parameters
    ----------
    model_args : dict
        Dictionary of model arguments.
    n_neighbors : int
        Number of neighbors to use for reconstruction.
    n_components : int
        Number of dimensions for the embedded space.
    hessian_tol : float, optional (default=1e-4)
        Tolerance for Hessian eigenmapping method.
    """

    def __init__(self, model_args: dict, n_neighbors: int, hessian_tol: float = 1e-4):
        super().__init__(model_args, n_neighbors)
        self.n_components = model_args['intrinsic'] if model_args['intrinsic'] is not None else 2
        self.hessian_tol = hessian_tol
        # Configure for HLLE-specific behavior
        self.use_smallest_eigenvalues = True  # HLLE needs smallest eigenvalues (null space)
        self.apply_sqrt_scaling = False  # HLLE uses eigenvectors directly
    
    def _neigh_matrix(self, X: np.ndarray):
        """
        Computes the k-nearest neighbor graph (non-symmetric).
        HLLE uses the direct neighbors for reconstruction.
        """
        # Use the default neigh_matrix which finds k closest neighbors for each point
        neigh_matrix = utils.neigh_matrix(X, self.n_neighbors)
        return neigh_matrix # Returns distances, we only need indices later

    def _fit(self, X: np.ndarray):
        """
        Compute the Hessian LLE embedding matrix M (following scikit-learn implementation).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        """
        n_samples, n_features = X.shape
        
        # Calculate dp = n_components * (n_components + 1) // 2
        dp = self.n_components * (self.n_components + 1) // 2
        
        if self.n_neighbors <= self.n_components + dp:
            utils.hard_warning(
                "for method='hessian', n_neighbors must be "
                "greater than "
                "[n_components * (n_components + 3) / 2]"
            )

        stamp.set()
        print("Computing Hessian LLE matrix M...")

        # 1. Find Neighbors (following sklearn approach)
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(X)
        neighbors_with_self = knn.kneighbors(X, return_distance=False)
        # Remove the first column (the point itself) to get just the neighbors
        neighbors = neighbors_with_self[:, 1:]  # Shape: (n_samples, n_neighbors)

        # Initialize Yi matrix for Hessian estimation
        Yi = np.empty((self.n_neighbors, 1 + self.n_components + dp), dtype=np.float64)
        Yi[:, 0] = 1  # First column is constant

        # Initialize the matrix M
        M = lil_matrix((n_samples, n_samples), dtype=np.float64)

        # Determine whether to use SVD or eigendecomposition
        use_svd = self.n_neighbors > n_features

        # 2. Compute Local Hessian Information for each point
        for i in range(n_samples):
            # Get neighborhood data and center it
            Gi = X[neighbors[i]]  # Shape: (n_neighbors, n_features)
            Gi = Gi - Gi.mean(0)  # Center the neighborhood

            # Build Hessian estimator
            if use_svd:
                # Use SVD: Gi = U @ S @ Vh, we want U (left singular vectors)
                U = svd(Gi, full_matrices=False)[0]  # Shape: (n_neighbors, min(n_neighbors, n_features))
            else:
                # Use eigendecomposition of covariance matrix
                Ci = np.dot(Gi, Gi.T)  # Shape: (n_neighbors, n_neighbors)
                eigenvals, U = eigh(Ci)
                # eigh returns eigenvalues in ascending order, reverse to get largest first
                U = U[:, ::-1]  # Shape: (n_neighbors, n_neighbors)

            # Fill Yi matrix: [constant, first n_components eigenvectors, Hessian terms]
            Yi[:, 1:1 + self.n_components] = U[:, :self.n_components]

            # Fill Hessian terms (quadratic combinations of eigenvectors)
            j = 1 + self.n_components
            for k in range(self.n_components):
                Yi[:, j:j + self.n_components - k] = U[:, k:k + 1] * U[:, k:self.n_components]
                j += self.n_components - k

            # QR decomposition to find orthogonal complement
            Q, R = qr(Yi)

            # Get the null space weights (orthogonal to the tangent space + Hessian)
            w = Q[:, self.n_components + 1:]  # Shape: (n_neighbors, remaining_dims)
            S = w.sum(0)  # Sum along neighborhoods

            # Regularize small sums
            S[np.where(abs(S) < self.hessian_tol)] = 1
            w /= S  # Normalize

            # Update matrix M: M[neighbors[i], neighbors[i]] += w @ w^T
            for r, row_idx in enumerate(neighbors[i]):
                for c, col_idx in enumerate(neighbors[i]):
                    M[row_idx, col_idx] += np.dot(w[r, :], w[c, :])

        # 3. Convert to CSR format and store
        self.kernel_ = M.tocsr()
        
        stamp.print(f"*\t {self.model_args['model']}\t Computed Hessian LLE Matrix M")
        return self

    def _eigenvalue_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform eigenvalue decomposition for Hessian LLE by finding the null space 
        (smallest eigenvalues) of the matrix M.
        """
        if self.kernel_ is None:
            raise ValueError("Hessian LLE matrix M is not initialized. Run fit(X) first.")

        if self.model_args['verbose']:
            print(f"Solving eigenvalue problem for Hessian LLE matrix M (finding null space)...")

        # Hessian LLE needs the SMALLEST eigenvalues of the matrix M (null space approach)
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


class ENG(models.extensions.ENG, HessianLLE):
    pass