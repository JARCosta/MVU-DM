# models/tsne.py

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import warnings

import models
from utils import stamp
from plot import plot
import utils

class TSNE(models.Spectral):
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction.

    Parameters
    ----------
    model_args : dict
        Dictionary of model arguments, including 'preview', 'overlap', 'verbose', '#neighs', 'model'.
    target_dimension : int, optional (default=2)
        The target dimension for the embedding.
    perplexity : float, optional (default=30.0)
        The perplexity parameter that controls the effective number of neighbors.
    learning_rate : float, optional (default=200.0)
        The learning rate for the optimization.
    n_iter : int, optional (default=1000)
        Maximum number of iterations for the optimization.
    early_exaggeration : float, optional (default=12.0)
        Controls how tight natural clusters in the original space are in the embedded space.
    min_grad_norm : float, optional (default=1e-7)
        If the gradient norm is below this threshold, the optimization will be stopped.
    """
    
    def __init__(self, model_args: dict, target_dimension: int = 2, perplexity: float = 30.0, 
                 learning_rate: float = 200.0, n_iter: int = 1000, 
                 early_exaggeration: float = 12.0, min_grad_norm: float = 1e-7):
        super().__init__(model_args)
        self.target_dimension = target_dimension
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.min_grad_norm = min_grad_norm
        self.P_ = None  # High-dimensional similarities
        self.Q_ = None  # Low-dimensional similarities
        self.gains_ = None  # Momentum gains
        self.velocities_ = None  # Momentum velocities

    def _fit(self, X: np.ndarray):
        """
        Compute the high-dimensional similarities and initialize the embedding.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.
        """
        n_samples = X.shape[0]
        
        if n_samples < 4 * self.perplexity:
            raise ValueError("perplexity must be less than n_samples / 4")
        
        stamp.set()
        
        # Compute pairwise distances
        distances = pdist(X, metric='euclidean')
        distances = squareform(distances)
        
        # Compute high-dimensional similarities (joint probabilities)
        self.P_ = self._compute_high_dimensional_similarities(distances)
        
        # Initialize embedding randomly
        np.random.seed(42)  # For reproducibility
        Y = np.random.randn(n_samples, self.target_dimension) * 1e-4
        
        # Initialize momentum
        self.gains_ = np.ones((n_samples, self.target_dimension))
        self.velocities_ = np.zeros((n_samples, self.target_dimension))
        
        stamp.print(f"*\t {self.model_args['model']}\t Computed high-dimensional similarities")
        
        # Store initial embedding for optimization
        self.embedding_ = Y
        return self

    def _compute_high_dimensional_similarities(self, distances):
        """Compute the high-dimensional similarities using Gaussian kernels."""
        n_samples = distances.shape[0]
        
        # Compute conditional probabilities
        P = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            # Find the optimal sigma for this point
            sigma = self._find_optimal_sigma(distances[i], i)
            
            # Compute conditional probabilities
            P[i] = self._compute_conditional_probabilities(distances[i], sigma)
        
        # Symmetrize the probabilities
        P = (P + P.T) / (2 * n_samples)
        
        # Ensure probabilities sum to 1
        P = np.maximum(P, 1e-12)  # Avoid division by zero
        P = P / np.sum(P)
        
        return P

    def _find_optimal_sigma(self, distances, point_idx):
        """Find the optimal sigma for a given point using binary search."""
        # Remove the distance to itself
        distances = distances[distances > 0]
        
        def entropy(sigma):
            P = self._compute_conditional_probabilities(distances, sigma)
            P = P[P > 0]  # Remove zeros
            return -np.sum(P * np.log2(P))
        
        # Binary search for the sigma that gives the target entropy
        target_entropy = np.log2(self.perplexity)
        
        sigma_min = 1e-10
        sigma_max = 1e10
        
        for _ in range(50):  # Maximum 50 iterations
            sigma = (sigma_min + sigma_max) / 2
            current_entropy = entropy(sigma)
            
            if np.abs(current_entropy - target_entropy) < 1e-5:
                break
            elif current_entropy > target_entropy:
                sigma_min = sigma
            else:
                sigma_max = sigma
        
        return sigma

    def _compute_conditional_probabilities(self, distances, sigma):
        """Compute conditional probabilities for a given point."""
        # Compute unnormalized probabilities
        P = np.exp(-distances**2 / (2 * sigma**2))
        
        # Normalize
        P = P / np.sum(P)
        
        return P

    def _transform(self):
        """
        Optimize the embedding using gradient descent to minimize KL divergence.
        """
        if self.P_ is None:
            raise ValueError("High-dimensional similarities not computed. Run fit(X) first.")
        
        stamp.set()
        
        # Optimize the embedding
        Y = self._optimize_embedding(self.embedding_)
        
        self.embedding_ = Y
        
        if self.model_args['preview']:
            plot(self.embedding_, title=f"{self.model_args['model']} output", block=False)
        
        stamp.print(f"*\t {self.model_args['model']}\t transform (t-SNE)")
        
        return self.embedding_

    def _optimize_embedding(self, Y):
        """Optimize the embedding using gradient descent with momentum."""
        n_samples = Y.shape[0]
        
        # Phase 1: Early exaggeration with lower momentum
        P = self.P_ * self.early_exaggeration
        P = P / np.sum(P)
        
        # Early exaggeration phase (first 250 iterations)
        early_exaggeration_iter = min(250, self.n_iter)
        
        for iteration in range(early_exaggeration_iter):
            # Compute low-dimensional similarities
            Q = self._compute_low_dimensional_similarities(Y)
            
            # Compute gradient
            gradient = self._compute_gradient(P, Q, Y)
            
            # Update with momentum (lower momentum in early phase)
            Y, self.gains_, self.velocities_ = self._update_with_momentum(
                Y, gradient, self.gains_, self.velocities_, iteration, momentum=0.5
            )
            
            # Check for convergence
            if np.linalg.norm(gradient) < self.min_grad_norm:
                if self.model_args['verbose']:
                    print(f"Converged at iteration {iteration}")
                return Y
            
            if self.model_args['verbose'] and iteration % 50 == 0:
                kl_div = self._compute_kl_divergence(P, Q)
                print(f"Iteration {iteration}: KL divergence = {kl_div:.4f}")
        
        # Phase 2: Normal phase with higher momentum
        P = self.P_
        P = P / np.sum(P)
        
        if self.model_args['verbose']:
            print(f"Switched to normal phase at iteration {early_exaggeration_iter}")
        
        # Continue optimization with higher momentum
        for iteration in range(early_exaggeration_iter, self.n_iter):
            # Compute low-dimensional similarities
            Q = self._compute_low_dimensional_similarities(Y)
            
            # Compute gradient
            gradient = self._compute_gradient(P, Q, Y)
            
            # Update with momentum (higher momentum in normal phase)
            Y, self.gains_, self.velocities_ = self._update_with_momentum(
                Y, gradient, self.gains_, self.velocities_, iteration, momentum=0.8
            )
            
            # Check for convergence
            if np.linalg.norm(gradient) < self.min_grad_norm:
                if self.model_args['verbose']:
                    print(f"Converged at iteration {iteration}")
                break
            
            if self.model_args['verbose'] and iteration % 100 == 0:
                kl_div = self._compute_kl_divergence(P, Q)
                print(f"Iteration {iteration}: KL divergence = {kl_div:.4f}")
        
        return Y

    def _compute_low_dimensional_similarities(self, Y):
        """Compute low-dimensional similarities using Student's t-distribution."""
        # Compute pairwise distances
        distances = pdist(Y, metric='euclidean')
        distances = squareform(distances)
        
        # Compute similarities using Student's t-distribution
        Q = 1 / (1 + distances**2)
        np.fill_diagonal(Q, 0)  # Set diagonal to 0
        
        # Normalize
        Q = Q / np.sum(Q)
        Q = np.maximum(Q, 1e-12)  # Avoid division by zero
        
        return Q

    def _compute_gradient(self, P, Q, Y):
        """Compute the gradient of the KL divergence."""
        n_samples = Y.shape[0]
        gradient = np.zeros_like(Y)
        
        # Compute pairwise differences
        Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]  # Shape: (n_samples, n_samples, target_dimension)
        
        # Compute the gradient
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    # Compute the gradient contribution
                    grad_contrib = 4 * (P[i, j] - Q[i, j]) * Q[i, j] * Y_diff[i, j]
                    gradient[i] += grad_contrib
        
        return gradient

    def _update_with_momentum(self, Y, gradient, gains, velocities, iteration, momentum=0.5):
        """Update the embedding using gradient descent with momentum."""
        # Update gains
        gains = np.where((gradient > 0) != (velocities > 0), 
                        gains + 0.2, 
                        gains * 0.8)
        gains = np.clip(gains, 0.01, None)  # Clip gains
        
        # Update velocities with specified momentum
        velocities = momentum * velocities - self.learning_rate * gains * gradient
        
        # Update embedding
        Y = Y + velocities
        
        return Y, gains, velocities

    def _compute_kl_divergence(self, P, Q):
        """Compute the Kullback-Leibler divergence between P and Q."""
        # Avoid log(0) by adding small epsilon
        P = np.maximum(P, 1e-12)
        Q = np.maximum(Q, 1e-12)
        
        return np.sum(P * np.log(P / Q))

