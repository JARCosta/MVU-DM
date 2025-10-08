import numpy as np

from scipy.sparse import csr_matrix, csgraph
from scipy.spatial.distance import cdist

import models
import utils

class Isomap(models.Neighbourhood):
    def __init__(self, model_args:dict, n_neighbors:int):
        super().__init__(model_args, n_neighbors)

    def _neigh_matrix(self, X:np.ndarray):
        return utils.neigh_matrix(X, self.n_neighbors, bidirectional=True)

    def _fit(self, X):
        """Fit the Isomap model and compute the low-dimensional embeddings."""

        """Compute geodesic distances using k-nearest neighbors graph."""
        n = X.shape[0]
        X = X - np.mean(X, axis=0)  # Center the data

        cc, labels = csgraph.connected_components(self.NM, directed=False)
        if cc > 1:
            self.model_args['artificial_connected'] = True
            self.NM = models.graph.connect_components_sklearn(X, self.NM, cc, labels)
            # self.NM = models.graph.connect_components_default(X, self.NM, cc, labels)

        # Compute shortest paths using Dijkstra
        D_sq = csgraph.shortest_path(self.NM, directed=False, method='D') ** 2  # Squared distance matrix

        n = D_sq.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
        self.kernel_ = -0.5 * H @ D_sq @ H  # Double centering
        
        return self

class Nystrom(models.extensions.Nystrom, Isomap):
    def __init__(self, model_args:dict, n_neighbors:int, ratio:int=None, subset_indices:list=None):
        Isomap.__init__(self, model_args, n_neighbors)
        super().__init__(ratio=ratio, subset_indices=subset_indices)

class ENG(models.extensions.ENG, Isomap):
    pass

class Adaptative(models.extensions.Adaptative, Isomap):
    def __init__(self, model_args, n_neighbors:int, k_max:int, eta:float):
        Isomap.__init__(self, model_args, n_neighbors)
        super().__init__(k_max, eta)

class Adaptative2(models.extensions.Adaptative2, Isomap):
    def __init__(self, model_args, n_neighbors:int, k_max:int, eta:float):
        Isomap.__init__(self, model_args, n_neighbors)
        super().__init__(k_max, eta)

class Our(models.extensions.Our, Isomap):
    def __init__(self, model_args:dict, n_neighbors:int):
        Isomap.__init__(self, model_args, n_neighbors)
        super().__init__(bidirectional=True)
