import numpy as np

from scipy.sparse import csgraph
from scipy.spatial.distance import cdist

import plot
from utils import stamp


import skdim


class ENG:
    def _neigh_matrix(self, X):
        def get_eta_d(X:np.ndarray):
            """
            Compute the eta_d value for the given dataset Y and intrinsic dimension d.
            """
            import utils
                        
            k = self.n_neighbors

            NG = utils.neigh_graph(X, self.n_neighbors)

            # Estimate intrinsic dimension using scikit-dimension MLE
            mle_estimator = skdim.id.MLE()
            d_estimated = mle_estimator.fit(X).dimension_
            d_estimated = max(1, min(int(round(d_estimated)), X.shape[1]))
            stamp.print_set(f"*\t MLE intrinsic dimension\t {d_estimated:.2f}")

            eta_d = np.zeros((X.shape[0]))
            for i in range(X.shape[0]):
                Xi_neighs = X[NG[i].indices]
                sigma_i = np.linalg.svd(Xi_neighs - X[i] @ np.ones((X.shape[1], 1)), compute_uv=False)                
                eta_d[i] = np.sum(sigma_i[:d_estimated]) / np.sum(sigma_i[range(min(X.shape[1], k))])
            eta_d = np.mean(eta_d)

            return eta_d, d_estimated

        def connections(comp1:np.ndarray, comp2:np.ndarray, d:int, xi=0.95):
            Xp, Xq = X[comp1], X[comp2]
            s = min(Xp.shape[0], Xq.shape[0])
            
            dist_matrix = cdist(Xp, Xq)  # Compute distances

            connections = np.zeros((s, 2)).astype(int)
            for u in range(s):
                a, b = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                connections[u] = [a, b]
                # print(dist_matrix[a][b])
                dist_matrix[a][b] = np.inf

            l_max = d+1-1
            for l in range(d+1, (s)+1):
                a, b = connections[:l].T # 2xl, closest l connections between p and q
                # print(Yp[a], Yq[b])
                delta_pq_l = Xp[a] - Xq[b]
                # print(delta_l)
                
                sigma_l = np.linalg.svd(delta_pq_l, compute_uv=False)
                # print(sigma_l)
                
                eta_dl = np.sum(sigma_l[range(d)]) / sum(sigma_l[range(min(X.shape[1], l))])
                # print(sigma_l[range(d)], sigma_l[range(min(X.shape[1], l))])
                # print(eta_dl, self.eta_d, xi * self.eta_d)

                l_max = l-1

                if eta_dl < xi * self.eta_d:
                    # print(l_max)
                    break 
            
            a, b = connections[:l_max].T
            return np.vstack((comp1[a], comp2[b])).T

        neigh_matrix = super()._neigh_matrix(X)
        oldNM = neigh_matrix.copy()
        cc = csgraph.connected_components(neigh_matrix)

        self.eta_d, d = get_eta_d(X)
        
        while cc[0] > 1:
            # print(cc[0])

            for i in range(cc[0]):
                comp1 = np.where(cc[1] == i)[0]
                rest_comp = np.where(cc[1] != i)[0]
                dist = cdist(X[comp1], X[rest_comp])
                a,b = np.unravel_index(np.argmin(dist), dist.shape)
                a, b = comp1[a], rest_comp[b]
                comp2 = np.where(cc[1] == cc[1][b])[0]

                p1, p2 = connections(comp1, comp2, d).T
                # print(np.vstack((p1, p2, np.sqrt(np.sum(np.square(X[p2]-X[p1]), axis=1)))).T)

                for a, b in zip(p1, p2):
                    dist = np.linalg.norm(X[a] - X[b])
                    neigh_matrix[a, b] = dist
                    # neigh_matrix[b, a] = dist

                # for a, b, d in np.vstack((p1, p2, np.sqrt(np.sum(np.square(X[p2]-X[p1]), axis=1)))).T:
                #     neigh_matrix[int(a)][int(b)] = d
                stamp.print_set(f"*\t connections\t comp_{i}, comp_{cc[1][b]}, #cons={len(p1)}, #total_cons={np.count_nonzero(neigh_matrix)}")
            cc = csgraph.connected_components(neigh_matrix)

        if self.model_args['preview']:
            plot.plot_two(X, X, oldNM, neigh_matrix, block=False, title=f"{self.model_args['dataname']} {self.model_args['#neighs']} neighbors")
        return neigh_matrix
