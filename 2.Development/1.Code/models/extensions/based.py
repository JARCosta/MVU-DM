from datetime import datetime
from re import I
import numpy as np
import multiprocessing
import threading
import time
import queue
import matplotlib.pyplot as plt


from scipy.sparse import csgraph
import scipy.spatial as sp

import models
import utils
import plot

np.set_printoptions(linewidth=200)

def get_NM(Xl:list[np.ndarray], M:list[list[int]], L:list[list[list[int]]], Yl:list[np.ndarray]=None):
    """
    Neighborhood Matrix containing the reference points, and their connections (intra and inter connections).

    
    """
    XM = [Xi[M[i]] for i, Xi in enumerate(Xl)]
    XMs = np.vstack(XM)

    if Yl is not None:
        YM = [Yi[M[i]] for i, Yi in enumerate(Yl)]

    NM = np.zeros((len(XMs), len(XMs)))
    # intra-component connections (distances from local embeddings)
    for i, XMi in enumerate(XM):
        ptr = np.sum([len(XM[j]) for j in range(i)])
        ptr = int(ptr)

        k = min(int(len(XMi)/2)+1, len(XMi) - 1)
        k = len(XMi) - 1


        neigh_matrix = utils.neigh_matrix(YM[i], k) if Yl is not None else utils.neigh_matrix(XMi, k)
        NM[ptr:ptr+len(XMi), ptr:ptr+len(XMi)] = neigh_matrix
    
    # inter-component connections (distances in original space)
    for [i, p], [j, q], dist in L:
        a = np.sum([len(XM[k]) for k in range(i)]) + p
        b = np.sum([len(XM[k]) for k in range(j)]) + q
        a, b = int(a), int(b)
        norm = np.linalg.norm(XMs[b] - XMs[a])
        if dist != norm: # cdist != norm
            utils.hard_warning(f"global_dist != point_dist: dist: {dist}, norm: {norm}")
        NM[a, b] = dist
    return NM

def get_NMg(Xl:list[np.ndarray], M:list[list[int]], L:list[list[list[int]]]):
    """
    Neighborhood Matrix containing all the points, but connections between reference (intra and inter connections)

    Parameters
    ----------
        Xl : list[np.ndarray]
            Componentwise data points. (stackable/same dimensions)
        M : list[list[int]]
            Componentwise representative points.
        L : list[list[list[int]]]
            Inter-component connections.

    Returns
    -------
        NM : np.ndarray
            Global connection matrix.
    """

    Xs = np.vstack(Xl)
    NM = np.zeros((len(Xs), len(Xs)))
    for i in range(len(Xl)):
        ptr = np.sum([len(Xl[k]) for k in range(i)])
        ptr = int(ptr)
        for a in M[i]:
            for b in M[i]:
                if b > a:
                    NM[ptr+a, ptr+b] = 1

    for [i, p], [j, q], _ in L:
        a_idx = np.sum([len(Xl[k]) for k in range(i)]) + M[i][p]
        b_idx = np.sum([len(Xl[k]) for k in range(j)]) + M[j][q]
        a_idx, b_idx = int(a_idx), int(b_idx)
        NM[a_idx, b_idx] = 1

    return NM

def _engine_worker(work_queue, Yl_results, results_lock, model_args, k1, k2, engine_id):
    # Each worker gets its own MATLAB engine and processes jobs from the shared queue
    
    eng = models.mvu.launch_matlab()
    utils.important(f"Engine {engine_id} started")
    
    while True:
        try:
            # Get work from queue (blocks until available)
            i, Xi = work_queue.get(timeout=1)
            if i is None:  # Sentinel to stop
                break
            
            utils.important(f"Engine {engine_id} - Component {i} started")
            start_time = datetime.now()
            
            temp = model_args['preview'], model_args['intrinsic']
            model_args['preview'], model_args['intrinsic'] = False, None
            
            # Create a new MVU model instance with the necessary parameters
            model = models.mvu.MVU(model_args, k1)
            model.matlab_eng = eng
            model.id = i
            Yi = model.fit_transform(Xi)

            model_args['preview'], model_args['intrinsic'] = temp
            
            with results_lock:
                Yl_results[i] = Yi
            
            end_time = datetime.now()
            utils.important(f"Engine {engine_id} - Component {i} took {end_time - start_time} seconds")
            # with open(f"process.csv", "a") as f:
            #     f.write(f"{i},{(end_time - start_time).total_seconds()},{start_time.strftime('%M:%S')},{end_time.strftime('%M:%S')} \n")
            work_queue.task_done()
        except:
            # Timeout or other error, continue checking for work
            continue


class Based:
    def __init__(self, k1:int, k2:int):
        self.k1 = k1
        self.k2 = 2

    def NG(self, X:np.ndarray, k1:int, Ls:np.ndarray) -> list[np.ndarray] | None:
        """
        Parameters
        ----------
            X : np.ndarray
                Original points.
            k1 : int
                Size of the k-neighbourhood.

        Returns
        -------
            Xl : list[np.ndarray]
                Componentwise original points.
        """

        NG = utils.neigh_graph(X, k1, bidirectional=True)
        cc, labels = csgraph.connected_components(NG, directed=False)
        if cc == 1:
            utils.warning("The data is already connected. Running default model.")
            self.model_args['artificial_connected'] = False
            return None, None
        
        Xl:list[np.ndarray] = [X[labels == c] for c in range(cc)]
        Ll:list[np.ndarray] = [Ls[labels == c] for c in range(cc)]
        return Xl, Ll

    def MVU_local(self, Xl:list[np.ndarray]):
        """
        Parameters
        ----------
            Xl : list[np.ndarray]
                Componentwise original points.

        Returns
        -------
            Yl : list[np.ndarray]
                Componentwise local embeddings.
        """

        # Two-engine pool with shared work queue for optimal load balancing
        # Create work queue and add all components
        work_queue = multiprocessing.Queue()
        for i, Xi in enumerate(Xl):
            work_queue.put((i, Xi))
        
        # Create shared results dictionary and lock for multiprocessing
        manager = multiprocessing.Manager()
        Yl_results = manager.dict()
        results_lock = manager.Lock()
        
        # Start two engine workers
        workers = []
        engines = 4
        for engine_id in range(engines):
            t = multiprocessing.Process(target=_engine_worker, args=(work_queue, Yl_results, results_lock, self.model_args, self.k1, self.k2, engine_id))
            t.start()
            workers.append(t)
        
        # Send sentinel values to stop workers
        for _ in range(engines):
            work_queue.put((None, None))
        
        # Wait for all work to complete
        for t in workers:
            t.join()
        
        for p in range(len(Xl)):
            print(f"{p}: X({len(Xl[p])}), Y({len(Yl_results[p])})")
        
        Yl:list[np.ndarray] = [Yl_results[i] for i in range(len(Xl))]
        # for i, Yi in Yl_results.items():
        #     # if Yi.shape[1] > 6:
        #     #     Yl.append(Yi[:, :3])
        #     # else:
        #         Yl.append(Yi)

        # if self.model_args['verbose']:
        #     for i, Xi in enumerate(Xl):
        #         Xi_dist = sp.distance.cdist(Xi, Xi)
        #         for a in range(len(Xi)):
        #             Xi_dist[a][a] = np.median(Xi_dist)
        #         print(f"idx_min=({np.argmin(Xi_dist)//len(Xi)},{np.argmin(Xi_dist)%len(Xi)}), value={np.min(Xi_dist)}")
        #         print(f"idx_max=({np.argmax(Xi_dist)//len(Xi)},{np.argmax(Xi_dist)%len(Xi)}), value={np.max(Xi_dist)}")
                
        #         Yi = Yl[i]
        #         Yi_dist = sp.distance.cdist(Yi, Yi)
        #         for a in range(len(Yi)):
        #             Yi_dist[a][a] = np.median(Yi_dist)
        #         print(f"idx_min=({np.argmin(Yi_dist)//len(Yi)},{np.argmin(Yi_dist)%len(Yi)}), value={np.min(Yi_dist)}")
        #         print(f"idx_max=({np.argmax(Yi_dist)//len(Yi)},{np.argmax(Yi_dist)%len(Yi)}), value={np.max(Yi_dist)}")
        utils.stamp.print(f"*\t {self.model_args['model']}\t MVU local")
        return Yl

    def chose_representative_points_iterative(self, Yl:list[np.ndarray], d:list[int], M:list[list[int]] = None):
        """
        For each component `Yi`, return the componentwise indexes of the representative points `Mi`.
        
        The next representative point is chosen as the point that is furthest from the current representative points (Sum distances).

        Parameters
        ----------
            Yl : list[np.ndarray]
                Componentwise local embeddings.
            d : list[int]
                Componentwise dimension.
            M : list[list[int]], optional
                Componentwise representative points.

        Returns
        -------
            M : list[list[int]]
                Componentwise representative points.
        """

        if M is None:
            M = [[] for _ in range(len(Yl))]

        def get_next_representative(Yp:np.ndarray, Mp:list[int]):
            if len(Mp) > 0:
                dist_matrix = np.zeros((Yp.shape[0], ))

                for ma in Mp:
                    temp_dist = np.linalg.norm(Yp - Yp[ma], axis=1)
                    dist_matrix = dist_matrix + temp_dist
                    dist_matrix[Mp] = -np.inf
                
                return np.argmax(dist_matrix)
            else:
                return np.argmax(np.linalg.norm(Yp, axis=1))

        for p, Yp in enumerate(Yl):
            for _ in range(len(M[p]), d[p]**2):
                M[p].append(get_next_representative(Yp, M[p]))
        return M

    def chose_representative_points_PCA(self, Yl:list[np.ndarray], d:list[int], M:list[list[int]] = None):
        """
        For each component `Yi`, return the componentwise indexes of the representative points `Mi`.
        
        Apply PCA to each component `Yi` and chose the two points that are furthest from the center, for each dimension `d[i]`.

        Parameters
        ----------
            Yl : list[np.ndarray]
                Componentwise local embeddings.
            d : list[int]
                Componentwise dimension.
            M : list[list[int]], optional
                Componentwise representative points.

        Returns
        -------
            M : list[list[int]]
                Componentwise representative points.
        """
        
        if M is None:
            M = [[] for _ in range(len(Yl))]

        for p, Yp in enumerate(Yl):
            kernel = Yp @ Yp.T
            H = np.eye(len(Yp)) - np.ones((len(Yp), len(Yp))) / len(Yp)
            kernel = H @ kernel @ H
            eigenvalues, eigenvectors = np.linalg.eigh(kernel)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
            eigenvalues = eigenvalues[:d[p]]
            eigenvectors = eigenvectors[:, :d[p]]
            restricted_embedding = eigenvectors @ np.diag(eigenvalues)
            

            max_idx = np.argmax(restricted_embedding, axis=0)
            M[p].extend([int(i) for i in max_idx])
            min_idx = np.argmin(restricted_embedding, axis=0)
            M[p].extend([int(i) for i in min_idx])
        return M


    def chose_representative_points_landmark(self, Yl:list[np.ndarray], d:list[int], M:list[list[int]] = None):
        """
        For each component `Yi`, return the componentwise indexes of the representative points `Mi`.
        
        Select a certain percentage of datapoints as representative points.
        """
        percentage = 0.1
        
        if M is None:
            M = [[] for _ in range(len(Yl))]
        

        for i, Yi in enumerate(Yl):
            selection = np.random.choice(len(Yi), size=int(len(Yi) * percentage), replace=False)
            print(f"Selection {i}: {len(selection)}")
            if len(selection) < self.k1:
                utils.hard_warning(f"Less than {self.k1}(k1) representative points selected for component {i}. Selecting {self.k1} points.")
                selection = np.random.choice(len(Yi), size=self.k1, replace=False)
            M[i].extend(selection)
        return M

    def choose_representative_points_hull(self, Yl:list[np.ndarray], d:list[int], M:list[list[int]] = None):
        """
        For each component, compute its convex hull and use its vertices as representative points.
        """

        if M is None:
            M = [[] for _ in range(len(Yl))]

        for p, Yp in enumerate(Yl):
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(Yp, qhull_options="Qa QJ")
                M[p].extend([int(v) for v in hull.vertices])
                print(f"Component {p}: {len(hull.vertices)} representative points selected.")
            except Exception as _e:
                utils.hard_warning(f"Convex hull could not be computed for component {p}. Using iterrative selection instead.")
                M[p] = self.chose_representative_points_iterative([Yp], [d[p]], [M[p]])[0]
                print(f"Component {p}: {len(M[p])} representative points selected.")
        return M


    def chose_intercomponent_connections_sklearn(self, Xl:list[np.ndarray], M:list[list[int]]):
        """
        Iteratively, connect each component to the next closest component, until all components are connected.
        """

        L:list[list[list[int]]] = [] # L.append([[0, 345], [2 , 741], 34.23])

        Xs = np.vstack(Xl)
        NG = utils.neigh_graph(Xs, self.k1)
        cc, labels = csgraph.connected_components(NG, directed=False)
        while cc > 1:
            Xc = [Xs[labels == c] for c in range(cc)]

            for r, Xr in enumerate(Xc):
                for s, Xs in enumerate(Xc):
                    if r != s:
                        distances = sp.distance.cdist(Xr, Xs, metric='euclidean')
                        shortest_distance = np.min(distances)
                        ab = np.where(distances == shortest_distance)

                        a_coords = Xr[ab[0]]
                        b_coords = Xs[ab[1]]

                        for p in range(len(Xl)):
                            if np.any(np.all(Xl[p] == a_coords, axis=1)):
                                a_component = p
                                a_idx = np.where(np.all(Xl[p] == a_coords, axis=1))[0][0]
                            if np.any(np.all(Xl[p] == b_coords, axis=1)):
                                b_component = p
                                b_idx = np.where(np.all(Xl[p] == b_coords, axis=1))[0][0]
                            
                            p, i = a_component, a_idx
                            q, j = b_component, b_idx
                        
                            if self.model_args['verbose']:
                                print(f"{p}/{len(Xl[i])} ({i}) -> {q}/{len(Xl[j])} ({j})")

                            a_Gidx = int(np.sum([len(Xl[k]) for k in range(a_component)]) + a_idx)
                            b_Gidx = int(np.sum([len(Xl[k]) for k in range(b_component)]) + b_idx)
                            NG[a_Gidx, b_Gidx] = shortest_distance
                            
                            if i not in M[p]:
                                M[p].append(i)
                            if j not in M[q]:
                                M[q].append(j)
                            i = M[p].index(i)
                            j = M[q].index(j)

                            L.append([[p, i], [q, j], shortest_distance])
                            print(f"{i}/{len(M[p])} ({p}) -> {j}/{len(M[q])} ({q})")
            cc, labels = csgraph.connected_components(NG, directed=False)



    def chose_intercomponent_connections_default(self, Xl:list[np.ndarray], M:list[list[int]]):
        """
        Iteratively connect the largest component to the next closest component.

        Parameters
        ----------
            Xl : list[np.ndarray]
                Componentwise original points.
            M : list[list[int]]
                Componentwise representative points.

        Returns
        -------
            L : list[list[list[int]]]
                Inter-component connections.
        """

        L:list[list[list[int]]] = [] # L.append([[0, 345], [2 , 741], 34.23])

        Xs = np.vstack(Xl)
        NG = utils.neigh_graph(Xs, self.k1)
        cc, labels = csgraph.connected_components(NG, directed=False)
        while cc > 1:
            largest_component = np.argmax(np.bincount(labels))
            largest_component_idx = np.where(labels == largest_component)[0]
            other_idx = np.where(labels != largest_component)[0]

            distances = sp.distance.cdist(Xs[largest_component_idx], Xs[other_idx], metric='euclidean')
            shortest_distance = np.min(distances)
            ab = np.where(distances == shortest_distance)

            a_coords = Xs[largest_component_idx][ab[0]]
            b_coords = Xs[other_idx][ab[1]]
            
            for p in range(len(Xl)):
                if np.any(np.all(Xl[p] == a_coords, axis=1)):
                    a_component = p
                    a_idx = np.where(np.all(Xl[p] == a_coords, axis=1))[0][0]
                if np.any(np.all(Xl[p] == b_coords, axis=1)):
                    b_component = p
                    b_idx = np.where(np.all(Xl[p] == b_coords, axis=1))[0][0]

            p, i = a_component, a_idx
            q, j = b_component, b_idx
            # if self.model_args['verbose']:
            print(f"{i}/{len(Xl[p])} ({p}) -> {j}/{len(Xl[q])} ({q})")

            a_Gidx = int(np.sum([len(Xl[k]) for k in range(a_component)]) + a_idx)
            b_Gidx = int(np.sum([len(Xl[k]) for k in range(b_component)]) + b_idx)
            NG[a_Gidx, b_Gidx] = shortest_distance

            if i not in M[p]:
                M[p].append(i)
            if j not in M[q]:
                M[q].append(j)            
            i = M[p].index(i)
            j = M[q].index(j)

            L.append([[p, i], [q, j], shortest_distance])
            print(M[p], i)
            print(M[q], j)
            print(f"{p}/{len(M[p])} ({i}) -> {q}/{len(M[q])} ({j})")
            # print(f"distance_check: {shortest_distance}, {sp.distance.cdist(a_coords, b_coords, metric='euclidean')} {np.linalg.norm(a_coords - b_coords)} {np.linalg.norm(Xl[i][M[i]][p] - Xl[j][M[j]][q])}")

            cc, labels = csgraph.connected_components(NG, directed=False)
        return L

    def chose_intercomponent_connections_k2(self, Xl:list[np.ndarray], M:list[list[int]], k2:int):
        """
        For each component `Xi`, chose the `k2` connections to the other components (representative points representation).

        Parameters
        ----------
            Xl : list[np.ndarray]
                Componentwise original points.
            M : list[list[int]]
                Componentwise representative points.
            k2 : int
                Size of the inter-component k-neighbourhood.

        Returns
        -------
            L : list[list[list[int]]]
                Inter-component connections.
        """

        L:list[list[list[int]]] = [] # L.append([[0, 345], [2 , 741], 34.23])
        for i, Xi in enumerate(Xl):
            Mi = M[i]
            XMi = Xi[Mi]
            for j, Xj in enumerate(Xl):
                Mj = M[j]
                XMj = Xj[Mj]
                if j != i:
                    dist_matrix = sp.distance.cdist(XMi, XMj)
                    
                    p, q = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                    p, q = int(p), int(q)
                    L.append([[i, p], [j, q], dist_matrix[p, q]])

                    connections = [L_ij for L_ij in L if L_ij[0][0] == i]
                    if len(connections) > k2:
                        connection_distances = [L_ij[2] for L_ij in connections]
                        argmax_idx = np.argmax(connection_distances)
                        L.remove(connections[argmax_idx])
        

        XMs = np.vstack([Xi[M[i]] for i, Xi in enumerate(Xl)])

        NM = get_NM(Xl, M, L)
        cc, labels = csgraph.connected_components(NM, directed=False)
        if cc > 1:
            utils.hard_warning(f"Global MVU not fully connected. Adding the shortest possible connections to merge components.")
            self.model_args['artificial_connected'] = True
            while cc > 1:
                largest_component = np.argmax(np.bincount(labels))
                largest_component_idx = np.where(labels == largest_component)[0]
                other_idx = np.where(labels != largest_component)[0]

                distances = sp.distance.cdist(XMs[largest_component_idx], XMs[other_idx])
                shortest_distance = np.min(distances)
                ab = np.where(distances == shortest_distance)
                if type(ab[0]) == np.ndarray:
                    ab = np.array(ab)[:, 0]

                a_idx, b_idx = largest_component_idx[ab[0]], other_idx[ab[1]] # index of a, b, inside the NM matrix
                print(f"{a_idx} -> {b_idx}")

                a_C, a_Cidx = 0, a_idx
                while a_Cidx > len(M[a_C]):
                    # print(f"a_Cidx: {a_Cidx} ({a_C}) -> {a_Cidx - len(M[a_C])} ({a_C + 1})")
                    a_Cidx -= len(M[a_C])
                    a_C += 1
                # print(f" Final a: {a_Cidx} ({a_C})")
                
                b_C, b_Cidx = 0, b_idx
                while b_Cidx > len(M[b_C]):
                    # print(f"b_Cidx: {b_Cidx} ({b_C}) -> {b_Cidx - len(M[b_C])} ({b_C + 1})")
                    b_Cidx -= len(M[b_C])
                    b_C += 1
                # print(f" Final b: {b_Cidx} ({b_C})")
                print(f"{a_Cidx}({a_C}) -> {b_Cidx}({b_C})")

                L.append([[a_C, a_Cidx], [b_C, b_Cidx], shortest_distance])
                NM = get_NM(Xl, M, L)
                cc, labels = csgraph.connected_components(NM, directed=False)

        return L

    def final_MVU(self, Xl:list[np.ndarray], Yl:list[np.ndarray], M:list[list[int]], L:list[list[list[int]]], d:int):
        """
        Compute the global embeddings of the representative points.
        
        Parameters
        ----------
            Xl : list[np.ndarray]
                Componentwise original points.
            Yl : list[np.ndarray]
                Componentwise local embeddings.
            M : list[list[int]]
                Componentwise representative points.
            L : list[list[list[int]]]
                Inter-component connections.

        Returns
        -------
            YM : list[np.ndarray]
                Componentwise global embeddings of the representative points.
        """
        
        NM = get_NM(Xl, M, L, Yl)
        self.NM = NM

        cc, labels = csgraph.connected_components(NM, directed=False)
        if cc > 1:
            print([len(np.where(labels == l)[0]) for l in range(cc)])
            utils.hard_warning(f"Global MVU should be fully connected (found {cc} components)")
            breakpoint()
        
        utils.stamp.print(f"*\t {self.model_args['model']}\t global MVU")
        # self.model_args['verbose'] = True
        # self.eps = 1e-15
        self.representation_threshold = 0.999
        self.id = "global"
        
        XMs = np.vstack([Xi[M[i]] for i, Xi in enumerate(Xl)])
        YMs = super().fit(XMs).transform()
        if YMs is None:
            return None

        self.representation_threshold = None
        assert XMs.shape[0] == YMs.shape[0], f"XMs.shape: {XMs.shape}, YMs.shape: {YMs.shape}"

        YM = []
        for i in range(len(Yl)):
            ptr = np.sum([len(M[k]) for k in range(i)])
            ptr = int(ptr)
            YMi = YMs[ptr:ptr+len(M[i])]
            YM.append(YMi)
        
        assert all(len(M[i]) == len(YM[i]) for i in range(len(M))), f"M: {M}, YM: {YM}"
        return YM


    def equalize_dimensions(self, Yl:list[np.ndarray]):
        """
        Equalize the dimensions of the local embeddings.
        """

        max_dim = max([Yi.shape[1] for Yi in Yl])
        return [np.hstack((Yi, np.zeros((Yi.shape[0], max_dim - Yi.shape[1])))) for Yi in Yl]

    def final_transformation(self, Yl:list[np.ndarray], YM:list[np.ndarray], M:list[list[int]]):
        """
        Parameters
        ----------
            Yl : list[np.ndarray]
                Componentwise local embeddings.
            YM : list[np.ndarray]
                Componentwise global embeddings of the representative points.
            M : list[list[int]]
                Componentwise representative points.
        Returns
        -------
            Tl : list[np.ndarray]
                Componentwise transformation positioning the local embeddings in the global embedding space.
        """
        
        Tl = []
        for i, Yi in enumerate(Yl):
            # M[i]'s position in local embeddings @ Ti      = M[i]'s position in the global embedding space
            # YMi                                 @ Tl[i]   = YM[i]
            
            
            # Ax + b = Y
            # [A, b] @ [x, 1] = Y
            # [A, b] = [X, 1]^-1 @ Y
            # T      = [X, 1]^-1 @ Y
            
            # YMi = Yi[M[i]] # X : local linearised M points
            # YM[i]          # Y : global lineasired points

            # # Ti = np.linalg.pinv(np.hstack((YMi, np.ones((len(YMi), 1)) ))) @ YM[i]
            # Ti, _, _, _ = np.linalg.lstsq(np.hstack((YMi, np.ones((len(YMi), 1)))), YM[i], rcond=None)
            # Tl.append(Ti)
            
            X = Yi[M[i]]
            Y = YM[i]
            n, D = X.shape

            X_homogeneous = np.hstack([X, np.ones((n, 1))])

            T = np.linalg.pinv(X_homogeneous) @ Y

            Tl.append(T)
            
            
            # P = Yi[M[i]]
            # Q = YM[i]

            # centroid_P = P.mean(axis=0)
            # centroid_Q = Q.mean(axis=0)

            # P_centered = P - centroid_P
            # Q_centered = Q - centroid_Q

            # H = P_centered.T @ Q_centered

            # U, S, Vt = np.linalg.svd(H)

            # R = Vt.T @ U.T
            
            # if np.linalg.det(R) < 0:
            #     Vt[-1, :] *= -1
            #     R = Vt.T @ U.T

            # s = np.sum(S) / np.sum(P_centered**2)

            # t = centroid_Q - s * R @ centroid_P

            # Tl.append([s, R, t])





        return Tl
    

    def final_final_yeet(self, Yl:list[np.ndarray], Tl:list[np.ndarray]):
        """
        Parameters
        ----------
            Yl : list[np.ndarray]
                Componentwise local embeddings.
            Tl : list[np.ndarray]
                Componentwise transformation positioning the local embeddings in the global embedding space.
        
        Returns
        -------
            Yl: list[np.ndarray]
                Componentwise embeddings.
        """

        for i, Yi in enumerate(Yl):
            # Yl[i] = np.hstack((Yi, np.ones((Yi.shape[0], 1)))) @ Tl[i]

            # s, R, t = Tl[i]
            # original_points = Yi
            # transformed_points = (s * (R @ original_points.T)).T + t
            
            Y = Yi

            m, D = Y.shape

            Y_homogeneous = np.hstack([Y, np.ones((m, 1))])

            transformed_points = Y_homogeneous @ Tl[i]
            
            
            Yl[i] = transformed_points
        return Yl





















    ########################################################
    # EXTEND ###############################################
    ########################################################

    def fit_transform(self, X:np.ndarray):
        Xs = X
        k1, k2 = self.k1, self.k2

        if 'labels' not in self.model_args:
            labels = np.zeros((Xs.shape[0], 1))
        else:
            labels = self.model_args['labels']
        Xl, Ll = self.NG(Xs, k1, labels)
        if Xl is None: # might be useless, it runs good without it, computes the global embeddings unnecessarilly
            return super().fit_transform(Xs)
        
        utils.stamp.print(f"*\t Found {len(Xl)} components: {[len(Xi) for Xi in Xl]}")
        
        labels = np.vstack(Ll)
        self.model_args['labels'] = labels
        Xs = np.vstack(Xl)
        self.model_args['X'] = Xs

        if self.model_args['photos']:
            for i, Xi in enumerate(Xl):
                fig, ax = plot.plot(Xi, c=[i] * len(Xi), c_scale=[-1, len(Xl)], block=False)
                ax.view_init(elev=5, azim=-75)
                ax.set_xticks([round(min(Xi[:, 0]), 0), round(max(Xi[:, 0]), 0)])
                ax.set_yticks([round(min(Xi[:, 1]), 0), round(max(Xi[:, 1]), 0)])
                ax.set_zticks([round(min(Xi[:, 2]), 0), round(max(Xi[:, 2]), 0)])
                ax.grid(False)
                fig.tight_layout()
                import os
                if not os.path.exists(f'images/{self.model_args["dataname"]}/'):
                    os.makedirs(f'images/{self.model_args["dataname"]}/')
                plt.savefig(f'images/{self.model_args["dataname"]}/1a.original.{i}.pdf', format='pdf', transparent=True, bbox_inches='tight')

        if self.model_args['photos']:
            color = [i for i in range(len(Xl)) for _ in range(len(Xl[i]))]
            fig, ax = plot.plot(Xs, c=color, c_scale=[-1, len(Xl)], block=False)
            ax.view_init(elev=15, azim=-60)
            ax.set_xticks([round(min(Xs[:, 0]), 0), round(max(Xs[:, 0]), 0)])
            ax.set_yticks([round(min(Xs[:, 1]), 0), round(max(Xs[:, 1]), 0)])
            ax.set_zticks([round(min(Xs[:, 2]), 0), round(max(Xs[:, 2]), 0)])
            ax.grid(False)
            plt.savefig(f'images/{self.model_args["dataname"]}/2a.original.pdf', format='pdf', transparent=True, bbox_inches='tight')


        # if self.model_args['preview']:
        #     color = [i for i in range(len(Xl)) for _ in range(len(Xl[i]))]
        #     plot.plot(Xs, c=color, block=False, title=f"{self.model_args['dataname']}({self.model_args['#neighs']}) MVU local stage")


        self.ratio = None # toggle compute independent ratio
        Yl = self.MVU_local(Xl)
        if Yl is None:
            return None

        if self.model_args['verbose']:
            print("Before transformation:")
            for i, Yi in enumerate(Yl):
                Xi, Li, Yi = Xl[i], Ll[i], Yl[i]
                print(f"Component {i}({len(Xi)}/{len(Xs)}):")
                utils.compute_measures(Xi, Yi, Li, k1)

        if self.model_args['photos']:
            for i, Yi in enumerate(Yl):
                fig, ax = plot.plot(Yi, c=[i] * len(Yi), c_scale=[-1, len(Yl)], block=False)
                # ax.view_init(elev=5, azim=95)
                ax.set_xticks([round(min(Yi[:, 0]), 0), round(max(Yi[:, 0]), 0)])
                ax.set_yticks([round(min(Yi[:, 1]), 0), round(max(Yi[:, 1]), 0)])
                ax.set_zticks([])
                ax.grid(False)
                ax.view_init(elev=90, azim=-90)
                plt.savefig(f'images/{self.model_args["dataname"]}/1b.linearised.{i}.pdf', format='pdf', transparent=True, bbox_inches='tight')


        utils.important(f"Building global MVU")

        d:list[int] = [Yi.shape[1] for Yi in Yl]
        M = [[] for _ in range(len(Yl))]
        M = self.chose_representative_points_PCA(Yl, d, M) # before any L
        # M = self.chose_representative_points_landmark(Yl, d, M)

        if self.model_args['photos']:
            for i, Yi in enumerate(Yl):

                representative_mask = np.ones(len(Yi)) * i
                representative_mask[M[i]] = -1
                fig, ax = plot.plot(Yi, c=representative_mask, c_scale=[-1, len(Yl)], block=False)
                ax.set_xticks([round(min(Yi[:, 0]), 0), round(max(Yi[:, 0]), 0)])
                ax.set_yticks([round(min(Yi[:, 1]), 0), round(max(Yi[:, 1]), 0)])
                ax.set_zticks([])
                ax.view_init(elev=90, azim=-90)
                ax.set_proj_type('ortho')
                ax.grid(False)
                plt.savefig(f'images/{self.model_args["dataname"]}/2b.linearised.representative.{i}.pdf', format='pdf', transparent=True, bbox_inches='tight')
        

                # Highlight convex hull of representative points
                try:
                    from scipy.spatial import ConvexHull
                    rep_points = Yi[M[i]]
                    # Ensure 3D for plotting on 3D axes
                    if rep_points.shape[1] < 3:
                        rep_points = np.hstack((rep_points, np.zeros((rep_points.shape[0], 3 - rep_points.shape[1]))))
                    # Compute hull in the intrinsic dimensions (with jitter if degenerate handled by QJ)
                    hull = ConvexHull(rep_points, qhull_options="Qa QJ")
                    # Draw hull edges
                    for simplex in hull.simplices:
                        verts = rep_points[simplex]
                        if len(simplex) == 2:
                            ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], color='red', linewidth=2, alpha=0.9)
                        elif len(simplex) == 3:
                            # three edges of the triangle
                            ax.plot([verts[0,0], verts[1,0]], [verts[0,1], verts[1,1]], [verts[0,2], verts[1,2]], color='red', linewidth=1.5, alpha=0.9)
                            ax.plot([verts[1,0], verts[2,0]], [verts[1,1], verts[2,1]], [verts[1,2], verts[2,2]], color='red', linewidth=1.5, alpha=0.9)
                            ax.plot([verts[2,0], verts[0,0]], [verts[2,1], verts[0,1]], [verts[2,2], verts[0,2]], color='red', linewidth=1.5, alpha=0.9)
                    # Emphasize representative points
                    ax.scatter(rep_points[:, 0], rep_points[:, 1], rep_points[:, 2], color='red', s=30, alpha=1.0)
                except Exception as _e:
                    pass
                # ax.view_init(elev=75, azim=75)
                ax.set_xticks([round(min(Yi[:, 0]), 0), round(max(Yi[:, 0]), 0)])
                ax.set_yticks([round(min(Yi[:, 1]), 0), round(max(Yi[:, 1]), 0)])
                ax.set_zticks([])
                ax.grid(False)
                ax.view_init(elev=90, azim=-90)
                plt.savefig(f'images/{self.model_args["dataname"]}/1c.linearised.representative.{i}.pdf', format='pdf', transparent=True, bbox_inches='tight')
        

        L = self.chose_intercomponent_connections_default(Xl, M) # after PCA, before iterative
        # M = self.chose_representative_points_iterative(Yl, d, M) # after default, before k2
        # L = self.chose_intercomponent_connections_k2(Xl, M, k2) # after any M

        Yl = self.equalize_dimensions(Yl) # equalize the dimensions across all the local MVUs
        YM = self.final_MVU(Xl, Yl, M, L, max(d))
        if YM is None:
            return None

        # no need to equalize dimensions between local MVUs and the global MVU
        

        if self.model_args['preview']:

            YMs = np.vstack(YM)
            color2 = [i for i in range(len(YM)) for _ in range(len(YM[i]))]

            # NM = get_NMg(Xl, M, L)
            NM = np.zeros((len(Xs), len(Xs)))
            for [i, p], [j, q], _ in L:
                a_idx = np.sum([len(Xl[k]) for k in range(i)]) + M[i][p]
                b_idx = np.sum([len(Xl[k]) for k in range(j)]) + M[j][q]
                a_idx, b_idx = int(a_idx), int(b_idx)
                NM[a_idx, b_idx] = 1
            
            NM2 = np.zeros((len(YMs), len(YMs)))
            for [i, p], [j, q], _ in L:
                a = np.sum([len(YM[k]) for k in range(i)]) + p
                b = np.sum([len(YM[k]) for k in range(j)]) + q
                a, b = int(a), int(b)
                NM2[a, b] = 1


            color = [0 if a in M[i] else 1 for i, Xi in enumerate(Xl) for a in range(len(Xi))]
            print(f"Xs: {Xs.shape}, YMs: {YMs.shape}, NM: {NM.shape}({np.count_nonzero(NM)}), NM2: {NM2.shape}({np.count_nonzero(NM2)})")
            plot.plot_two(Xs, YMs, NM, NM2, scale=True, scale2=False, c1=color, c2=color2, block=False, title=f"{self.model_args['dataname']}({self.model_args['#neighs']}) global MVU, X and Y")

        if self.model_args['photos']:
            color = [i for i in range(len(YM)) for _ in range(len(YM[i]))]
            
            # plt.clf()
            # plt.scatter(YMs[:, 0], YMs[:, 1], c=color, s=10, cmap="rainbow", vmin=0, vmax=len(YM))
            # N = get_NM(Xl, M, L, Yl)
            # for i in range(len(N)):
            #     for j in range(len(N[i])):
            #         if N[i, j] != 0:
            #             plt.plot([YMs[i, 0], YMs[j, 0]], [YMs[i, 1], YMs[j, 1]], c='red', alpha=0.5)
            # plt.axis('equal')
            # plt.savefig(f'images/2c.paralell.linearised.representative.F.pdf', format='pdf', transparent=True, bbox_inches='tight')
            # plt.show(block=False)



            fig, ax = plot.plot(YMs, get_NM(Xl, M, L, Yl), c=color, c_scale=[-1, len(YM)], block=False)
            ax.set_xticks([round(min(YMs[:, 0]), 0), round(max(YMs[:, 0]), 0)])
            ax.set_yticks([round(min(YMs[:, 1]), 0), round(max(YMs[:, 1]), 0)])
            ax.set_zticks([])
            ax.view_init(elev=90, azim=-90)
            ax.set_proj_type('ortho')
            ax.grid(False)
            plt.savefig(f'images/{self.model_args["dataname"]}/2c.linearised.representative.pdf', format='pdf', transparent=True, bbox_inches='tight')

            # component_selection = [0, 1, 3, 4, 5]
            # YM_selected = [YM[i] for i in component_selection]
            # color_selected = [[i]*len(YM_selected[component_selection.index(i)]) for i in component_selection]
            
            # YMs_selected = np.vstack(YM_selected)
            # NM_selected = NM.copy()
            # mask = np.zeros((len(NM_selected),), dtype=bool)
            # # for i in range(len(YM)), if i not in component_selection, delete the lines and column reffering to that component
            # for i in range(len(YM)):
            #     if i not in component_selection:

            #         starting_idx = sum([len(YM[k]) for k in range(i)])
            #         ending_idx = starting_idx + len(YM[i])
            #         starting_idx, ending_idx = int(starting_idx), int(ending_idx)
            #         rng = list(range(starting_idx, ending_idx))
            #         print(f"Component {i} starting index: {starting_idx}, ending index: {ending_idx}, length: {len(rng)}")
            #         mask[rng] = True

            # print(f"mask points: {np.count_nonzero(mask)}")
            # print(f"NM: {NM_selected.shape}")
            
            # NM_selected = NM_selected[mask]
            # NM_selected = NM_selected[:, mask]

            # color_selected = np.hstack(color_selected)
            # print(f"YMs: {YMs_selected.shape}")
            # print(f"NM: {NM_selected.shape}")
            # print(f"color: {color_selected.shape}")

            # plot.plot(YMs_selected, NM_selected, c=color_selected, c_scale=[0, len(YM)], block=False, title=f"{self.model_args['dataname']}({self.model_args['#neighs']}) global stage, X and Y (selected {component_selection})")
        

        utils.important(f"Final transformation")


        Tl = self.final_transformation(Yl, YM, M)
        Yl = self.final_final_yeet(Yl, Tl)

        if self.model_args['verbose']:
            print("After transformation:")
            for i, _ in enumerate(Yl):
                Xi, Li, Yi = Xl[i], Ll[i], Yl[i]
                print(f"Component {i}({len(Xi)}/{len(Xs)}):")
                utils.compute_measures(Xi, Yi, Li, k1)

            print("Final measures:")
            utils.compute_measures(Xs, np.vstack(Yl), np.vstack(Ll), k1)
            

            def compute_nn(Y:np.ndarray, L:np.ndarray, dim:int) -> np.ndarray:
                """
                Compute the nearest neighbor matrix for a given component.
                
                Parameters
                ----------
                    Y : np.ndarray
                        Embeddings.
                    L : np.ndarray
                        True labels.
                    dim : int
                        Dimension of the labels.
                
                Returns
                -------
                    i_individual_nn : np.ndarray
                        Nearest neighbor matrix.
                """

                temp_NG = utils.neigh_graph(Y, 1)
                nearest_neigh_idx = [temp_NG[p].indices[0] for p in range(Y.shape[0])]
                # print(Y.shape, L.shape, nearest_neigh_idx.shape)
                L_predicted = L[nearest_neigh_idx]
                
                individual_nn = np.zeros((dim, dim))
                for p in range(Y.shape[0]):
                    actual_label = int(L[p])
                    predicted_label = int(L_predicted[p])
                    individual_nn[actual_label, predicted_label] += 1
                return individual_nn
                



            print("Global collisions:")
            dim = int(np.max(labels)) + 1
            individual_nn = np.zeros((dim, dim))
            for Yi, Li in zip(Yl, Ll):
                individual_nn += compute_nn(Yi, Li, dim)

            Ym, Lm = np.vstack(Yl), np.vstack(Ll)
            merged_nn = compute_nn(Ym, Lm, dim)
            # print(f"diff: \n{merged_nn - individual_nn}")

            one_nn = individual_nn.copy()
            for i in range(dim):
                one_nn[i, i] = 0
            one_nn = np.sum(one_nn) / np.sum(individual_nn)
            print(f"1_NN (No-Collisions): \n{one_nn}")

            one_nn = merged_nn.copy()
            for i in range(dim):
                one_nn[i, i] = 0
            one_nn = np.sum(one_nn) / np.sum(merged_nn)
            print(f"1_NN (Actual): \n{one_nn}")
            
        # if self.model_args['overlap']:
        #     plot.overlap(Yl, self.model_args, show=)
        Ys, Ls = np.vstack(Yl), np.vstack(Ll)

        if self.model_args['preview']:
            # NM = get_NMg(Yl, M, L)

            NM = np.zeros((len(Ys), len(Ys)))
            for [i, p], [j, q], _ in L:
                a_idx = np.sum([len(Yl[k]) for k in range(i)]) + M[i][p]
                b_idx = np.sum([len(Yl[k]) for k in range(j)]) + M[j][q]
                a_idx, b_idx = int(a_idx), int(b_idx)
                NM[a_idx, b_idx] = 1
            color2 = [i for i in range(len(Yl)) for _ in range(len(Yl[i]))]
            plot.plot(Ys, NM, c=color2, block=False, title=f"{self.model_args['dataname']}({self.model_args['#neighs']}) Component separation in the global embedding space")


        if self.model_args['photos']:
            color = [i if p not in M[i] else -1 for i in range(len(Yl)) for p in range(len(Yl[i]))]

            # plt.clf()
            # plt.scatter(Ys[:, 0], Ys[:, 1], c=color, s=10, cmap="rainbow", vmin=0, vmax=len(Yl))
            # plt.axis('equal')
            # plt.savefig(f'images/2d.paralell.linearised.representative.F.pdf', format='pdf', transparent=True, bbox_inches='tight')
            # plt.show(block=False)

            fig, ax = plot.plot(Ys, NM, c=color, c_scale=[-1, len(Yl)], block=False)
            ax.set_xticks([round(min(Ys[:, 0]), 0), round(max(Ys[:, 0]), 0)])
            ax.set_yticks([round(min(Ys[:, 1]), 0), round(max(Ys[:, 1]), 0)])
            ax.set_zticks([])
            ax.grid(False)
            ax.view_init(elev=90, azim=-90)
            ax.set_proj_type('ortho')
            plt.savefig(f'images/2d.paralell.linearised.representative.pdf', format='pdf', transparent=True, bbox_inches='tight')
            
            
            # component_selection = [1, 3, 4, 5]
            # X_selected = np.vstack([Xl[i] for i in component_selection])
            # Y_selected = np.vstack([Yl[i] for i in component_selection])
            # NM_selected = NM.copy()
            # mask = np.array([False] * len(NM_selected))
            # for i in component_selection:
            #     starting_idx = sum([len(Yl[k]) for k in range(i)])
            #     ending_idx = starting_idx + len(Yl[i])
            #     starting_idx, ending_idx = int(starting_idx), int(ending_idx)
            #     rng = list(range(starting_idx, ending_idx))
            #     mask[rng] = True
            # mask = mask.astype(bool)
            # NM_selected = NM_selected[mask]
            # NM_selected = NM_selected[:, mask]

            # color = np.hstack([[i]*len(Yl[i]) for i in component_selection])
            
            # plot.plot(Y_selected, NM_selected, c=color, c_scale=[0, len(Yl)], block=False, title=f"{self.model_args['dataname']}({self.model_args['#neighs']}) final stage (selected {component_selection})")
            # plot.plot_two(X_selected, Y_selected, NM_selected, NM_selected, c1=color, c2=color, c1_scale=range(len(Xl)), c2_scale=range(len(Yl)), scale=True, scale2=True, block=False, title=f"{self.model_args['dataname']}({self.model_args['#neighs']}) final stage (selected {component_selection})")

        self.embedding_ = Ys
        self.model_args['labels'] = Ls
        return Ys