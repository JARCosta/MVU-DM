import numpy as np
from scipy.sparse import csgraph
from scipy.spatial.distance import cdist

import utils

def connect_components_sklearn(X:np.ndarray, NM:np.ndarray, cc:int, labels:np.ndarray):
    from sklearn.metrics.pairwise import pairwise_distances

    for i in range(cc):
        idx_i = np.flatnonzero(labels == i)
        Xi = X[idx_i]
        for j in range(i):
            idx_j = np.flatnonzero(labels == j)
            Xj = X[idx_j]
            
            D = pairwise_distances(Xi, Xj, n_jobs=-1)
            shortest_distance = np.min(D)
            ab = np.where(D == shortest_distance)
            a_idx, b_idx = idx_i[ab[0]], idx_j[ab[1]]
            NM[a_idx, b_idx] = shortest_distance
            # NM[b_idx, a_idx] = shortest_distance
    utils.hard_warning(f"Graph not fully connected. Added connections according to sklearn algorithm.")
    return NM





def connect_components_default(X:np.ndarray, NM:np.ndarray, cc:int, labels:np.ndarray):
    from sklearn.metrics.pairwise import pairwise_distances

    counter = 0
    while cc > 1:
        largest_component = np.argmax(np.bincount(labels))
        largest_component_idx = np.where(labels == largest_component)[0]
        other_idx = np.where(labels != largest_component)[0]

        distances = pairwise_distances(X[largest_component_idx], X[other_idx], n_jobs=-1)
        shortest_distance = np.min(distances)
        ab = np.where(distances == shortest_distance)
        a_idx, b_idx = largest_component_idx[ab[0]], other_idx[ab[1]]

        NM[a_idx, b_idx] = np.linalg.norm(X[a_idx] - X[b_idx])
        cc, labels = csgraph.connected_components(NM, directed=False)
        counter += 1
    utils.hard_warning(f"MVU not fully connected. Added {counter} connections to merge components.")
    return NM


def connect_components_k2(X:np.ndarray, NM:np.ndarray, cc:int, labels:np.ndarray):
    from sklearn.metrics.pairwise import pairwise_distances

    k2 = 2

    utils.hard_warning(f"MVU not fully connected ({cc} components). Connecting components to it's `k2` nearest neighbors.")

    Xl = [X[labels == c] for c in range(cc)]
    inter_distances = np.zeros((len(Xl), len(Xl)))
    inter_origin_idx = np.zeros((len(Xl), len(Xl))) # index of the point that is closest to the other component
    inter_target_idx = np.zeros((len(Xl), len(Xl))) # index of the point that is closest to self component

    for i, Xi in enumerate(Xl):
        for j, Xj in enumerate(Xl):
            if i != j:
                distances = pairwise_distances(Xi, Xj, n_jobs=-1)
                shortest_distance = np.min(distances)
                ab = np.where(distances == shortest_distance)
                
                inter_distances[i, j] = shortest_distance
                inter_origin_idx[i, j] = ab[0]
                inter_target_idx[i, j] = ab[1]
    
    L = []
    for i, Xi in enumerate(Xl):
        closest_components_idx = np.argsort(inter_distances[i])[:k2]
        for j in closest_components_idx:
            L.append([[i, inter_origin_idx[i, j]], [j, inter_target_idx[i, j]], inter_distances[i, j]])
    
    for [i, p], [j, q], dist in L:
        a_idx = np.sum([len(Xl[k]) for k in range(i)]) + p
        b_idx = np.sum([len(Xl[k]) for k in range(j)]) + q
        a_idx, b_idx = int(a_idx), int(b_idx)
        NM[a_idx, b_idx] = dist
    
    cc, labels = csgraph.connected_components(NM, directed=False)
    if cc > 1:
        utils.hard_warning(f"MVU_k2 not fully connected ({cc} components). Using default method to merge components.")

        counter = 0
        while cc > 1:
            largest_component = np.argmax(np.bincount(labels))
            largest_component_idx = np.where(labels == largest_component)[0]
            other_idx = np.where(labels != largest_component)[0]

            distances = pairwise_distances(X[largest_component_idx], X[other_idx], n_jobs=-1)
            shortest_distance = np.min(distances)
            ab = np.where(distances == shortest_distance)
            a_idx, b_idx = largest_component_idx[ab[0]], other_idx[ab[1]]
            NM[a_idx, b_idx] = shortest_distance
            
            cc, labels = csgraph.connected_components(NM, directed=False)
            counter += 1

    return NM