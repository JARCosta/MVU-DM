import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import random

import utils

def _plot_data(ax:Axes, data:np.ndarray, NM:np.ndarray=np.zeros((1,1)), c="blue", c_scale=None):

    data = data.copy()

    connections = [[a_idx, b_idx] for a_idx in range(NM.shape[0]) for b_idx in range(NM.shape[1]) if NM[a_idx][b_idx] != 0]

    x = data.T[0]
    if not data.shape[1] > 1:
        data = np.hstack((data, np.ones((data.shape[0], 1)) * random.normalvariate(0, 1e-10)))
    y = data.T[1]
    if not data.shape[1] > 2:
        data = np.hstack((data, np.ones((data.shape[0], 1)) * random.normalvariate(0, 1e-10)))
    z = data.T[2]

    if type(c) != str:
        cmap = "rainbow"
        assert len(c) == len(data), f"len(c) != len(data): {len(c)} != {len(data)}"
    else:
        cmap = None
    if c_scale is not None:
        scatter = ax.scatter(x, y, z, c=c, cmap=cmap, label='Original Data', s=20, alpha=0.2, vmin=c_scale[0], vmax=c_scale[1])
    else:
        scatter = ax.scatter(x, y, z, c=c, cmap=cmap, label='Original Data', s=20, alpha=0.2)
    for a_idx, b_idx in connections:
        a, b = data[a_idx], data[b_idx]
        x_line = [a[0], b[0]]
        y_line = [a[1], b[1]]
        z_line = [a[2], b[2]]
        ax.plot(x_line, y_line, z_line, color="blue", alpha=0.2)
    # ax.set_xlabel(f"D1")
    # ax.set_ylabel(f"D2")
    # ax.set_zlabel(f"D3")
    # axs.tick_params(color="white", labelcolor="white")
    # plt.colorbar(scatter, ax=ax).set_alpha(1)



def plot_scales(data:np.ndarray, NM:np.ndarray=np.zeros((1,1)), c="blue", c_scale=None, block=True, title="", legend=""):
    data = data.real
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})
    fig.text(.1, .1, str(legend).replace(", ", "\n"))
    fig.suptitle(title)

    try:
        _plot_data(axs[0], data, NM, c, c_scale)
    except:
        utils.warning("Error plotting data")
        _plot_data(axs[0], data, NM)

    axs[0].set_aspect("equal", adjustable="box")
    
    _plot_data(axs[1], data, NM, c, c_scale)

    plt.show(block=block)

def plot_two(
    data:np.ndarray, data2:np.ndarray,
    NM:np.ndarray=np.zeros((1,1)), NM2:np.ndarray=np.zeros((1,1)),
    c1="blue", c2="blue",
    c1_scale=None, c2_scale=None,
    scale=True, scale2=True,
    block=True, title="", legend=""):
    
    data, data2 = data.real, data2.real
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})
    fig.text(.1, .1, str(legend).replace(", ", "\n"))
    fig.suptitle(title)
    
    _plot_data(axs[0], data, NM, c1, c1_scale)
    axs[0].set_aspect("equal", adjustable="box") if scale else None
    
    _plot_data(axs[1], data2, NM2, c2, c2_scale)
    axs[1].set_aspect("equal", adjustable="box") if scale2 else None
    
    plt.show(block=block)
    
    

def plot(data:np.ndarray, NM:np.ndarray=np.zeros((1,1)), c="blue", c_scale=None, scale=True, block=True, title="", legend=""):
    data = data.real
    fig, axs = plt.subplots(1, 1, figsize=(10, 5), subplot_kw={'projection': '3d'})
    fig.text(.1, .1, str(legend).replace(", ", "\n"))
    fig.suptitle(title)

    _plot_data(axs, data, NM, c, c_scale)
    axs.set_aspect("equal", adjustable="box") if scale else None

    plt.show(block=block)
    return fig, axs
    

def plot_array(data:list[np.ndarray], NM:list[np.ndarray], c="blue", block=True, title="", legend=""):

    fig = plt.figure()
    fig.text(.1, .1, str(legend).replace(", ", "\n"))
    fig.suptitle(title)
    
    ax = fig.add_subplot(111, projection='3d')
    scatter_objects = []
    for i in range(len(data)):
        data[i] = data[i].real
        connections = [[a_idx, b_idx] for a_idx in range(NM[i].shape[0]) for b_idx in range(NM[i].shape[1]) if NM[i][a_idx][b_idx] != 0]
        x = data[i].T[0]
        y = data[i].T[1] if data[i].shape[1] > 1 else random.normalvariate(0, 0.001)
        z = data[i].T[2] if data[i].shape[1] > 2 else random.normalvariate(0, 0.001)
        scatter = ax.scatter(x, y, z, c=c, label='Original Data', s=20, alpha=0.4)
        scatter_objects.append(scatter)
        for a_idx, b_idx in connections:
            a, b = data[i][a_idx], data[i][b_idx]
            x = [a[0], b[0]]
            y = [a[1], b[1]] if data[i].shape[1] > 1 else [random.normalvariate(0, 0.001), random.normalvariate(0, 0.001)]
            z = [a[2], b[2]] if data[i].shape[1] > 2 else [random.normalvariate(0, 0.001), random.normalvariate(0, 0.001)]
            ax.plot(x, y, z, color="blue", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Component 1")
    ax.set_ylabel(f"Component 2")
    ax.set_zlabel(f"Component 3")
    ax.set_title(title)
    # ax.tick_params(color="white", labelcolor="white")
    plt.show(block=block)




from scipy.spatial import ConvexHull

def overlap_MonteCarlo_hull_based(Yl:list[np.ndarray], preview:bool=False) -> float:
    """
    Calculate the overlap between two sets of points.
    Percentage of points inside A's hull that are also inside B's hull.
    """

    def is_inside_hull(points: np.ndarray, hull: ConvexHull):
        # hull.equations has shape (n_facets, d+1) where the last column is the constant term
        # We need to compute: hull.equations[:, :-1] @ point + hull.equations[:, -1] <= 0
        try:
            error = 1e-6
            return np.all(np.dot(points, hull.equations[:, :-1].T) + hull.equations[:, -1] <= error, axis=1)
        except ValueError as e:
            utils.hard_warning(f"Error in is_inside_hull: {e}")
            return np.zeros(len(points), dtype=bool)

    def generate_points_inside_hull(n_points: int, hull: ConvexHull):
        # Efficient rejection sampling with vectorized operations
        vertices = hull.points[hull.vertices]
        
        # Use tighter bounding box based on actual vertices
        min_bound = np.min(vertices, axis=0)
        max_bound = np.max(vertices, axis=0)

        # print(f"min_bound: {min_bound}")
        # print(f"max_bound: {max_bound}")
        
        # Generate points in batches for efficiency
        batch_size = min(1000, n_points)
        points_list = []
        
        loop_count = 0
        while len(points_list) < n_points and loop_count < 1000:
            utils.stamp.print(f"*\t loop_count: {loop_count}, len(points_list): {len(points_list)}", end="\r")
            loop_count += 1
            # Generate batch of candidate points
            remaining = n_points - len(points_list)
            current_batch = min(batch_size, remaining)
            
            # Generate random points in bounding box
            candidates = np.random.uniform(min_bound, max_bound, size=(current_batch, len(min_bound)))
            
            # Vectorized check for points inside hull
            inside_mask = is_inside_hull(candidates, hull)
            
            # Keep only points inside the hull
            points_list.extend(candidates[inside_mask])
        print()

        if len(points_list) > 0:
            return np.array(points_list[:n_points])
        else:
            utils.hard_warning("Couldn't generate points inside hull")
            return np.zeros((0, vertices.shape[1]))
    
    overlap = np.zeros((len(Yl), len(Yl)))
    for i in range(len(overlap)):
        if Yl[i].shape[1] >= Yl[i].shape[0]:
            utils.hard_warning(f"Can't compute Y[{i}] hull because it has more dimensions than points")
            continue
        hull_i = ConvexHull(Yl[i], qhull_options="Qa QJ")
        utils.stamp.print_set(f"*\t hull_{i} computed {Yl[i].shape}")

        mask = np.zeros(len(Yl[i]))
        mask[hull_i.vertices] = 1
        print(f"hull_{i} vertices/points: {np.count_nonzero(mask)}/{len(Yl[i])}")

        n_points = 2000
        Y_A_random = generate_points_inside_hull(n_points, hull_i)
        

        # from matplotlib import pyplot as plt
        # fig = plt.figure(figsize=(20, 10))
        # ax1 = fig.add_subplot(111, projection='3d')
        
        # points_A = np.vstack([Yl[i], Y_A_random])
        # mask = np.array([0]*len(Yl[i]) + [1]*len(Y_A_random))
        
        # import models
        # points_A_3d = models.pca.PCA(model_args, 3).fit_transform(points_A)
        # hull_A_3d = ConvexHull(points_A_3d)

        # print(len(hull_A_3d.simplices))
        # for simplex in hull_A_3d.simplices:
        #     vertices = points_A_3d[simplex]
        #     ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
        #                     alpha=0.3, color='red', label='Hull A (3D Proj)' if simplex[0] == 0 else "")

        # ax1.scatter(Yl[i][:, 0], Yl[i][:, 1], Yl[i][:, 2], 
        #     color='red', alpha=0.6, s=20, label='Points A (3D Proj)')
        # ax1.scatter(Y_A_random[:, 0], Y_A_random[:, 1], Y_A_random[:, 2], 
        #     color='blue', alpha=0.6, s=20, label='Generated points')

        assert len(Yl[i]) == len(hull_i.points)

        A_points = np.vstack([Yl[i], Y_A_random]) if Y_A_random.shape[0] > 0 else Yl[i]

        for j in range(len(overlap)):
            if i != j:
                if Yl[j].shape[1] >= Yl[j].shape[0]:
                    utils.hard_warning(f"Can't compute Y[{j}] hull because it has more dimensions than points")
                    continue
                hull_j = ConvexHull(Yl[j], qhull_options="Qa QJ")
                

                collisions_count = 0
                for k in range(int(np.sqrt(A_points.shape[0]))+1):
                    start_idx = k * int(np.sqrt(A_points.shape[0]))
                    end_idx = (k + 1) * int(np.sqrt(A_points.shape[0]))
                    A_points_split = A_points[start_idx:end_idx]
                    collisions_count += np.sum(is_inside_hull(A_points_split, hull_j))


                    
                overlap[i, j] =  collisions_count / len(A_points)

                if preview and overlap[i, j] > 0:
                    Y_B_random = generate_points_inside_hull(n_points, hull_j)

                    B_points = np.vstack([Yl[j], Y_B_random]) if Y_B_random.shape[0] > 0 else Yl[j]

                    c2 = [0.1 if i not in hull_j.vertices else 0 for i in range(len(hull_j.points))]
                    c2 = c2 + [0.5] * len(Y_B_random)
                    
                    assert len(Yl[j]) == len(hull_j.points)

                    c1 = [0.9 if i not in hull_i.vertices else 1 for i in range(len(hull_i.points))]
                    c1 = c1 + [0.5] * len(Y_A_random)

                    data = np.vstack([A_points, B_points])
                    c = c1 + c2
                    c = np.hstack([i * np.ones(len(A_points)), j * np.ones(len(B_points))])

                    plot(data, c=np.array(c), c_scale=[0, len(Yl)], scale=False, title=f"Hull {i} and {j} overlap ({overlap[i, j]:.2f})", block=False)
    return overlap

def overlap_hull_volume_based(Yl:list[np.ndarray]):
    """
    Calculate the overlap between two sets of points.
    Volume in the Convex Hull of the intersection of the two sets of points.
    """

    def convex_hull_inequalities(points:np.ndarray):
        """
        Given a set of points, compute convex hull inequalities Hx <= b
        using scipy's ConvexHull.
        """
        # Round data to reduce hull complexity (crucial for high-dimensional data)
        if points.shape[1] > 3:  # Only for high-dimensional data
            # Use very aggressive rounding to drastically reduce facets
            # Start with 2 decimal places
            points_rounded = np.round(points, 2)
            points_rounded = np.unique(points_rounded, axis=0)
            
            # If still too many points, round even more aggressively
            if len(points_rounded) > 500:
                points_rounded = np.round(points, 1)
                points_rounded = np.unique(points_rounded, axis=0)
                print(f"DEBUG: Aggressive rounding: {len(points)} -> {len(points_rounded)} unique points")
            else:
                print(f"DEBUG: After rounding: {len(points)} -> {len(points_rounded)} unique points")
        else:
            points_rounded = points
        
        hull = ConvexHull(points_rounded, qhull_options="Qa QJ")
        H = hull.equations[:, :-1]   # normals
        b = -hull.equations[:, -1]   # offsets
        
        print(f"DEBUG: Hull {len(points)} points -> {len(hull.vertices)} vertices, {len(hull.equations)} facets")
        print(f"DEBUG: H shape: {H.shape}, b shape: {b.shape}")
        
        # If still too many facets, use a simplified approach
        if len(hull.equations) > 10000:
            print(f"DEBUG: WARNING: Too many facets ({len(hull.equations)}), using simplified hull")
            # Use a subset of points to create a simpler hull
            indices = np.linspace(0, len(points_rounded)-1, min(200, len(points_rounded)), dtype=int)
            points_simple = points_rounded[indices]
            hull_simple = ConvexHull(points_simple, qhull_options="Qa QJ")
            H = hull_simple.equations[:, :-1]
            b = -hull_simple.equations[:, -1]
            print(f"DEBUG: Simplified hull: {len(hull_simple.vertices)} vertices, {len(hull_simple.equations)} facets")
        
        return H, b

    # def convex_hull_intersection(H_A, b_A, H_B, b_B):
    #     """
    #     Compute intersection polytope of convex hulls of two point sets.
    #     Returns vertices of the intersection (possibly empty).
    #     """
    #     # Combine inequalities
    #     H = np.vstack((H_A, H_B))
    #     b = np.hstack((b_A, b_B))

    #     print(f"DEBUG: Intersection - H_A: {H_A.shape}, H_B: {H_B.shape} -> Combined: {H.shape}")

    #     # Compute vertices of intersection polytope
    #     try:
    #         vertices = pypoman.compute_polytope_vertices(H, b)
    #         print(f"DEBUG: Pypoman returned {len(vertices)} vertices")
    #         return np.array(vertices)
    #     except Exception as e:
    #         print(f"DEBUG: Pypoman failed: {e}")
    #         return np.empty((0, H.shape[1]))
    
    def convex_hull_intersection_lp(H_A, b_A, H_B, b_B, tol=1e-9):
        """
        Faster intersection via:
        1) Concatenate and prune redundant halfspaces
        2) Try SciPy HalfspaceIntersection (C) with Chebyshev-center interior point
        3) Fallback to pypoman if SciPy fails
        Returns vertices as (m x d) array (possibly empty).
        """
        import numpy as np
        from numpy.linalg import norm
        from scipy.optimize import linprog
        from scipy.spatial import HalfspaceIntersection

        H = np.vstack((H_A, H_B))
        b = np.hstack((b_A, b_B))

        # Normalize rows to stabilize redundancy check
        row_norms = np.maximum(norm(H, axis=1, keepdims=True), tol)
        Hn = H / row_norms
        bn = b / row_norms.ravel()

        # 1) Redundancy pruning: keep constraint i only if removing it enlarges feasible set
        # Do quick LP test: maximize violation of constraint i subject to others
        keep = np.ones(len(bn), dtype=bool)
        d = Hn.shape[1]
        # Small heuristic cap to avoid O(m^2) blowup for very large m
        max_checks = 2000
        indices = range(len(bn)) if len(bn) <= max_checks else np.random.choice(len(bn), max_checks, replace=False)
        for i in indices:
            if not keep[i]:
                continue
            A_ub = np.delete(Hn, i, axis=0)
            b_ub = np.delete(bn, i, axis=0)
            # Maximize Hn[i]·x - bn[i]  <=> minimize -(Hn[i]·x - bn[i])
            res = linprog(c=-(Hn[i]), A_ub=A_ub, b_ub=b_ub, bounds=[(-np.inf, np.inf)]*d, method="highs")
            if res.success and Hn[i] @ res.x <= bn[i] + tol:
                keep[i] = False
        Hn = Hn[keep]
        bn = bn[keep]
        utils.stamp.print_set(f"*\t #constraints: {len(Hn)}")

        # 2) Find interior point via Chebyshev center (maximize r s.t. Hx + ||H_i|| r <= b)
        # After normalization, ||H_i|| = 1, so constraints become Hn x + r <= bn
        c = np.zeros(d + 1); c[-1] = -1.0  # maximize r => minimize -r
        A_ub = np.hstack([Hn, np.ones((Hn.shape[0], 1))])
        b_ub = bn.copy()
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[(-np.inf, np.inf)]*d + [(0, np.inf)], method="highs")
        if not res.success or res.x[-1] < tol:
            utils.hard_warning("SciPy linprog failed. Returning empty array.")
            # Empty or degenerate: no intersection or zero-volume
            return np.empty((0, d))

        interior_pt = res.x[:d]

        # 3) Use SciPy HalfspaceIntersection
        try:
            # HalfspaceIntersection expects A x + b' <= 0; convert Hn x <= bn -> A=Hn, b'= -bn
            hs = np.hstack([Hn, -bn[:, None]])
            hs_int = HalfspaceIntersection(hs, interior_pt)
            verts = np.array(hs_int.intersections)
            if verts.size == 0:
                utils.hard_warning("SciPy HalfspaceIntersection failed. Returning empty array.")
                return np.empty((0, d))
            # Optional: deduplicate near-duplicate vertices
            # using a simple rounding tolerance
            uniq = np.unique(np.round(verts, 10), axis=0)
            return uniq
        except Exception:
            utils.hard_warning("SciPy HalfspaceIntersection failed. Falling back to pypoman.")
            # 4) Fallback to pypoman (slower)
            import pypoman
            vertices = pypoman.compute_polytope_vertices(H, b)
            return np.array(vertices)

    def compute_polytope_volume(vertices):
        """
        Compute the volume of a polytope given its vertices.
        Uses convex hull volume computation.
        """
        print(f"DEBUG: Computing volume for {len(vertices)} vertices in {vertices.shape[1]}D")
        
        if len(vertices) < vertices.shape[1] + 1:  # Need at least dim+1 points for volume
            print(f"DEBUG: Not enough vertices for volume")
            return 0.0
        
        try:
            # Compute convex hull of the intersection vertices
            hull = ConvexHull(vertices, qhull_options="Qa QJ")
            volume = hull.volume
            print(f"DEBUG: Volume computed: {volume}")
            return volume
        except Exception as e:
            print(f"DEBUG: Volume computation failed: {e}")
            return 0.0


    overlap = np.zeros((len(Yl), len(Yl)))
    for i in range(len(Yl)):
        H_A, b_A = convex_hull_inequalities(Yl[i])
        utils.stamp.print_set(f"*\t H_A: {H_A.shape}, b_A: {b_A.shape}")

        for j in range(len(Yl)):
            if i != j:
                H_B, b_B = convex_hull_inequalities(Yl[j])
                utils.stamp.print_set(f"*\t H_B: {H_B.shape}, b_B: {b_B.shape}")

                # Fast volume estimation
                intersection_vertices = convex_hull_intersection_lp(H_A, b_A, H_B, b_B)
                utils.stamp.print_set(f"*\t #vertices: {len(intersection_vertices)}")
                if len(intersection_vertices) > 0:
                    hull_volume = compute_polytope_volume(intersection_vertices)
                    overlap[i, j] = hull_volume
                else:
                    utils.warning("no collision")

                print(f"overlap between component {i} and {j}: {overlap[i, j]}")
    return overlap

def overlap(Yl:list[np.ndarray], model_args:dict, show:bool=False):

    print("Component overlap:")
    def prepare_for_hull(points):
        """Add small perturbations to degenerate dimensions to make hull computation possible"""
        
        for dim in range(points.shape[1]):
            if np.allclose(points[:, dim], points[0, dim]):
                points[:, dim] += np.random.normal(0, 1e-8, len(points))
        return points
    
    


    Yl_prepared = [prepare_for_hull(Yi.copy()) for Yi in Yl]
    # overlap = overlap_hull_volume_based(Yl_prepared)
    overlap = overlap_MonteCarlo_hull_based(Yl_prepared, model_args['preview'])

    print(f"overlap:")
    for overlap_i in overlap:
        print(list([float(overlap_ij) for overlap_ij in overlap_i]))
    with open(f"overlap.csv", "w") as f:
        for overlap_i in overlap:
            txt = ""
            for overlap_ij in overlap_i:
                txt += f"{overlap_ij},"
            txt = txt[:-1] + "\n"
            f.write(txt)
    R, C = overlap.shape
    for s in range(R + C - 1): # s is the sum of row and column indices
        for r in range(R):
            c = s - r
            if 0 <= c < C:

                if overlap[r, c] > 0:
                    print(f"component {r} and {c} overlap: {overlap[r, c]}")
    overlap_sum = np.sum(overlap)
    
    import matplotlib.pyplot as plt
    # plt.clf()
    overlap[np.where(overlap == 0)] = np.nan
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    heatmap = ax.imshow(overlap, cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(f"{model_args['dataname']}({model_args['#neighs']}) Overlap (sum={overlap_sum:.2f})")
    ax.set_xticks(range(R), [f"C{i}" for i in range(R)])
    ax.set_yticks(range(C), [f"C{i}" for i in range(C)])
    plt.colorbar(heatmap, label="Overlap", ticks=[0, 0.25, 0.5, 0.75, 1])
    if show:
        plt.savefig(f"{model_args['model']}({model_args['#neighs']})_overlap.png")
        plt.show(block=show)
