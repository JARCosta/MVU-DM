import numpy as np
from scipy.spatial import ConvexHull
import pypoman

def convex_hull_inequalities(points):
    """
    Given a set of points, compute convex hull inequalities Hx <= b
    using scipy's ConvexHull.
    """
    hull = ConvexHull(points)
    H = hull.equations[:, :-1]   # normals
    b = -hull.equations[:, -1]   # offsets
    return H, b

def convex_hull_intersection(points_A, points_B):
    """
    Compute intersection polytope of convex hulls of two point sets.
    Returns vertices of the intersection (possibly empty).
    """
    H_A, b_A = convex_hull_inequalities(points_A)
    H_B, b_B = convex_hull_inequalities(points_B)

    # Combine inequalities
    H = np.vstack((H_A, H_B))
    b = np.hstack((b_A, b_B))

    # Compute vertices of intersection polytope
    vertices = pypoman.compute_polytope_vertices(H, b)
    return np.array(vertices)

def compute_polytope_volume(vertices):
    """
    Compute the volume of a polytope given its vertices.
    Uses convex hull volume computation.
    """
    if len(vertices) < vertices.shape[1] + 1:  # Need at least dim+1 points for volume
        return 0.0
    
    try:
        # Compute convex hull of the intersection vertices
        hull = ConvexHull(vertices)
        return hull.volume
    except:
        return 0.0

def project_to_3d(points_nd, projection_matrix=None):
    """
    Project n-dimensional points to 3D using a random projection matrix.
    """
    n_dims = points_nd.shape[1]
    if n_dims <= 3:
        return points_nd
    
    if projection_matrix is None:
        np.random.seed(42)  # Fixed seed for reproducible projections
        projection_matrix = np.random.randn(3, n_dims)
        # Normalize the projection matrix
        projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=1, keepdims=True)
    
    return points_nd @ projection_matrix.T

def project_to_2d(points_nd, projection_matrix=None):
    """
    Project n-dimensional points to 2D using a random projection matrix.
    """
    n_dims = points_nd.shape[1]
    if n_dims <= 2:
        return points_nd
    
    if projection_matrix is None:
        np.random.seed(43)  # Different seed for 2D projection
        projection_matrix = np.random.randn(2, n_dims)
        # Normalize the projection matrix
        projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=1, keepdims=True)
    
    return points_nd @ projection_matrix.T

def generate_random_points(n_points, n_dims, center, std=1.0):
    """
    Generate random points in n-dimensional space.
    """
    return np.random.randn(n_points, n_dims) * std + np.array(center)

def visualize_intersection(points_A, points_B, intersection_vertices, n_dims):
    """
    Visualize the intersection based on the number of dimensions.
    """
    import matplotlib.pyplot as plt
    
    if n_dims == 2:
        visualize_2d(points_A, points_B, intersection_vertices)
    elif n_dims == 3:
        visualize_3d(points_A, points_B, intersection_vertices)
    else:
        visualize_nd(points_A, points_B, intersection_vertices, n_dims)

def visualize_2d(points_A, points_B, intersection_vertices):
    """
    Visualize 2D intersection.
    """
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    
    hull_A = ConvexHull(points_A)
    hull_B = ConvexHull(points_B)
    
    plt.figure(figsize=(8, 8))
    
    # Plot hulls
    for simplex in hull_A.simplices:
        plt.plot(points_A[simplex, 0], points_A[simplex, 1], 'r-', alpha=0.7)
    for simplex in hull_B.simplices:
        plt.plot(points_B[simplex, 0], points_B[simplex, 1], 'b-', alpha=0.7)
    
    # Plot intersection vertices
    if len(intersection_vertices) > 0:
        plt.scatter(intersection_vertices[:, 0], intersection_vertices[:, 1], 
                   color='green', s=100, label='Intersection')
    
    # Plot original points
    plt.scatter(points_A[:, 0], points_A[:, 1], color='red', alpha=0.6, s=20, label='Points A')
    plt.scatter(points_B[:, 0], points_B[:, 1], color='blue', alpha=0.6, s=20, label='Points B')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Convex Hull Intersection')
    plt.legend()
    plt.axis('equal')
    plt.show()

def visualize_3d(points_A, points_B, intersection_vertices):
    """
    Visualize 3D intersection.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial import ConvexHull
    
    hull_A = ConvexHull(points_A)
    hull_B = ConvexHull(points_B)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot hulls
    for simplex in hull_A.simplices:
        vertices = points_A[simplex]
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       alpha=0.3, color='red', label='Hull A' if simplex[0] == 0 else "")
    
    for simplex in hull_B.simplices:
        vertices = points_B[simplex]
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       alpha=0.3, color='blue', label='Hull B' if simplex[0] == 0 else "")
    
    # Plot intersection vertices
    if len(intersection_vertices) > 0:
        ax.scatter(intersection_vertices[:, 0], intersection_vertices[:, 1], intersection_vertices[:, 2], 
                  color='green', s=100, label='Intersection')
    
    # Plot original points
    ax.scatter(points_A[:, 0], points_A[:, 1], points_A[:, 2], 
              color='red', alpha=0.6, s=20, label='Points A')
    ax.scatter(points_B[:, 0], points_B[:, 1], points_B[:, 2], 
              color='blue', alpha=0.6, s=20, label='Points B')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Convex Hull Intersection')
    ax.legend()
    plt.show()

def visualize_nd(points_A, points_B, intersection_vertices, n_dims):
    """
    Visualize n-dimensional intersection using projections and statistics.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Project to 3D and 2D for visualization
    points_A_3d = project_to_3d(points_A)
    points_B_3d = project_to_3d(points_B)
    points_A_2d = project_to_2d(points_A)
    points_B_2d = project_to_2d(points_B)
    
    if len(intersection_vertices) > 0:
        intersection_3d = project_to_3d(intersection_vertices)
        intersection_2d = project_to_2d(intersection_vertices)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 10))
    
    # 3D projection
    ax1 = fig.add_subplot(131, projection='3d')
    hull_A_3d = ConvexHull(points_A_3d)
    hull_B_3d = ConvexHull(points_B_3d)
    
    for simplex in hull_A_3d.simplices:
        vertices = points_A_3d[simplex]
        ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                         alpha=0.3, color='red', label='Hull A (3D Proj)' if simplex[0] == 0 else "")
    
    for simplex in hull_B_3d.simplices:
        vertices = points_B_3d[simplex]
        ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                         alpha=0.3, color='blue', label='Hull B (3D Proj)' if simplex[0] == 0 else "")
    
    if len(intersection_vertices) > 0:
        ax1.scatter(intersection_3d[:, 0], intersection_3d[:, 1], intersection_3d[:, 2], 
                    color='green', s=100, label='Intersection (3D Proj)')
    
    ax1.scatter(points_A_3d[:, 0], points_A_3d[:, 1], points_A_3d[:, 2], 
                color='red', alpha=0.6, s=20, label='Points A (3D Proj)')
    ax1.scatter(points_B_3d[:, 0], points_B_3d[:, 1], points_B_3d[:, 2], 
                color='blue', alpha=0.6, s=20, label='Points B (3D Proj)')
    
    ax1.set_xlabel('X (3D Proj)')
    ax1.set_ylabel('Y (3D Proj)')
    ax1.set_zlabel('Z (3D Proj)')
    ax1.set_title(f'{n_dims}D → 3D Projection')
    ax1.legend()
    
    # 2D projection
    ax2 = fig.add_subplot(132)
    hull_A_2d = ConvexHull(points_A_2d)
    hull_B_2d = ConvexHull(points_B_2d)
    
    for simplex in hull_A_2d.simplices:
        ax2.plot(points_A_2d[simplex, 0], points_A_2d[simplex, 1], 'r-', alpha=0.7)
    for simplex in hull_B_2d.simplices:
        ax2.plot(points_B_2d[simplex, 0], points_B_2d[simplex, 1], 'b-', alpha=0.7)
    
    if len(intersection_vertices) > 0:
        ax2.scatter(intersection_2d[:, 0], intersection_2d[:, 1], 
                   color='green', s=100, label='Intersection (2D Proj)')
    
    ax2.scatter(points_A_2d[:, 0], points_A_2d[:, 1], color='red', alpha=0.6, s=20, label='Points A (2D Proj)')
    ax2.scatter(points_B_2d[:, 0], points_B_2d[:, 1], color='blue', alpha=0.6, s=20, label='Points B (2D Proj)')
    
    ax2.set_xlabel('X (2D Proj)')
    ax2.set_ylabel('Y (2D Proj)')
    ax2.set_title(f'{n_dims}D → 2D Projection')
    ax2.legend()
    ax2.axis('equal')
    
    # Correlation matrix
    ax3 = fig.add_subplot(133)
    all_points = np.vstack([points_A, points_B])
    correlation_matrix = np.corrcoef(all_points.T)
    
    dimensions = [f'D{i+1}' for i in range(n_dims)]
    im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax3.set_xticks(range(n_dims))
    ax3.set_yticks(range(n_dims))
    ax3.set_xticklabels(dimensions)
    ax3.set_yticklabels(dimensions)
    ax3.set_title(f'{n_dims}D Data Correlation Matrix')
    
    for i in range(n_dims):
        for j in range(n_dims):
            ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax3)
    plt.tight_layout()
    plt.show()

# ---------------- Example ----------------

def main(n_dims=4, n_points=30):
    """
    Main function to run intersection analysis for any number of dimensions.
    """
    print(f"Running {n_dims}D intersection analysis...")
    
    # Generate random points in n-dimensional space
    np.random.seed(0)
    center_A = [0] * n_dims
    center_B = [1] * n_dims
    
    points_A = generate_random_points(n_points, n_dims, center_A)
    points_B = generate_random_points(n_points, n_dims, center_B)
    
    print(f"Generated {n_points} points in {n_dims}D space")
    print(f"Points A centered at: {center_A}")
    print(f"Points B centered at: {center_B}")
    
    # Compute intersection
    intersection_vertices = convex_hull_intersection(points_A, points_B)
    
    print(f"\n{n_dims}D Intersection vertices:\n", intersection_vertices)
    print(f"Number of intersection vertices: {len(intersection_vertices)}")
    
    # Compute volume if intersection exists
    if len(intersection_vertices) > 0:
        hull_volume = compute_polytope_volume(intersection_vertices)
        print(f"\nVolume Analysis:")
        print(f"Convex Hull Volume: {hull_volume:.6f}")
    
    # Visualize based on dimensions
    visualize_intersection(points_A, points_B, intersection_vertices, n_dims)
    
    # Print statistics
    print(f"\n{n_dims}D Data Statistics:")
    print("Points A - Mean:", np.mean(points_A, axis=0))
    print("Points A - Std:", np.std(points_A, axis=0))
    print("Points B - Mean:", np.mean(points_B, axis=0))
    print("Points B - Std:", np.std(points_B, axis=0))
    
    if len(intersection_vertices) > 0:
        print("Intersection - Mean:", np.mean(intersection_vertices, axis=0))
        print("Intersection - Std:", np.std(intersection_vertices, axis=0))

if __name__ == "__main__":
    # You can change these parameters to test different dimensions
    main(n_dims=4, n_points=30)
    
    # Uncomment to test other dimensions:
    # main(n_dims=2, n_points=30)  # 2D
    # main(n_dims=3, n_points=30)  # 3D
    # main(n_dims=5, n_points=30)  # 5D
    # main(n_dims=6, n_points=30)  # 6D
