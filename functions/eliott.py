import numpy as np
from perlin_noise import PerlinNoise # `pip install perlin-noise`
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm # `pip install tqdm`
import plotly.graph_objects as go # `pip install plotly`
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import numba
from numba import jit, prange
from scipy.stats import gaussian_kde

def create_perlin_noise_grid(grid: np.array, height: float = 2.0, scale: float = 0.5, octaves: int = 1, 
                               z_height: float=0, random_seed: int = 42, save: str='') -> np.array:
    """
    Generate a 3D grid of points by applying Perlin noise to a 2D grid.
    
    Parameters:
    -----------
    grid : np.array
        A 2D numpy array of shape (N, 2) containing the (x, y) coordinates of the points 
        to which noise will be applied.
    height : float, default=2.0
        The height by which the noise is scaled.
    scale : float, default=0.5
        Controls the frequency of the noise (frequency control).
    z_height : float, default=0
        The base z value to be added to each point.
    random_seed : int, default=42
        Global seed shift to alter the noise pattern.
    save : str, default=''
        If provided, the function will save the generated points to the specified file path.

    Returns:
    --------
    np.array
        A 3D numpy array of shape (N, 3) where each row is a point (x, y, perlin_z + height),
        with perlin_z computed via Perlin noise.
    """
    # Initialize Perlin noise generator
    noise = PerlinNoise(octaves=octaves, seed=random_seed)
    
    # Prepare an array to hold the output points
    output = np.empty((grid.shape[0], 3))
    
    # Iterate over each point in the input grid
    for i, (x, y) in enumerate(grid):
        # Scale the coordinates
        x_scaled = x * scale
        y_scaled = y * scale

        # Calculate the Perlin noise value for the point.
        perlin_z = noise([x_scaled, y_scaled])
        
        # Store the point with the noise-adjusted z value
        output[i] = [x, y, perlin_z * height + z_height]
    
    # Save the generated points to a file if a path is provided
    if save:
        np.save(save, output) # to retrieve use np.load("filename.npy")
    
    return output

def compute_normal_vectors(function_grid: np.array, ray_coordinates: np.array) -> np.array:
    """
    For each (x, y) coordinate in ray_coordinates, compute the normalized surface normal 
    vector of the surface represented by function_grid. The surface is assumed to be a height 
    field where each point is (x, y, z), and the density of points in function_grid is assumed 
    to be much higher than that of ray_coordinates. This allows a local plane fit (via least squares)
    to approximate the surface at each ray coordinate.

    For a local neighborhood around each ray coordinate, a plane of the form:
        z = a*x + b*y + c
    is fit to the nearby points. The surface normal is then computed as:
        n = (-a, -b, 1)
    which is normalized to unit length. The normal vector is adjusted, if necessary, so that 
    its z component is positive.

    Parameters:
    -----------
    function_grid : np.array
        A 2D numpy array of shape (N, 3) containing points (x, y, z) that represent the surface.
    ray_coordinates : np.array
        A 2D numpy array of shape (M, 2) containing the (x, y) coordinates at which the surface 
        normal is to be computed.

    Returns:
    --------
    np.array
        A 2D numpy array of shape (M, 3) where each row is a normalized normal vector (with positive 
        z component) corresponding to the surface at the respective ray coordinate.

    Notes:
    ------
    The algorithm uses a local least squares plane fit with a fixed number of nearest neighbors 
    (k=8) for each ray coordinate. This is appropriate if the density of function_grid is much higher 
    than that of ray_coordinates.
    """
    normals = []
    k = 8  # number of nearest neighbors to use for plane fitting

    # Pre-calculate the x and y coordinates from function_grid for distance computation
    surface_xy = function_grid[:, :2]
    
    for rx, ry in tqdm(ray_coordinates, desc="Calculating normal vectors"):
        # Compute Euclidean distances from the ray coordinate to all surface points (in x-y)
        distances = np.linalg.norm(surface_xy - np.array([rx, ry]), axis=1)
        # Get indices of the k closest neighbors
        if len(distances) < k:
            idx = np.argsort(distances)
        else:
            idx = np.argsort(distances)[:k]
        
        # Extract the local neighborhood points
        local_points = function_grid[idx]  # shape (k, 3) or less if not enough points
        
        # Set up the design matrix for plane fitting: [x, y, 1]
        A = np.c_[local_points[:, 0], local_points[:, 1], np.ones(local_points.shape[0])]
        b = local_points[:, 2]
        
        # Solve the least squares problem to find coefficients a, b, c in z = a*x + b*y + c
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        a, b_coef, _ = coeffs
        
        # Compute the normal vector as (-a, -b, 1)
        normal = np.array([-a, -b_coef, 1.0])
        
        # Normalize the normal vector
        norm = np.linalg.norm(normal)
        if norm != 0:
            normal /= norm
        
        # Ensure the z-component is positive
        if normal[2] < 0:
            normal = -normal
        
        normals.append(normal)
    
    return np.array(normals)

@jit(nopython=True)
def compute_normal_numba(local_points):
    """
    JIT-compiled function to compute normal from local points
    """
    # Set up the design matrix for plane fitting
    n_points = local_points.shape[0]
    A = np.zeros((n_points, 3))
    A[:, 0] = local_points[:, 0]  # x
    A[:, 1] = local_points[:, 1]  # y
    A[:, 2] = 1.0                 # constant term
    b = local_points[:, 2]        # z values
    
    # Use Numba's implementation of least squares for simplicity
    # (we'll use a simpler approach for the core calculation)
    ATA = np.zeros((3, 3))
    ATb = np.zeros(3)
    
    # Manual matrix multiplication for A^T*A and A^T*b
    for i in range(n_points):
        for j in range(3):
            ATb[j] += A[i, j] * b[i]
            for k in range(3):
                ATA[j, k] += A[i, j] * A[i, k]
    
    # Simple 3x3 matrix inversion 
    det = (ATA[0, 0] * (ATA[1, 1] * ATA[2, 2] - ATA[1, 2] * ATA[2, 1]) -
           ATA[0, 1] * (ATA[1, 0] * ATA[2, 2] - ATA[1, 2] * ATA[2, 0]) +
           ATA[0, 2] * (ATA[1, 0] * ATA[2, 1] - ATA[1, 1] * ATA[2, 0]))
    
    if abs(det) > 1e-10:
        inv_det = 1.0 / det
        coeffs = np.zeros(3)
        
        # Compute coefficients using Cramer's rule
        coeffs[0] = ((ATA[1, 1] * ATA[2, 2] - ATA[1, 2] * ATA[2, 1]) * ATb[0] +
                     (ATA[0, 2] * ATA[2, 1] - ATA[0, 1] * ATA[2, 2]) * ATb[1] +
                     (ATA[0, 1] * ATA[1, 2] - ATA[0, 2] * ATA[1, 1]) * ATb[2]) * inv_det
        
        coeffs[1] = ((ATA[1, 2] * ATA[2, 0] - ATA[1, 0] * ATA[2, 2]) * ATb[0] +
                     (ATA[0, 0] * ATA[2, 2] - ATA[0, 2] * ATA[2, 0]) * ATb[1] +
                     (ATA[0, 2] * ATA[1, 0] - ATA[0, 0] * ATA[1, 2]) * ATb[2]) * inv_det
        
        coeffs[2] = ((ATA[1, 0] * ATA[2, 1] - ATA[1, 1] * ATA[2, 0]) * ATb[0] +
                     (ATA[0, 1] * ATA[2, 0] - ATA[0, 0] * ATA[2, 1]) * ATb[1] +
                     (ATA[0, 0] * ATA[1, 1] - ATA[0, 1] * ATA[1, 0]) * ATb[2]) * inv_det
        
        a, b_coef = coeffs[0], coeffs[1]
    else:
        # Default to flat plane normal if singular
        a, b_coef = 0.0, 0.0
    
    # Compute the normal vector as (-a, -b, 1)
    normal = np.zeros(3)
    normal[0] = -a
    normal[1] = -b_coef
    normal[2] = 1.0
    
    # Normalize the normal vector
    norm = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    if norm > 1e-10:
        normal[0] /= norm
        normal[1] /= norm
        normal[2] /= norm
    
    # Ensure the z-component is positive
    if normal[2] < 0:
        normal[0] = -normal[0]
        normal[1] = -normal[1]
        normal[2] = -normal[2]
        
    return normal

def compute_normal_vectors_combined(function_grid, ray_coordinates, k=8, n_jobs=-1, batch_size=100):
    """
    Combined approach using both Numba JIT acceleration and parallel processing
    
    Parameters:
    -----------
    function_grid : np.array
        A 2D numpy array of shape (N, 3) containing points (x, y, z) that represent the surface.
    ray_coordinates : np.array
        A 2D numpy array of shape (M, 2) containing the (x, y) coordinates at which the surface 
        normal is to be computed.
    k : int, default=8
        Number of nearest neighbors to use for plane fitting.
    n_jobs : int, default=-1
        Number of jobs to run in parallel. -1 means using all processors.
    batch_size : int, default=100
        Number of ray coordinates to process in each parallel batch
    
    Returns:
    --------
    np.array
        A 2D numpy array of shape (M, 3) where each row is a normalized normal vector.
    """
    # Build KD-Tree from surface xy coordinates for efficient nearest neighbor search
    surface_xy = function_grid[:, :2]
    tree = cKDTree(surface_xy)
    
    # Precompute all nearest neighbors to avoid redundant tree queries
    print("Finding nearest neighbors for all points...")
    distances, indices = tree.query(ray_coordinates, k=k)
    
    # Process a batch of ray coordinates
    def process_batch(batch_idx):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, ray_coordinates.shape[0])
        batch_size_actual = end_idx - start_idx
        
        batch_normals = np.zeros((batch_size_actual, 3))
        
        for i in range(batch_size_actual):
            # Extract the local neighborhood points
            idx = indices[start_idx + i]
            local_points = function_grid[idx]
            
            # Use the Numba-accelerated function
            batch_normals[i] = compute_normal_numba(local_points)
            
        return batch_idx, batch_normals
    
    # Prepare batches
    num_batches = int(np.ceil(ray_coordinates.shape[0] / batch_size))
    print(f"Processing {num_batches} batches in parallel...")
    
    # Process batches in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_batch)(batch_idx) for batch_idx in range(num_batches)
    )
    
    # Reassemble results
    normals = np.zeros((ray_coordinates.shape[0], 3))
    for batch_idx, batch_normals in results:
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, ray_coordinates.shape[0])
        normals[start_idx:end_idx] = batch_normals
    
    return normals

def compute_angles(normals: np.array, light_rays: np.array) -> np.array:
    """
    Compute the angle (in radians) between each pair of corresponding normal and light beam vectors
    using the dot product of the vectors.
    
    Parameters:
    -----------
    normals : np.array
        A 2D numpy array of shape (N, 3) where each row is a normal vector to a surface.
    light_beams : np.array
        A 2D numpy array of shape (N, 3) where each row is an incident light beam vector.
    
    Returns:
    --------
    np.array
        A 1D numpy array of shape (N,) where each element is the angle (in radians) between the 
        corresponding normal and light beam vector.
    """
    # Compute dot products between corresponding vectors
    dot_products = np.sum(normals * light_rays, axis=1)
    
    # Compute the magnitudes of the normals and light beams
    normals_norm = np.linalg.norm(normals, axis=1)
    beams_norm = np.linalg.norm(light_rays, axis=1)
    
    # Avoid division by zero by replacing zeros with a small epsilon value
    epsilon = 1e-8
    normals_norm = np.where(normals_norm == 0, epsilon, normals_norm)
    beams_norm = np.where(beams_norm == 0, epsilon, beams_norm)
    
    # Compute cosine of angles and clip to the valid range [-1, 1] for arccos
    cos_angles = np.clip(dot_products / (normals_norm * beams_norm), -1.0, 1.0)
    
    # Compute angles in radians
    angles = np.arccos(cos_angles)
    
    return angles

import numpy as np

def refract_light(light_rays: np.array, normals: np.array, 
                  n_1: float, n_2: float) -> np.array:
    """
    Compute the refracted light directions using Snell's law for each incident ray.
    
    For each incident light beam and its corresponding surface normal, the function computes 
    the refracted direction using the vector form of Snell's law. The formula used is:
    
        η = n_1 / n_2
        cosθ_i = -dot(I, N)
        sin²θ_t = η² * (1 - cos²θ_i)
        cosθ_t = sqrt(1 - sin²θ_t)
        T = η * I + (η * cosθ_i - cosθ_t) * N
    
    where:
      - I is the normalized incident light beam vector,
      - N is the normalized surface normal vector (with positive z component),
      - θ_i is the angle between the incident ray and the normal (provided in `angles` for reference).
      
    It is assumed that:
      - The incident light beams and normals are provided as arrays of the same length.
      - The density of the data guarantees that total internal reflection does not occur.
      - The provided `angles` array corresponds (1-to-1) with the light beams and normals.
    
    Parameters:
    -----------
    light_beams : np.array
        A 2D numpy array of shape (N, 3) where each row represents an incident light beam vector.
    normals : np.array
        A 2D numpy array of shape (N, 3) where each row is a normalized surface normal vector.
    n_1 : int
        The refractive index of the medium from which the light is coming.
    n_2 : int
        The refractive index of the medium into which the light is transmitted.
        
    Returns:
    --------
    np.array
        A 2D numpy array of shape (N, 3) where each row is a normalized vector representing 
        the refracted light beam direction after applying Snell's law.
    """
    eta = n_1 / n_2  # Ratio of refractive indices
    refracted = []
    
    for i, (I, N) in enumerate(tqdm(zip(light_rays, normals), total=len(light_rays), desc="Calculating refracted rays")):
        # Normalize the incident light beam and surface normal
        I_norm = I / np.linalg.norm(I)
        N_norm = N / np.linalg.norm(N)
        
        # Compute the cosine of the incidence angle using the dot product.
        # (Assuming the light beam points towards the surface, we use -dot(I, N))
        cos_theta_i = -np.dot(I_norm, N_norm)
        
        # Calculate the squared sine of the transmission angle using Snell's law.
        sin_theta_t_sq = eta**2 * (1 - cos_theta_i**2)
        
        # For total internal reflection, sin_theta_t_sq > 1. Here, we assume TIR does not occur.
        cos_theta_t = np.sqrt(1 - sin_theta_t_sq)
        
        # Compute the refracted direction using the vector form of Snell's law.
        T = eta * I_norm + (eta * cos_theta_i - cos_theta_t) * N_norm
        
        # Normalize the refracted vector.
        T_norm = T / np.linalg.norm(T)
        refracted.append(T_norm)
    
    return np.array(refracted)

def compute_intersection_with_plane(refracted: np.array, ray_3coordinates: np.array, z_value: float = 0) -> np.array:
    """
    Compute the intersection points (x, y) of rays with a horizontal plane at a given z value.
    
    Parameters:
    -----------
    refracted : np.array
        A 2D numpy array of shape (N, 3) representing the direction vectors of the refracted rays.
    ray_3coordinates : np.array
        A 2D numpy array of shape (N, 3) representing the 3D coordinates where each ray originates.
    z_value : float, default=0
        The z-coordinate of the horizontal plane with which to compute the intersection.
        
    Returns:
    --------
    np.array
        A 2D numpy array of shape (N, 2) where each row represents the (x, y) coordinates of the 
        intersection point of the ray with the plane at z = z_value.
    """
    intersections = []
    
    for ray_point, direction in tqdm(zip(ray_3coordinates, refracted), total=len(ray_3coordinates), desc="Calculating screen interesections"):
        # Extract components of the point and direction vector.
        x0, y0, z0 = ray_point
        vx, vy, vz = direction
        
        # Calculate the parameter t at which the ray intersects the plane z = z_value.
        t = (z_value - z0) / vz
        
        # Compute the intersection coordinates in the xy-plane.
        x_intersect = x0 + t * vx
        y_intersect = y0 + t * vy
        
        intersections.append([x_intersect, y_intersect])
    
    return np.array(intersections)

def plot_interactions(interactions: np.array, kde: bool = False, save: str='') -> None:
    """
    Plot interaction points either as a scatter plot or as a kernel density estimation (KDE) map.
    
    Parameters:
    -----------
    interactions : np.array
        A 2D numpy array of shape (N, 2) where each row represents an (x, y) coordinate.
    kde : bool, default=False
        If False, a scatter plot of the points is produced. If True, a kernel density plot is generated.
    save : str, default=''
        If provided, the function will save the plot to the specified file path.
    """
    
    plt.figure(figsize=(8, 6))
    
    if not kde:
        # Scatter plot of the interaction points
        plt.scatter(interactions[:, 0], interactions[:, 1], c='blue', edgecolors='k', alpha=0.7)
        plt.title("Scatter Plot of Interaction Points")
        plt.xlabel("X")
        plt.ylabel("Y")
    else:
        # Compute the kernel density estimation
        x = interactions[:, 0]
        y = interactions[:, 1]
        
        # Create a grid over which to evaluate the KDE
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        x_grid, y_grid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
        
        kde_estimator = gaussian_kde(np.vstack([x, y]), bw_method=0.08)
        density = kde_estimator(grid_coords).reshape(x_grid.shape)
        
        # Plot the density using imshow with a grayscale colormap
        # The cmap 'gray' maps low values to black and high values to white.
        plt.imshow(density.T, origin='lower', cmap='gray', 
                   extent=[xmin, xmax, ymin, ymax], aspect='auto')
        plt.title("Kernel Density Estimation of Interaction Points")
        plt.xlabel("X")
        plt.ylabel("Y")
    
    plt.colorbar(label="Density" if kde else "")
    plt.tight_layout()

    if save:
        plt.savefig(save)
    else:
        plt.show()

def plot_interactions_3D(function_grid: np.array, ray_3coordinates: np.array, 
                         light_beams: np.array, interactions: np.array, 
                         z_value: float = 0, ray_count: int = 20, save: str=''
                         ) -> None:
    """
    3D plot combining a surface, caustic points, and ray paths.
    
    Parameters:
    -----------
    function_grid : np.array
        (N,3) array with surface (x,y,z) points.
    ray_3coordinates : np.array 
        (N,3) array with ray origin points.
    light_beams : np.array 
        (N,3) array with incident light beam vectors.
    interactions : np.array 
        (N,2) array with (x,y) caustic points (at z=z_value).
    z_value : float, default=0
        z-coordinate for the caustic plane.
    ray_count : int, default=20
        number of rays to render.
    save : str, default=''
        If provided, the function will save the plot to the specified file path.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface: assume function_grid forms a mesh grid.
    xs = function_grid[:, 0]
    ys = function_grid[:, 1]
    zs = function_grid[:, 2]
    x_unique = np.sort(np.unique(xs))
    y_unique = np.sort(np.unique(ys))
    X, Y = np.meshgrid(x_unique, y_unique)
    Z = function_grid[:, 2].reshape(X.shape)
    ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.5, edgecolor='none')
    
    # Plot 200 of caustics as scatter points at z=z_value.
    ax.scatter(*(interactions[np.random.choice(interactions.shape[0], 200, replace=False)].T), z_value, color='orange', s=40, label='Caustics')
    
    # Select evenly spaced rays.
    num_rays = min(ray_count, len(ray_3coordinates))
    indices = np.linspace(0, len(ray_3coordinates) - 1, num=num_rays, dtype=int)
    ray_color = 'red'
    
    # Draw lines from ray origin to caustic point.
    for idx in indices:
        origin = ray_3coordinates[idx]
        caustic = np.array([interactions[idx, 0], interactions[idx, 1], z_value])
        ax.plot([origin[0], caustic[0]],
                [origin[1], caustic[1]],
                [origin[2], caustic[2]],
                color=ray_color, linewidth=2)
    
    # Upward rays: from origin, in opposite direction to light_beams, up to 1.2*max(Z).
    max_z = np.max(Z)
    target_z = 1.2 * max_z
    for idx in indices:
        origin = ray_3coordinates[idx]
        beam = light_beams[idx] / np.linalg.norm(light_beams[idx])
        # Opposite direction (upward if beam is downward)
        d = -beam  
        if d[2] <= 0:
            continue
        t = (target_z - origin[2]) / d[2]
        endpoint = origin + t * d
        ax.plot([origin[0], endpoint[0]],
                [origin[1], endpoint[1]],
                [origin[2], endpoint[2]],
                color=ray_color, linewidth=2, linestyle='--')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Surface, Caustics, and Ray Paths")
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()

def plot_interactions_3D_plotly(function_grid: np.array, ray_3coordinates: np.array, 
                                  light_beams: np.array, interactions: np.array, 
                                  z_value: float = 0, ray_count: int = 20, save: str='') -> None:
    """
    3D plot combining a surface, caustic points, and ray paths using Plotly.
    
    Parameters:
    -----------
    function_grid : np.array
        (N,3) array with surface (x,y,z) points.
    ray_3coordinates : np.array 
        (N,3) array with ray origin points.
    light_beams : np.array 
        (N,3) array with incident light beam vectors.
    interactions : np.array 
        (N,2) array with (x,y) caustic points (at z=z_value).
    z_value : float, default=0
        z-coordinate for the caustic plane.
    ray_count : int, default=20
        number of rays to render.
    save : str, default=''
        If provided, the function will save the plot to the specified file path as an HTML file.
    """
    # Prepare surface data: assume function_grid forms a mesh grid.
    xs = function_grid[:, 0]
    ys = function_grid[:, 1]
    zs = function_grid[:, 2]
    x_unique = np.sort(np.unique(xs))
    y_unique = np.sort(np.unique(ys))
    X, Y = np.meshgrid(x_unique, y_unique)
    Z = function_grid[:, 2].reshape(X.shape)
    
    # Create the Plotly figure.
    fig = go.Figure()

    # Add surface trace.
    fig.add_trace(go.Surface(
        x=X, 
        y=Y, 
        z=Z, 
        colorscale='Blues',
        opacity=0.5,
        showscale=False,
        name='Surface'
    ))
    
    # Plot 200 caustics as scatter points at z=z_value.
    n_caustics = min(200, interactions.shape[0])
    idx_choices = np.random.choice(interactions.shape[0], n_caustics, replace=False)
    caustics = interactions[idx_choices]
    fig.add_trace(go.Scatter3d(
        x=caustics[:, 0],
        y=caustics[:, 1],
        z=np.full(n_caustics, z_value),
        mode='markers',
        marker=dict(size=5, color='orange'),
        name='Caustics'
    ))
    
    # Select evenly spaced rays.
    num_rays = min(ray_count, len(ray_3coordinates))
    indices = np.linspace(0, len(ray_3coordinates) - 1, num=num_rays, dtype=int)
    ray_color = 'red'
    
    # Draw lines from ray origin to caustic point.
    for idx in indices:
        origin = ray_3coordinates[idx]
        caustic = np.array([interactions[idx, 0], interactions[idx, 1], z_value])
        fig.add_trace(go.Scatter3d(
            x=[origin[0], caustic[0]],
            y=[origin[1], caustic[1]],
            z=[origin[2], caustic[2]],
            mode='lines',
            line=dict(color=ray_color, width=4),
            showlegend=False
        ))
    
    # Upward rays: from origin, in opposite direction to light_beams, up to 1.2*max(Z).
    max_z = np.max(Z)
    target_z = 1.2 * max_z
    for idx in indices:
        origin = ray_3coordinates[idx]
        beam = light_beams[idx] / np.linalg.norm(light_beams[idx])
        # Opposite direction (upward if beam is downward)
        d = -beam  
        if d[2] <= 0:
            continue
        t = (target_z - origin[2]) / d[2]
        endpoint = origin + t * d
        fig.add_trace(go.Scatter3d(
            x=[origin[0], endpoint[0]],
            y=[origin[1], endpoint[1]],
            z=[origin[2], endpoint[2]],
            mode='lines',
            line=dict(color=ray_color, width=4, dash='dash'),
            showlegend=False
        ))
    
    # Update layout with labels and title.
    fig.update_layout(
        title="3D Surface, Caustics, and Ray Paths",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # Save the plot if a file path is provided.
    if save:
        fig.write_html(save)
    fig.show()

def plot_interactions_rgb(interactions_red: np.array, interactions_green: np.array, interactions_blue: np.array, 
                          kde: bool = False, save: str='') -> None:
    """
    Plot interaction points for red, green, and blue light simultaneously, either as a scatter plot 
    or as a kernel density estimation (KDE) map with additive color mixing.
    
    Parameters:
    -----------
    interactions_red : np.array
        A 2D numpy array of shape (N, 2) where each row represents an (x, y) coordinate for red light.
    interactions_green : np.array
        A 2D numpy array of shape (N, 2) where each row represents an (x, y) coordinate for green light.
    interactions_blue : np.array
        A 2D numpy array of shape (N, 2) where each row represents an (x, y) coordinate for blue light.
    kde : bool, default=False
        If False, a scatter plot of the points is produced. If True, a kernel density plot is generated.
    save : str, default=''
        If provided, the function will save the plot to the specified file path.
    """
    
    plt.figure(figsize=(10, 8))
    
    if not kde:
        # Scatter plot of the interaction points
        plt.scatter(interactions_red[:, 0], interactions_red[:, 1], c='red', alpha=0.3, s=3, label='Red')
        plt.scatter(interactions_green[:, 0], interactions_green[:, 1], c='green', alpha=0.3, s=3, label='Green')
        plt.scatter(interactions_blue[:, 0], interactions_blue[:, 1], c='blue', alpha=0.3, s=3, label='Blue')
        
        plt.title("RGB Scatter Plot of Light Refraction")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
    else:
        # Find common bounds for all interaction points
        x_all = np.concatenate([interactions_red[:, 0], interactions_green[:, 0], interactions_blue[:, 0]])
        y_all = np.concatenate([interactions_red[:, 1], interactions_green[:, 1], interactions_blue[:, 1]])
        xmin, xmax = x_all.min(), x_all.max()
        ymin, ymax = y_all.min(), y_all.max()
        
        # Create a grid over which to evaluate the KDEs
        x_grid, y_grid = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
        grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
        
        # Compute the KDEs for each color
        bw = 0.08  # Bandwidth parameter - adjust as needed
        kde_red = gaussian_kde(np.vstack([interactions_red[:, 0], interactions_red[:, 1]]), bw_method=bw)
        kde_green = gaussian_kde(np.vstack([interactions_green[:, 0], interactions_green[:, 1]]), bw_method=bw)
        kde_blue = gaussian_kde(np.vstack([interactions_blue[:, 0], interactions_blue[:, 1]]), bw_method=bw)
        
        density_red = kde_red(grid_coords).reshape(x_grid.shape)
        density_green = kde_green(grid_coords).reshape(x_grid.shape)
        density_blue = kde_blue(grid_coords).reshape(x_grid.shape)
        
        # Scale each KDE by 255 divided by its maximum value to set the color intensity
        max_density = max(density_red.max(), density_green.max(), density_blue.max(), 1e-10)
        scaled_red = density_red * (255.0 / max_density)
        scaled_green = density_green * (255.0 / max_density)
        scaled_blue = density_blue * (255.0 / max_density)
        
        # Combine the scaled KDEs into an RGB image.
        # Note: we transpose the arrays so that the image dimensions match the grid.
        rgb_image = np.zeros((x_grid.shape[1], x_grid.shape[0], 3), dtype=np.uint8)
        rgb_image[:, :, 0] = scaled_red.T.astype(np.uint8)   # Red channel
        rgb_image[:, :, 1] = scaled_green.T.astype(np.uint8)   # Green channel
        rgb_image[:, :, 2] = scaled_blue.T.astype(np.uint8)    # Blue channel
        
        # Display the RGB image
        plt.imshow(rgb_image, origin='lower', extent=[xmin, xmax, ymin, ymax], aspect='auto')
        plt.title("RGB Kernel Density Estimation of Light Refraction")
        plt.xlabel("X")
        plt.ylabel("Y")
    
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300)
    else:
        plt.show()

def plot_interactions_3D_plotly_rgb(function_grid: np.array, ray_3coordinates: np.array, 
                                 light_beams: np.array, interactions_red: np.array, 
                                 interactions_green: np.array, interactions_blue: np.array,
                                 z_value: float = 0, ray_count: int = 20, save: str='') -> None:
    """
    3D plot combining a surface, RGB caustic points, and colored ray paths using Plotly.
    
    Parameters:
    -----------
    function_grid : np.array
        (N,3) array with surface (x,y,z) points.
    ray_3coordinates : np.array 
        (N,3) array with ray origin points.
    light_beams : np.array 
        (N,3) array with incident light beam vectors.
    interactions_red : np.array 
        (N,2) array with (x,y) caustic points for red light (at z=z_value).
    interactions_green : np.array 
        (N,2) array with (x,y) caustic points for green light (at z=z_value).
    interactions_blue : np.array 
        (N,2) array with (x,y) caustic points for blue light (at z=z_value).
    z_value : float, default=0
        z-coordinate for the caustic plane.
    ray_count : int, default=20
        number of rays to render per color.
    save : str, default=''
        If provided, the function will save the plot to the specified file path as an HTML file.
    """
    # Prepare surface data
    xs = function_grid[:, 0]
    ys = function_grid[:, 1]
    zs = function_grid[:, 2]
    x_unique = np.sort(np.unique(xs))
    y_unique = np.sort(np.unique(ys))
    X, Y = np.meshgrid(x_unique, y_unique)
    Z = function_grid[:, 2].reshape(X.shape)
    
    # Create the Plotly figure
    fig = go.Figure()

    # Add surface trace for the refracting surface
    fig.add_trace(go.Surface(
        x=X, 
        y=Y, 
        z=Z, 
        colorscale='Blues',
        opacity=0.5,
        showscale=False,
        name='Surface'
    ))
    
    # Create caustic plane coordinates
    # Determine bounds from all interactions
    x_min = min(interactions_red[:, 0].min(), interactions_green[:, 0].min(), interactions_blue[:, 0].min())
    x_max = max(interactions_red[:, 0].max(), interactions_green[:, 0].max(), interactions_blue[:, 0].max())
    y_min = min(interactions_red[:, 1].min(), interactions_green[:, 1].min(), interactions_blue[:, 1].min())
    y_max = max(interactions_red[:, 1].max(), interactions_green[:, 1].max(), interactions_blue[:, 1].max())
    
    resolution = 300  # Use higher resolution as in the matplotlib version
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X_caustic, Y_caustic = np.meshgrid(x_grid, y_grid)
    points = np.vstack([X_caustic.flatten(), Y_caustic.flatten()])
    
    bw = 0.08  # Bandwidth for KDE
    # Compute KDE for each channel using the interactions (transpose if needed)
    kde_red = gaussian_kde(interactions_red.T, bw_method=bw)
    kde_green = gaussian_kde(interactions_green.T, bw_method=bw)
    kde_blue = gaussian_kde(interactions_blue.T, bw_method=bw)
    
    density_red = kde_red(points).reshape(resolution, resolution)
    density_green = kde_green(points).reshape(resolution, resolution)
    density_blue = kde_blue(points).reshape(resolution, resolution)
    
    # ---- Global Normalization ----
    # Instead of normalizing each channel by its own maximum, use the global maximum
    global_max = max(density_red.max(), density_green.max(), density_blue.max()) + 1e-10
    scaled_red = density_red * (255.0 / global_max)
    scaled_green = density_green * (255.0 / global_max)
    scaled_blue = density_blue * (255.0 / global_max)
    
    # Compute a combined intensity for mapping (here simply the sum)
    intensity = scaled_red + scaled_green + scaled_blue
    intensity_norm = intensity / intensity.max()  # Normalize to 0-1

    # Build an array of RGB strings for each grid cell (without iterating over every point for the colorscale)
    # Instead, we will sample a fixed number of intensity values to build the colorscale.
    n_colors = 10
    # Prepare an array to hold the per-cell RGB values
    rgb_array = np.empty((resolution, resolution), dtype=object)
    for i in range(resolution):
        for j in range(resolution):
            r = int(scaled_red[i, j])
            g = int(scaled_green[i, j])
            b = int(scaled_blue[i, j])
            rgb_array[i, j] = f'rgb({r},{g},{b})'
    
    # Build a custom colorscale by sampling n_colors values from the intensity_norm range
    colorscale = []
    sample_vals = np.linspace(0, 1, n_colors)
    for val in sample_vals:
        # Find the grid cell where intensity_norm is closest to this sample value
        diff = np.abs(intensity_norm - val)
        idx_flat = diff.argmin()
        i, j = np.unravel_index(idx_flat, intensity_norm.shape)
        colorscale.append([val, rgb_array[i, j]])
    
    # Add the caustic plane as a surface. The z-value is constant.
    fig.add_trace(go.Surface(
        x=X_caustic,
        y=Y_caustic,
        z=np.full((resolution, resolution), z_value),
        surfacecolor=intensity_norm,
        colorscale=colorscale,
        showscale=False,
        opacity=1.0,
        name='RGB Caustics'
    ))
    
    # Draw colored rays for each color
    ray_colors = ['red', 'green', 'blue']
    interaction_sets = [interactions_red, interactions_green, interactions_blue]
    
    # Create downward rays for each color
    for interactions, color in zip(interaction_sets, ray_colors):
        num_rays = min(ray_count, len(ray_3coordinates))
        indices = np.linspace(0, len(ray_3coordinates) - 1, num=num_rays, dtype=int)
        
        # Draw lines from ray origin to caustic point
        for idx in indices:
            if idx >= len(interactions):
                continue
                
            origin = ray_3coordinates[idx]
            caustic = np.array([interactions[idx, 0], interactions[idx, 1], z_value])
            fig.add_trace(go.Scatter3d(
                x=[origin[0], caustic[0]],
                y=[origin[1], caustic[1]],
                z=[origin[2], caustic[2]],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False
            ))
        
        # Create upward rays
        max_z = np.max(Z)
        target_z = 1.2 * max_z
        for idx in indices:
            origin = ray_3coordinates[idx]
            beam = light_beams[idx] / np.linalg.norm(light_beams[idx])
            # Opposite direction (upward if beam is downward)
            d = -beam  
            if d[2] <= 0:
                continue
            t = (target_z - origin[2]) / d[2]
            endpoint = origin + t * d
            fig.add_trace(go.Scatter3d(
                x=[origin[0], endpoint[0]],
                y=[origin[1], endpoint[1]],
                z=[origin[2], endpoint[2]],
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                showlegend=False
            ))
    
    # Update layout with labels and title
    fig.update_layout(
        title="RGB Caustics and Ray Paths",
        scene=dict(
            aspectmode='data',
            xaxis=dict(title="X", showgrid=False),
            yaxis=dict(title="Y", showgrid=False),
            zaxis=dict(title="Z", showgrid=False),
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # Save the plot if a file path is provided
    if save:
        fig.write_html(save)
        
    fig.show()