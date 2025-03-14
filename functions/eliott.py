import numpy as np
from perlin_noise import PerlinNoise # `pip install perlin-noise`
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm # `pip install tqdm`


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
        
        kde_estimator = gaussian_kde(np.vstack([x, y]))
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