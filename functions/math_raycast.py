import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from typing import Callable

def normalized_normal(f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                      X: np.ndarray, 
                      Y: np.ndarray, 
                      h: float = 1e-4) -> np.ndarray:
    """
    Compute the normalized normal vector of the surface defined by z = f(x, y) at the points (X, Y).

    This function approximates the partial derivatives of f with respect to x and y using central differencing.
    The normal vector at any point on the surface can be expressed as (-f_x, -f_y, 1). This function returns 
    the unit normal vector by normalizing this vector.

    Parameters:
        f : Callable[[np.ndarray, np.ndarray], np.ndarray]
            A scalar function of two variables.
        X : np.ndarray
            Array of x-coordinates.
        Y : np.ndarray
            Array of y-coordinates.
        h : float, optional
            Step size for central differencing (default is 1e-4).

    Returns:
        np.ndarray
            An array of shape (..., 3) containing the normalized normal vectors at the corresponding (X, Y) points.
    """
    # Compute partial derivatives using central differencing
    fx = (f(X + h, Y) - f(X - h, Y)) / (2 * h)
    fy = (f(X, Y + h) - f(X, Y - h)) / (2 * h)
    
    # Construct the normal vector (-fx, -fy, 1)
    normal = np.stack([-fx, -fy, np.ones_like(fx)], axis=-1)
    
    # Normalize the vector
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    unit_normal = normal / norm
    
    return unit_normal

def refracted_ray(n_0: float, n_1: float, unit_normal: np.ndarray) -> np.ndarray:
    """

    Compute the refracted ray vector using the vector form of Snell's law.
    
    This function assumes an incident ray with direction (0, 0, -1) impinging 
    on a surface at a point with a given unit normal vector. The refracted ray is
    computed using:
    
        t = eta * i + (eta * cos(theta_i) - sqrt(1 - eta**2 * (1 - cos(theta_i)**2))) * n
    
    where:
        - i is the incident ray direction (0, 0, -1),
        - n is the unit normal vector at the point of incidence,
        - eta = n_0 / n_1,
        - cos(theta_i) = - n · i   (the angle between the incident ray and the normal),
        - n_0 is the refractive index of the incident medium,
        - n_1 is the refractive index of the refracted medium.
    
    Parameters:
        n_0 : float
            Refractive index of the incident medium.
        n_1 : float
            Refractive index of the refracted medium.
        unit_normal : np.ndarray
            The unit normal vector at the point of incidence.
            Can be of shape (3,) for a single vector or (..., 3) for multiple vectors.
    
    Returns:
        np.ndarray
            The refracted ray vector with the same shape as unit_normal.
    
    Raises:
        ValueError: If total internal reflection occurs (i.e., the square root becomes negative).
    """
    # Define the incident ray vector
    i = np.array([0.0, 0.0, -1.0])
    
    # Calculate the ratio of refractive indices
    eta = n_0 / n_1
    
    # Compute cosine of the incident angle: cos(theta_i) = - n · i
    # Use np.sum for potential broadcasting when unit_normal has shape (..., 3)
    cos_theta_i = -np.sum(unit_normal * i, axis=-1)
    
    # Compute the term inside the square root.
    radicand = 1 - eta**2 * (1 - cos_theta_i**2)
    if np.any(radicand < 0):
        raise ValueError("Total internal reflection: no real refracted ray exists for these parameters.")
    
    sqrt_term = np.sqrt(radicand)
    
    # Compute the refracted ray using Snell's law in vector form.
    # The term (eta * cos_theta_i - sqrt_term) must be broadcasted to match the vector shape.
    t = eta * i + (eta * cos_theta_i - sqrt_term)[..., np.newaxis] * unit_normal
    return t

def intersection_with_plane(refracted_vector: np.ndarray,
                            f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                            X: np.ndarray, 
                            Y: np.ndarray, 
                            z_0: float) -> np.ndarray:
    """
    Compute the (x, y) coordinates of the intersection between a ray and a horizontal plane z = z_0.
    
    The ray is defined by its starting point on the surface z = f(x, y) at (X, Y, f(X, Y)) and its 
    direction given by `refracted_vector`. The ray's equation is:
    
        R(s) = (X, Y, f(X, Y)) + s * refracted_vector,
    
    where s is a scalar parameter. The intersection with the plane z = z_0 occurs when the z-component 
    of R(s) equals z_0. Solving for s:
    
        f(X, Y) + s * (refracted_vector[..., 2]) = z_0,
        s = (z_0 - f(X, Y)) / refracted_vector[..., 2].
    
    Then the (x, y) coordinates of the intersection point are:
    
        x = X + s * refracted_vector[..., 0],
        y = Y + s * refracted_vector[..., 1].
    
    Parameters:
        refracted_vector : np.ndarray
            The refracted ray vector (or vectors) with shape (..., 3) as obtained from the refraction function.
        f : Callable[[np.ndarray, np.ndarray], np.ndarray]
            The function defining the surface, where z = f(x, y).
        X : np.ndarray
            Array of x-coordinates where the ray originates on the surface.
        Y : np.ndarray
            Array of y-coordinates where the ray originates on the surface.
        z_0 : float
            The z-coordinate of the horizontal plane with which the ray intersects.
    
    Returns:
        np.ndarray:
            An array of shape (..., 2) containing the (x, y) coordinates of the intersection points.
    
    Raises:
        ValueError: If the z-component of the refracted_vector is zero (ray is parallel to the plane).
    """
    # Compute the z-coordinate of the starting point on the surface.
    z_start = f(X, Y)
    
    # Extract the z-component of the refracted vector.
    t_z = refracted_vector[..., 2]
    
    # Check if any t_z is zero to avoid division by zero.
    if np.any(np.isclose(t_z, 0)):
        raise ValueError("The z-component of the refracted vector is zero; the ray is parallel to the plane z=z_0.")
    
    # Compute the scalar parameter s for the ray equation at intersection.
    s = (z_0 - z_start) / t_z
    
    # Compute the intersection coordinates.
    x_int = X + s * refracted_vector[..., 0]
    y_int = Y + s * refracted_vector[..., 1]
    
    # Stack the results to obtain an array of shape (..., 2).
    intersection_points = np.stack([x_int, y_int], axis=-1)
    
    return intersection_points

def create_colour_map(intersections_red: np.ndarray, 
                      intersections_green: np.ndarray, 
                      intersection_blue: np.ndarray, 
                      bins: int, x_lim: list, 
                      y_lim: list, sigma: float = 0.5,
                      brighten: float=1.0) -> np.ndarray:
    """
    Create a colour map image ready for imshow based on binned intersections.
    
    Parameters:
        intersections_red (np.ndarray): Data points for the red channel. 
            Should be of shape (N, 2) where each row is [x, y].
        intersections_green (np.ndarray): Data points for the green channel.
            Should be of shape (N, 2) where each row is [x, y].
        intersection_blue (np.ndarray): Data points for the blue channel.
            Should be of shape (N, 2) where each row is [x, y].
        bins (int): Number of bins to use in both the x and y directions.
        x_lim (list): [x_min, x_max] specifying the binning range for x.
        y_lim (list): [y_min, y_max] specifying the binning range for y.
        sigma (float): Sigma for the Gaussian filter (default 0.5).
        
    Returns:
        np.ndarray: An image array of shape (num_y_bins, num_x_bins, 3) with values
        scaled to 0-255 and ready to be given to plt.imshow.
    """
    
    # Flatten the input arrays from shape (N, M, 2) to (N*M, 2)
    intersections_red = intersections_red.reshape(-1, 2)
    intersections_green = intersections_green.reshape(-1, 2)
    intersection_blue = intersection_blue.reshape(-1, 2)

    # Bin the red intersections using np.histogram2d
    H_red, x_edges, y_edges = np.histogram2d(
        intersections_red[:, 0], intersections_red[:, 1], 
        bins=bins, range=[x_lim, y_lim]
    )
    
    # Bin the green intersections
    H_green, _, _ = np.histogram2d(
        intersections_green[:, 0], intersections_green[:, 1], 
        bins=bins, range=[x_lim, y_lim]
    )
    
    # Bin the blue intersections
    H_blue, _, _ = np.histogram2d(
        intersection_blue[:, 0], intersection_blue[:, 1], 
        bins=bins, range=[x_lim, y_lim]
    )
    
    # Apply Gaussian filter to smooth the histograms
    H_red_filtered = ndimage.gaussian_filter(H_red, sigma=sigma)
    H_green_filtered = ndimage.gaussian_filter(H_green, sigma=sigma)
    H_blue_filtered = ndimage.gaussian_filter(H_blue, sigma=sigma)
    
    # Find the maximum density among all channels
    max_density = max(H_red_filtered.max(), H_green_filtered.max(), H_blue_filtered.max())
    
    # Renormalise the histograms to the range [0, 255] and apply the brighten factor
    scaled_red = np.clip(H_red_filtered * (255.0 / max_density) * brighten, 0, 255)
    scaled_green = np.clip(H_green_filtered * (255.0 / max_density) * brighten, 0, 255)
    scaled_blue = np.clip(H_blue_filtered * (255.0 / max_density) * brighten, 0, 255)
    
    # Create the RGB image
    # Note: Transpose the histograms so that the image dimensions match the grid
    rgb_image = np.zeros((scaled_red.shape[1], scaled_red.shape[0], 3), dtype=np.uint8)
    rgb_image[:, :, 0] = scaled_red.T.astype(np.uint8)   # Red channel
    rgb_image[:, :, 1] = scaled_green.T.astype(np.uint8)   # Green channel
    rgb_image[:, :, 2] = scaled_blue.T.astype(np.uint8)    # Blue channel
    
    return rgb_image

def save_caustic_map(colour_map: np.ndarray, x_lim: list, y_lim: list, filename: str):
    """
    Plot the RGB kernel density estimation of light refraction.
    
    Parameters:
        colour_map (np.ndarray): An image array of shape (num_y_bins, num_x_bins, 3)
            with values scaled to 0-255 and ready to be given to plt.imshow.
        x_lim (list): [x_min, x_max] specifying the binning range for x.
        y_lim (list): [y_min, y_max] specifying the binning range for y.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(colour_map, origin='lower', extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], aspect='auto')
    plt.title("RGB Kernel Density Estimation of Light Refraction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)