import numpy as np
from typing import Callable
from functions import math_raycast as mr
import plotly.express as px
import plotly.graph_objects as go
import json
import os

def plot_caustic_map(colour_map: np.ndarray, x_lim: list, y_lim: list):
    """
    Plot the RGB kernel density estimation of light refraction using Plotly.

    Parameters:
        colour_map (np.ndarray): An image array of shape (num_y_bins, num_x_bins, 3)
            with values scaled to 0-255.
        x_lim (list): [x_min, x_max] specifying the binning range for x.
        y_lim (list): [y_min, y_max] specifying the binning range for y.
        as_html (bool): If True, returns the figure as an HTML string (suitable for embedding in Netlify pages).

    Returns:
        A Plotly Figure object, or its HTML representation if as_html is True.
    """

    # Create coordinate arrays that match the dimensions of the colour_map.
    x_coords = np.linspace(x_lim[0], x_lim[1], colour_map.shape[1])
    y_coords = np.linspace(y_lim[0], y_lim[1], colour_map.shape[0])
    
    # Create the Plotly figure.
    fig = px.imshow(colour_map, x=x_coords, y=y_coords, origin='lower')
    fig.update_layout(
        title="RGB Kernel Density Estimation of Light Refraction",
        xaxis_title="x",
        yaxis_title="y",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.show()

def plot_interactions(f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                      X: np.ndarray, Y: np.ndarray,colour_map: np.ndarray,
                      x_lim: list, y_lim: list, intersections_red: np.ndarray,
                      intersections_green: np.ndarray, intersections_blue: np.ndarray,
                      light_ray_count: int, z_surface: float ) -> None:
    """
    Plot a 3D surface with light ray interactions and a caustic pattern.
    
    Parameters:
    -----------
    f : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function that takes X and Y coordinates and returns Z values
    X : np.ndarray
        X coordinates grid
    Y : np.ndarray
        Y coordinates grid
    colour_map : np.ndarray
        3D array representing the RGB caustic pattern to be displayed at z_surface
    x_lim : list
        X-axis limits [min, max]
    y_lim : list
        Y-axis limits [min, max]
    intersections_red : np.ndarray
        Red light ray intersection points at z_surface, same shape as X and Y
    intersections_green : np.ndarray
        Green light ray intersection points at z_surface, same shape as X and Y
    intersections_blue : np.ndarray
        Blue light ray intersection points at z_surface, same shape as X and Y
    light_ray_count : int
        Number of light rays to plot
    z_surface : float
        Z-coordinate where the caustic pattern and ray intersections occur
        
    Returns:
    --------
    None
    """
    # Calculate Z values for the surface
    Z = f(X, Y)
    
    # Mask values outside x_lim and y_lim
    mask = (X >= x_lim[0]) & (X <= x_lim[1]) & (Y >= y_lim[0]) & (Y <= y_lim[1])
    
    # Apply mask to keep only points within limits
    X_masked = np.where(mask, X, np.nan)
    Y_masked = np.where(mask, Y, np.nan)
    Z_masked = np.where(mask, Z, np.nan)
    
    # Create a 3D figure
    fig = go.Figure()
    
    # 1. Plot the main surface
    fig.add_trace(go.Surface(
        x=X_masked, y=Y_masked, z=Z_masked,
        colorscale='Blues',
        opacity=0.5,
        showscale=False,
        name='Surface',
        showlegend=True
    ))
    
    # 2. Plot the colour map at z=z_surface
    x_coords = np.linspace(x_lim[0], x_lim[1], colour_map.shape[1])
    y_coords = np.linspace(y_lim[0], y_lim[1], colour_map.shape[0])
    X_caustic, Y_caustic = np.meshgrid(x_coords, y_coords)
    Z_caustic = np.ones_like(X_caustic) * z_surface
    
    # Add the caustic as a surface at z=z_surface using RGB values directly
    # Compute the intensity (sum of RGB values)
    intensity = np.sum(colour_map, axis=2)  # Sum over the RGB channels
    intensity_norm = intensity / intensity.max()  # Normalize to 0-1

    # Create an array of RGB strings for each grid cell
    rgb_array = np.empty(colour_map.shape[:2], dtype=object)
    for y in range(colour_map.shape[0]):
        for x in range(colour_map.shape[1]):
            r = int(colour_map[y, x, 0])
            g = int(colour_map[y, x, 1])
            b = int(colour_map[y, x, 2])
            rgb_array[y, x] = f'rgb({r},{g},{b})'

    # Build a custom colorscale
    n_colors = 10
    colorscale = []
    sample_vals = np.linspace(0, 1, n_colors)
    for val in sample_vals:
        # Find the grid cell where intensity_norm is closest to this sample value
        diff = np.abs(intensity_norm - val)
        idx_flat = diff.argmin()
        y, x = np.unravel_index(idx_flat, intensity_norm.shape)
        colorscale.append([val, rgb_array[y, x]])

    fig.add_trace(go.Surface(
        x=X_caustic,
        y=Y_caustic,
        z=Z_caustic,
        surfacecolor=intensity_norm,
        colorscale=colorscale,
        showscale=False,
        opacity=1.0,
        name='Caustics',
        showlegend=True
    ))
    
    # 3. Select and plot light rays only within x_lim and y_lim
    # Find indices of points within limits
    within_limits = (X >= x_lim[0]) & (X <= x_lim[1]) & (Y >= y_lim[0]) & (Y <= y_lim[1])
    valid_indices = np.where(within_limits)
    
    # Only proceed if we have valid points
    if len(valid_indices[0]) > 0:
        # Use 2D index array for selection
        valid_indices_array = np.array(valid_indices).T
        
        # Select evenly distributed points from valid indices
        num_valid = len(valid_indices_array)
        if num_valid >= light_ray_count:
            # For uniform distribution, select indices with reasonably even spacing
            selected_idx = np.linspace(0, num_valid-1, light_ray_count, dtype=int)
            selected_points = valid_indices_array[selected_idx]
        else:
            # If we have fewer points than requested, use all available
            selected_points = valid_indices_array
            
        max_f = np.nanmax(Z_masked)
        
        # Plot selected rays
        for k, (i, j) in enumerate(selected_points):
            x, y = X[i, j], Y[i, j]
            z = Z[i, j]
            
            # 3.1 Plot black lines from surface to extended height
            fig.add_trace(go.Scatter3d(
                x=[x, x],
                y=[y, y],
                z=[z, max_f*1.2],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))
            
            # 3.2 Plot colored rays to their respective intersections
            # Red ray
            red_x, red_y = intersections_red[i, j]
            fig.add_trace(go.Scatter3d(
                x=[x, red_x],
                y=[y, red_y],
                z=[z, z_surface],
                mode='lines',
                line=dict(color='rgb(255,0,0)', width=3),
                name='Red Ray' if k == 0 else None,
                showlegend=bool(k == 0)
            ))
            
            # Green ray
            green_x, green_y = intersections_green[i, j]
            fig.add_trace(go.Scatter3d(
                x=[x, green_x],
                y=[y, green_y],
                z=[z, z_surface],
                mode='lines',
                line=dict(color='rgb(0,255,0)', width=3),
                name='Green Ray' if k == 0 else None,
                showlegend=bool(k == 0)
            ))
            
            # Blue ray
            blue_x, blue_y = intersections_blue[i, j]
            fig.add_trace(go.Scatter3d(
                x=[x, blue_x],
                y=[y, blue_y],
                z=[z, z_surface],
                mode='lines',
                line=dict(color='rgb(0,0,255)', width=3),
                name='Blue Ray' if k == 0 else None,
                showlegend=bool(k == 0)
            ))
    
    # Update layout for better visualization
    fig.update_layout(
        title="3D Surface with Light Ray Interactions",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectratio=dict(x=1, y=1, z=0.8)
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Show the figure
    fig.show()

def main():
    ### Surface parameters
    def f(x, y):
        return 1 + 0.1 * (x**2+y**2) # This should be a function input box where the user can input a function of x and y, when this is modified you need to rerun all calculations and update the plot section
    
    ### Ray parameters
    resolution = 1500 # Should be an int slider from 200 to 5000, when this is modified you need to rerun all calculations and update the plot section
    n_0 = 1 # float input box, when this is modified you need to rerun all refracted vector calculations and intersection calculations then update the plot section
    n_red = 1.5 # float input box, when this is modified you only need to run  refracted_vectors_red = refracted_ray(n_0, n_red, unit_normals), intersection_red = intersection_with_plane(refracted_vectors_red, f, X, Y, z_surface) and update the plot section
    n_blue = 1.5 # float input box, when this is modified you only need to run  refracted_vectors_blue = refracted_ray(n_0, n_blue, unit_normals), intersection_blue = intersection_with_plane(refracted_vectors_blue, f, X, Y, z_surface) and update the plot section
    n_green = 1.5 # float input box, when this is modified you only need to run  refracted_vectors_green = refracted_ray(n_0, n_green, unit_normals), intersection_green = intersection_with_plane(refracted_vectors_green, f, X, Y, z_surface) and update the plot section
    light_ray_count = 20 # int sloder which goes from 5 to 50, when thi is modified only the plot section need to be updated

    ### Advanced parameters
    x_min, x_max = -5, 5 # float input boxes, when this is modified you need to rerun all calculations and update the plot section
    y_min, y_max = -5, 5 # float input boxes, when this is modified you need to rerun all calculations and update the plot section
    sigma = 0.5 # float input box, when this is modified you only need to update the plot section
    z_surface = -1  # float input box, when this  is modified you only need to rerun intersection calculations and update the plot section
    brighten = 1 # float slider from 1 to 10, when this is modified you only need to update the plot section
    folder_name = 'simulation_1' # string input box, when this is modified nothing should be re-run this is only to save
    
    # Computation section of code
    # Compute the grid ranges and number of rays
    x_range = [x_min * 1.4, x_max * 1.4]
    y_range = [y_min * 1.4, y_max * 1.4]
    ray_number = round(resolution * 1.4)
    
    # Create the initial grid on the surface
    x_grid = np.linspace(x_range[0], x_range[1], ray_number)
    y_grid = np.linspace(y_range[0], y_range[1], ray_number)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Compute the normalized normal vectors at (X, Y)
    unit_normals = mr.normalized_normal(f, X, Y, h=1e-4)
    
    # Compute the refracted ray vectors based on the unit normals
    refracted_vectors_red = mr.refracted_ray(n_0, n_red, unit_normals)
    refracted_vectors_green = mr.refracted_ray(n_0, n_blue, unit_normals)
    refracted_vectors_blue = mr.refracted_ray(n_0, n_green, unit_normals)
    
    # Find the (x, y) intersection of the refracted rays with the plane z=z_surface
    intersections_red = mr.intersection_with_plane(refracted_vectors_red, f, X, Y, z_surface)
    intersections_green = mr.intersection_with_plane(refracted_vectors_green, f, X, Y, z_surface)
    intersections_blue = mr.intersection_with_plane(refracted_vectors_blue, f, X, Y, z_surface)

    # Plot section of the code
    colour_map = mr.create_colour_map(intersections_red, intersections_green, intersections_blue, bins=round(resolution*0.5), x_lim=[x_min, x_max], y_lim=[y_min, y_max], sigma=sigma, brighten=brighten)

    plot_interactions(f, X, Y, colour_map, [x_min, x_max], [y_min, y_max], intersections_red, intersections_green, intersections_blue, light_ray_count, z_surface)
    plot_caustic_map(colour_map, [x_min, x_max], [y_min, y_max])

    # Save the caustic
    # Ensure the /data/ folder and the specified folder_name exist
    output_folder = os.path.join('data', folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Save the configuration parameters to a JSON file
    config = {
        "surface_function": "1 + 0.5 * np.sin(4 * x) * np.sin(y)",
        "resolution": resolution,
        "n_0": n_0,
        "n_red": n_red,
        "n_blue": n_blue,
        "n_green": n_green,
        "light_ray_count": light_ray_count,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "sigma": sigma,
        "z_surface": z_surface,
        "brighten": brighten,
        "folder_name": folder_name
    }

    config_path = os.path.join(output_folder, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file, indent=4)

    # Save the caustic map to the specified folder
    caustic_map_path = os.path.join(output_folder, 'caustic_map.pdf')
    mr.save_caustic_map(colour_map, [x_min, x_max], [y_min, y_max], caustic_map_path)

if __name__ == "__main__":
    # Interactive Caustics Simulation
    main()