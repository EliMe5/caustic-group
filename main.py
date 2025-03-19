import numpy as np
import os
import json
import time
import functions.eliott as el
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import dash
from dash import dcc, html, callback, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import dash_bootstrap_components as dbc

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout for the app
app.layout = html.Div([
    html.H1("Interactive Caustics Simulation", className="text-center my-4"),
    
    dbc.Row([
        # Left column: Control panel
        dbc.Col([
            html.Div([
                html.H3("Surface Parameters"),
                
                html.Label("Surface Mesh Resolution:"),
                dcc.Input(id="surface-mesh", type="number", value=1000, min=100, max=2000, step=100),
                
                html.Label("Perlin Noise Octaves:"),
                dcc.Slider(id="octaves", min=1, max=5, step=1, value=3, marks={i: str(i) for i in range(1, 6)}),
                
                html.Label("Perlin Noise Scale:"),
                dcc.Slider(id="scale", min=0.01, max=0.5, step=0.01, value=0.1, marks={i/10: str(i/10) for i in range(1, 6)}),
                
                html.Label("Surface Height:"),
                dcc.Slider(id="height", min=0.1, max=2.0, step=0.1, value=0.8, marks={i/10: str(i/10) for i in range(1, 21)}),
                
                html.Label("Z Height Maximum:"),
                dcc.Slider(id="z-height", min=0.5, max=5.0, step=0.5, value=2.0, marks={i: str(i) for i in range(1, 6)}),
                
                html.H3("Ray Parameters", className="mt-4"),
                
                html.Label("Rays Mesh Resolution:"),
                dcc.Input(id="rays-mesh", type="number", value=100, min=10, max=500, step=10),
                
                html.Label("Theta Angle (0-360°):"),
                dcc.Slider(id="theta", min=0, max=360, step=5, value=0, marks={i: str(i) for i in range(0, 361, 90)}),
                
                html.Label("Phi Angle (0-90°):"),
                dcc.Slider(id="phi", min=0, max=90, step=5, value=90, marks={i: str(i) for i in range(0, 91, 30)}),
                
                html.Label("Refractive Index (Red):"),
                dcc.Input(id="n-red", type="number", value=2.00, min=1.0, max=3.0, step=0.01),
                
                html.Label("Refractive Index (Green):"),
                dcc.Input(id="n-green", type="number", value=1.50, min=1.0, max=3.0, step=0.01),
                
                html.Label("Refractive Index (Blue):"),
                dcc.Input(id="n-blue", type="number", value=1.00, min=1.0, max=3.0, step=0.01),
                
                html.Label("Light Ray Count (for visualization):"),
                dcc.Slider(id="light-ray-count", min=5, max=50, step=5, value=20, marks={i: str(i) for i in range(5, 51, 10)}),
                
                html.H3("Advanced Parameters", className="mt-4"),
                
                html.Label("X Min:"),
                dcc.Input(id="x-min", type="number", value=-5, step=1),
                
                html.Label("X Max:"),
                dcc.Input(id="x-max", type="number", value=5, step=1),
                
                html.Label("Y Min:"),
                dcc.Input(id="y-min", type="number", value=-5, step=1),
                
                html.Label("Y Max:"),
                dcc.Input(id="y-max", type="number", value=5, step=1),
                
                html.Label("Z Screen:"),
                dcc.Input(id="z-screen", type="number", value=0, step=0.1),
                
                html.Label("Random Seed:"),
                dcc.Input(id="random-seed", type="number", value=42, min=1, step=1),
                
                html.H3("Save Options", className="mt-4"),
                
                html.Label("Save Filename:"),
                dcc.Input(id="save-filename", type="text", value="simulation_1", placeholder="Enter filename without extension"),
                
                html.Div([
                    dbc.Button("Render", id="render-button", color="primary", className="me-2"),
                    dbc.Button("Save", id="save-button", color="success", disabled=True),
                ], className="d-flex justify-content-between mt-4"),
                
                html.Div(id="status-message", className="mt-2")
            ], className="p-4 border rounded")
        ], width=4),
        
        # Right column: Visualization
        dbc.Col([
            dcc.Loading(
                id="loading-vis",
                type="circle",
                children=[
                    html.Div([
                        html.H3("3D Visualization"),
                        dcc.Graph(id="3d-plot", style={"height": "70vh"}),
                        
                        html.H3("RGB Caustic Pattern"),
                        dcc.Graph(id="rgb-plot", style={"height": "30vh"})
                    ])
                ]
            )
        ], width=8)
    ])
], className="container-fluid")

# Store for holding the current simulation results
app.layout.children.append(dcc.Store(id="simulation-results"))


@app.callback(
    [Output("3d-plot", "figure"),
     Output("rgb-plot", "figure"),
     Output("simulation-results", "data"),
     Output("status-message", "children"),
     Output("save-button", "disabled")],
    [Input("render-button", "n_clicks")],
    [State("surface-mesh", "value"),
     State("octaves", "value"),
     State("scale", "value"),
     State("height", "value"),
     State("z-height", "value"),
     State("rays-mesh", "value"),
     State("theta", "value"),
     State("phi", "value"),
     State("n-red", "value"),
     State("n-green", "value"),
     State("n-blue", "value"),
     State("light-ray-count", "value"),
     State("x-min", "value"),
     State("x-max", "value"),
     State("y-min", "value"),
     State("y-max", "value"),
     State("z-screen", "value"),
     State("random-seed", "value")]
)
def render_simulation(n_clicks, surface_mesh, octaves, scale, height, z_height, rays_mesh, 
                     theta, phi, n_red, n_green, n_blue, light_ray_count, 
                     x_min, x_max, y_min, y_max, z_screen, random_seed):
    if n_clicks is None:
        # Initial load - return empty figures
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Click 'Render' to generate visualization",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return empty_fig, empty_fig, None, "Ready to simulate", True
    
    # Calculate ray angle from theta and phi
    # Convert to radians
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    
    # Calculate the direction vector
    # For phi, 0 is horizontal and 90 is vertical downward
    x_component = np.sin(phi_rad) * np.cos(theta_rad)
    y_component = np.sin(phi_rad) * np.sin(theta_rad)
    z_component = -np.cos(phi_rad)  # Negative because we want downward to be negative z
    
    rays_angle = [0, 0, -1]
    
    # Fixed parameters
    n_1 = 1  # Air refractive index
    
    # Create a temporary data folder if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Generate surface
    x = np.linspace(x_min, x_max, surface_mesh)
    y = np.linspace(y_min, y_max, surface_mesh)
    grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    
    # Create a temporary path for this calculation
    temp_surface_path = os.path.join('data', f'temp_surface_{random_seed}.npy')
    
    surface = el.create_perlin_noise_grid(
        grid, octaves=octaves, scale=scale, height=height, 
        z_height=z_height, random_seed=random_seed, save=temp_surface_path
    )
    
    # Define the rays 
    x_ray = np.linspace(x_min, x_max, rays_mesh)
    y_ray = np.linspace(y_min, y_max, rays_mesh)
    ray_coordinates = np.array(np.meshgrid(x_ray, y_ray)).T.reshape(-1, 2)
    ray_3coordinates = el.create_perlin_noise_grid(
        ray_coordinates, octaves=octaves, scale=scale, height=height, 
        z_height=z_height, random_seed=random_seed
    )
    light_rays = np.array([rays_angle] * len(ray_3coordinates)).reshape(-1, 3)
    
    # Calculate the normals
    normals = el.compute_normal_vectors_combined(surface, ray_coordinates, n_jobs=-1, batch_size=100)
    
    # Calculate the refracted rays for each color
    refracted_red = el.refract_light(light_rays, normals, n_1, n_red)
    refracted_green = el.refract_light(light_rays, normals, n_1, n_green)
    refracted_blue = el.refract_light(light_rays, normals, n_1, n_blue)
    
    # Calculate the intersections with the screen for each color
    intersection_red = el.compute_intersection_with_plane(refracted_red, ray_3coordinates, z_screen)
    intersection_green = el.compute_intersection_with_plane(refracted_green, ray_3coordinates, z_screen)
    intersection_blue = el.compute_intersection_with_plane(refracted_blue, ray_3coordinates, z_screen)
    
    # Create RGB caustic visualization
    rgb_fig = create_rgb_caustic_plot(intersection_red, intersection_green, intersection_blue)
    
    # Create 3D visualization
    fig_3d = create_3d_visualization(
        surface, ray_3coordinates, light_rays, 
        intersection_red, intersection_green, intersection_blue, 
        z_screen, light_ray_count
    )
    
    # Store all parameters and results for saving later
    all_params = {
        "surface_mesh": surface_mesh,
        "octaves": octaves,
        "scale": scale,
        "height": height,
        "z_height": z_height,
        "rays_mesh": rays_mesh,
        "theta": theta,
        "phi": phi,
        "n_red": n_red,
        "n_green": n_green,
        "n_blue": n_blue,
        "light_ray_count": light_ray_count,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_screen": z_screen,
        "random_seed": random_seed,
    }
    
    # Store results for save button
    simulation_data = {
        "params": all_params,
        "temp_surface_path": temp_surface_path,
        "intersection_red": intersection_red.tolist(),
        "intersection_green": intersection_green.tolist(),
        "intersection_blue": intersection_blue.tolist(),
    }
    
    return fig_3d, rgb_fig, simulation_data, "Simulation complete! Click 'Save' to save results.", False


def create_rgb_caustic_plot(intersection_red, intersection_green, intersection_blue):
    """Create a 2D RGB caustic plot using KDE."""
    
    # Determine bounds from all interactions
    x_min = min(intersection_red[:, 0].min(), intersection_green[:, 0].min(), intersection_blue[:, 0].min())
    x_max = max(intersection_red[:, 0].max(), intersection_green[:, 0].max(), intersection_blue[:, 0].max())
    y_min = min(intersection_red[:, 1].min(), intersection_green[:, 1].min(), intersection_blue[:, 1].min())
    y_max = max(intersection_red[:, 1].max(), intersection_green[:, 1].max(), intersection_blue[:, 1].max())
    
    resolution = 300  # Use higher resolution for better visualization
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X_caustic, Y_caustic = np.meshgrid(x_grid, y_grid)
    points = np.vstack([X_caustic.flatten(), Y_caustic.flatten()])
    
    bw = 0.08  # Bandwidth for KDE
    # Compute KDE for each channel
    kde_red = gaussian_kde(intersection_red.T, bw_method=bw)
    kde_green = gaussian_kde(intersection_green.T, bw_method=bw)
    kde_blue = gaussian_kde(intersection_blue.T, bw_method=bw)
    
    density_red = kde_red(points).reshape(resolution, resolution)
    density_green = kde_green(points).reshape(resolution, resolution)
    density_blue = kde_blue(points).reshape(resolution, resolution)
    
    # Global normalization
    global_max = max(density_red.max(), density_green.max(), density_blue.max()) + 1e-10
    scaled_red = density_red * (255.0 / global_max)
    scaled_green = density_green * (255.0 / global_max)
    scaled_blue = density_blue * (255.0 / global_max)
    
    # Create RGB array for visualization
    rgb_array = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    rgb_array[:, :, 0] = scaled_red.T  # Red channel
    rgb_array[:, :, 1] = scaled_green.T  # Green channel
    rgb_array[:, :, 2] = scaled_blue.T  # Blue channel
    
    # Create a figure with the RGB image
    fig = px.imshow(rgb_array, origin='lower')
    fig.update_layout(
        title="RGB Caustic Pattern",
        xaxis=dict(title="X", showticklabels=False),
        yaxis=dict(title="Y", showticklabels=False),
        coloraxis_showscale=False
    )
    
    return fig


def create_3d_visualization(function_grid, ray_3coordinates, light_beams, 
                           interactions_red, interactions_green, interactions_blue,
                           z_value, ray_count):
    """
    Create a 3D visualization with surface, caustics, and ray paths.
    """
    # Prepare surface data
    xs = function_grid[:, 0]
    ys = function_grid[:, 1]
    zs = function_grid[:, 2]
    x_unique = np.sort(np.unique(xs))
    y_unique = np.sort(np.unique(ys))
    X, Y = np.meshgrid(x_unique, y_unique)
    Z = zs.reshape(len(y_unique), len(x_unique))
    
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
    
    resolution = 100  # Lower resolution for 3D visualization for performance
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X_caustic, Y_caustic = np.meshgrid(x_grid, y_grid)
    points = np.vstack([X_caustic.flatten(), Y_caustic.flatten()])
    
    bw = 0.08  # Bandwidth for KDE
    # Compute KDE for each channel
    kde_red = gaussian_kde(interactions_red.T, bw_method=bw)
    kde_green = gaussian_kde(interactions_green.T, bw_method=bw)
    kde_blue = gaussian_kde(interactions_blue.T, bw_method=bw)
    
    density_red = kde_red(points).reshape(resolution, resolution)
    density_green = kde_green(points).reshape(resolution, resolution)
    density_blue = kde_blue(points).reshape(resolution, resolution)
    
    # Global normalization
    global_max = max(density_red.max(), density_green.max(), density_blue.max()) + 1e-10
    scaled_red = density_red * (255.0 / global_max)
    scaled_green = density_green * (255.0 / global_max)
    scaled_blue = density_blue * (255.0 / global_max)
    
    # Create RGB values for caustic plane
    rgb_values = []
    for i in range(resolution):
        row = []
        for j in range(resolution):
            r = int(scaled_red[i, j])
            g = int(scaled_green[i, j])
            b = int(scaled_blue[i, j])
            row.append(f'rgb({r},{g},{b})')
        rgb_values.append(row)
    
    # Add caustic plane
    fig.add_trace(go.Surface(
        x=X_caustic,
        y=Y_caustic,
        z=np.full((resolution, resolution), z_value),
        surfacecolor=scaled_red + scaled_green + scaled_blue,  # Use combined intensity
        colorscale='Viridis',
        showscale=False,
        opacity=0.8,
        name='RGB Caustics'
    ))
    
    # Draw colored rays for visualization
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
    
    # Update layout
    fig.update_layout(
        title="3D Caustics and Ray Paths",
        scene=dict(
            aspectmode='data',
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig


@app.callback(
    Output("status-message", "children", allow_duplicate=True),
    [Input("save-button", "n_clicks")],
    [State("simulation-results", "data"),
     State("save-filename", "value")],
    prevent_initial_call=True
)
def save_simulation_data(n_clicks, simulation_data, filename):
    if n_clicks is None or simulation_data is None:
        raise PreventUpdate
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Create directory for this simulation
    save_dir = os.path.join('data', filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Extract data
    params = simulation_data["params"]
    intersection_red = np.array(simulation_data["intersection_red"])
    intersection_green = np.array(simulation_data["intersection_green"])
    intersection_blue = np.array(simulation_data["intersection_blue"])
    
    # Save parameters as JSON
    with open(os.path.join(save_dir, 'parameters.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # Save intersections
    np.save(os.path.join(save_dir, 'intersection_red.npy'), intersection_red)
    np.save(os.path.join(save_dir, 'intersection_green.npy'), intersection_green)
    np.save(os.path.join(save_dir, 'intersection_blue.npy'), intersection_blue)
    
    # Save surface (copy from temp)
    if os.path.exists(simulation_data["temp_surface_path"]):
        surface = np.load(simulation_data["temp_surface_path"])
        np.save(os.path.join(save_dir, 'surface.npy'), surface)
    
    # Generate and save the PDF plot
    el.plot_interactions_rgb(
        intersection_red, intersection_green, intersection_blue, 
        kde=True, save=os.path.join(save_dir, 'caustic_rgb.pdf')
    )
    
    # Create and save 3D plot as HTML
    fig_3d = create_3d_visualization(
        np.load(os.path.join(save_dir, 'surface.npy')),
        el.create_perlin_noise_grid(
            np.array(np.meshgrid(
                np.linspace(params["x_min"], params["x_max"], params["rays_mesh"]),
                np.linspace(params["y_min"], params["y_max"], params["rays_mesh"])
            )).T.reshape(-1, 2),
            octaves=params["octaves"],
            scale=params["scale"],
            height=params["height"],
            z_height=params["z_height"],
            random_seed=params["random_seed"]
        ),
        np.array([[
            np.sin(np.radians(params["phi"])) * np.cos(np.radians(params["theta"])),
            np.sin(np.radians(params["phi"])) * np.sin(np.radians(params["theta"])),
            -np.cos(np.radians(params["phi"]))
        ]] * (params["rays_mesh"]**2)).reshape(-1, 3),
        intersection_red, intersection_green, intersection_blue,
        params["z_screen"], params["light_ray_count"]
    )
    fig_3d.write_html(os.path.join(save_dir, '3D_caustic_rgb.html'))
    
    return f"Simulation data saved to: {save_dir}"


if __name__ == '__main__':
    app.run(debug=True)