import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import os
from functions import math_raycast as mr

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create the app layout with all input components
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Interactive Caustics Simulation"),
            html.Hr(),
        ], width=12)
    ]),
    
    # Main layout with parameters on left, plots on right
    dbc.Row([
        # Left column - All control parameters
        dbc.Col([
            # Surface function input
            html.H4("Surface Parameters"),
            dbc.Label("Surface Function f(x, y):"),
            dbc.Input(
                id="surface-function",
                type="text",
                value="1 + 0.5 * np.sin(4 * x) * np.sin(y)",
                placeholder="Enter a function of x and y",
                className="mb-3"
            ),
            
            # Ray parameters
            html.H4("Ray Parameters"),
            dbc.Label("Resolution:"),
            dcc.Slider(
                id="resolution-slider",
                min=200,
                max=5000,
                step=100,
                value=1000,
                marks={i: str(i) for i in range(200, 5001, 800)},
                className="mb-3"
            ),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("n_0 (Incident medium):"),
                    dbc.Input(id="n0-input", type="number", value=1, min=1, max=2, step=0.01),
                ], width=6),
                dbc.Col([
                    dbc.Label("n_red (Red light):"),
                    dbc.Input(id="nred-input", type="number", value=1.33, min=1, max=2, step=0.01),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("n_green (Green light):"),
                    dbc.Input(id="ngreen-input", type="number", value=1.45, min=1, max=2, step=0.01),
                ], width=6),
                dbc.Col([
                    dbc.Label("n_blue (Blue light):"),
                    dbc.Input(id="nblue-input", type="number", value=1.5, min=1, max=2, step=0.01),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Label("Light Ray Count:"),
            dcc.Slider(
                id="light-ray-count-slider",
                min=5,
                max=50,
                step=1,
                value=20,
                marks={i: str(i) for i in range(5, 51, 5)},
                className="mb-4"
            ),
            
            # Advanced parameters
            html.H4("Advanced Parameters"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("x_min:"),
                    dbc.Input(id="xmin-input", type="number", value=-5, step=0.1),
                ], width=6),
                dbc.Col([
                    dbc.Label("x_max:"),
                    dbc.Input(id="xmax-input", type="number", value=5, step=0.1),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("y_min:"),
                    dbc.Input(id="ymin-input", type="number", value=-5, step=0.1),
                ], width=6),
                dbc.Col([
                    dbc.Label("y_max:"),
                    dbc.Input(id="ymax-input", type="number", value=5, step=0.1),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("sigma:"),
                    dbc.Input(id="sigma-input", type="number", value=0.5, min=0.1, max=2, step=0.1),
                ], width=6),
                dbc.Col([
                    dbc.Label("z_surface:"),
                    dbc.Input(id="zsurface-input", type="number", value=-1, step=0.1),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Label("brighten:"),
            dcc.Slider(
                id="brighten-slider",
                min=1,
                max=10,
                step=0.1,
                value=3,
                marks={i: str(i) for i in range(1, 11)},
                className="mb-4"
            ),
            
            # Save options
            html.H4("Save Options"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Folder Name:"),
                    dbc.Input(id="folder-name-input", type="text", value="simulation_1"),
                ], width=8),
                dbc.Col([
                    dbc.Button("Save", id="save-button", color="primary", className="mt-4"),
                ], width=4),
            ], className="mb-3"),
            
            # Notification area for save status
            html.Div(id="save-notification", className="mt-3"),
            
        ], width=4, className="pr-4"),
        
        # Right column - Plots
        dbc.Col([
            # Loading indicator
            dcc.Loading(
                id="loading",
                type="circle",
                children=[
                    # Plot output areas
                    html.Div([
                        html.H4("3D Surface with Light Ray Interactions"),
                        dcc.Graph(id="interactions-plot", style={"height": "500px"}),
                    ], className="mb-4"),
                    
                    html.Div([
                        html.H4("RGB Kernel Density Estimation of Light Refraction"),
                        dcc.Graph(id="caustic-map-plot", style={"height": "500px"}),
                    ]),
                ]
            ),
        ], width=8),
    ]),
    
], fluid=True)

def create_interactions_plot(
    f, X, Y, colour_map, x_lim, y_lim, 
    intersections_red, intersections_green, intersections_blue, 
    light_ray_count, z_surface
):
    """Create a Plotly figure for the 3D surface with light ray interactions."""
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
    
    return fig

def create_caustic_map_plot(colour_map, x_lim, y_lim):
    """Create a Plotly figure for the RGB kernel density estimation."""
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
    
    return fig

@callback(
    [Output("interactions-plot", "figure"),
     Output("caustic-map-plot", "figure")],
    [Input("surface-function", "value"),
     Input("resolution-slider", "value"),
     Input("n0-input", "value"),
     Input("nred-input", "value"),
     Input("nblue-input", "value"),
     Input("ngreen-input", "value"),
     Input("light-ray-count-slider", "value"),
     Input("xmin-input", "value"),
     Input("xmax-input", "value"),
     Input("ymin-input", "value"),
     Input("ymax-input", "value"),
     Input("sigma-input", "value"),
     Input("zsurface-input", "value"),
     Input("brighten-slider", "value")]
)
def update_plots(
    surface_function, resolution, n_0, n_red, n_blue, n_green, 
    light_ray_count, x_min, x_max, y_min, y_max, sigma, z_surface, brighten
):
    # Create the surface function from the input string
    import numpy as np  # Import inside function for eval to work
    
    # Define the function using the input string
    def f(x, y):
        return eval(surface_function)
    
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
    refracted_vectors_green = mr.refracted_ray(n_0, n_green, unit_normals)
    refracted_vectors_blue = mr.refracted_ray(n_0, n_blue, unit_normals)
    
    # Find the (x, y) intersection of the refracted rays with the plane z=z_surface
    intersections_red = mr.intersection_with_plane(refracted_vectors_red, f, X, Y, z_surface)
    intersections_green = mr.intersection_with_plane(refracted_vectors_green, f, X, Y, z_surface)
    intersections_blue = mr.intersection_with_plane(refracted_vectors_blue, f, X, Y, z_surface)

    # Create the colour map
    colour_map = mr.create_colour_map(
        intersections_red, intersections_green, intersections_blue, 
        bins=round(resolution*0.5), x_lim=[x_min, x_max], y_lim=[y_min, y_max], 
        sigma=sigma, brighten=brighten
    )

    # Create the interaction plot
    interactions_fig = create_interactions_plot(
        f, X, Y, colour_map, [x_min, x_max], [y_min, y_max], 
        intersections_red, intersections_green, intersections_blue, 
        light_ray_count, z_surface
    )
    
    # Create the caustic map plot
    caustic_fig = create_caustic_map_plot(
        colour_map, [x_min, x_max], [y_min, y_max]
    )
    
    return interactions_fig, caustic_fig

@callback(
    Output("save-notification", "children"),
    [Input("save-button", "n_clicks")],
    [State("surface-function", "value"),
     State("resolution-slider", "value"),
     State("n0-input", "value"),
     State("nred-input", "value"),
     State("nblue-input", "value"),
     State("ngreen-input", "value"),
     State("light-ray-count-slider", "value"),
     State("xmin-input", "value"),
     State("xmax-input", "value"),
     State("ymin-input", "value"),
     State("ymax-input", "value"),
     State("sigma-input", "value"),
     State("zsurface-input", "value"),
     State("brighten-slider", "value"),
     State("folder-name-input", "value"),
     State("interactions-plot", "figure"),
     State("caustic-map-plot", "figure")]
)
def save_simulation(
    n_clicks, surface_function, resolution, n_0, n_red, n_blue, n_green, 
    light_ray_count, x_min, x_max, y_min, y_max, sigma, z_surface, brighten,
    folder_name, interactions_plot, caustic_map_plot
):
    if n_clicks is None:
        return ""
    
    # Ensure the /data/ folder and the specified folder_name exist
    output_folder = os.path.join('data', folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Save the configuration parameters to a JSON file
    config = {
        "surface_function": surface_function,
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
    
    # Save the plots as HTML files
    interactions_path = os.path.join(output_folder, 'interactions_plot.html')
    caustic_map_path = os.path.join(output_folder, 'caustic_map.html')
    
    import plotly.io as pio
    
    # Create figures from the stored figure data
    interactions_fig = go.Figure(interactions_plot)
    caustic_fig = go.Figure(caustic_map_plot)
    
    # Save as HTML files
    interactions_fig.write_html(interactions_path)
    caustic_fig.write_html(caustic_map_path)
    
    # Create the surface function from the input string for mr to use
    import numpy as np
    def f(x, y):
        return eval(surface_function)
    
    # Recompute all calculations to get the colour_map for saving
    x_range = [x_min * 1.4, x_max * 1.4]
    y_range = [y_min * 1.4, y_max * 1.4]
    ray_number = round(resolution * 1.4)
    
    x_grid = np.linspace(x_range[0], x_range[1], ray_number)
    y_grid = np.linspace(y_range[0], y_range[1], ray_number)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    unit_normals = mr.normalized_normal(f, X, Y, h=1e-4)
    
    refracted_vectors_red = mr.refracted_ray(n_0, n_red, unit_normals)
    refracted_vectors_green = mr.refracted_ray(n_0, n_green, unit_normals)
    refracted_vectors_blue = mr.refracted_ray(n_0, n_blue, unit_normals)
    
    intersections_red = mr.intersection_with_plane(refracted_vectors_red, f, X, Y, z_surface)
    intersections_green = mr.intersection_with_plane(refracted_vectors_green, f, X, Y, z_surface)
    intersections_blue = mr.intersection_with_plane(refracted_vectors_blue, f, X, Y, z_surface)
    
    colour_map = mr.create_colour_map(
        intersections_red, intersections_green, intersections_blue, 
        bins=round(resolution*0.5), x_lim=[x_min, x_max], y_lim=[y_min, y_max], 
        sigma=sigma, brighten=brighten
    )
    
    caustic_map_pdf_path = os.path.join(output_folder, 'caustic_map.pdf')
    mr.save_caustic_map(colour_map, [x_min, x_max], [y_min, y_max], caustic_map_pdf_path)
    
    return html.Div([
        html.P(f"Saved to folder: {output_folder}", className="text-success")
    ])

if __name__ == "__main__":
    app.run(debug=True)