import numpy as np
import os
import functions.eliott as el

# General static scene parameters
x_min = -5 # Defined manually in code
x_max = 5 # Defined manually in code
y_min = -5 # Defined manually in code
y_max = 5 # Defined manually in code
random_seed = 42 # Defined manually in the code
z_screen = 0 # Defined manually in the code

# Setup the Perlin parameters (Note these will be saved in the data folder)
surface_mesh = 4000 # Should be 10 times the amount of rays_mesh
octaves = 30000 # This should have a slider (int)
scale = 0.1 # This should have a slider (float)
height = 0.8 # This should have a slider it cant be lower than 0 and higher than z_height (float)
z_height = 2 # This should be a slider (float) has to be larger than 0
rays_mesh = 200 # should be a int box input not a slider

# Setup the screen parameters
rays_angle = [0, 0 ,-1] # This should be 2 sliders, one for theta and one for phi where theta is the angle in the x-y plane and phi is the angle from the x-y plane, phi can go from 0 to 90 degrees (has to point down) and theta 0 to 360
n_1 = 1 # Defined manually in the code

# Define separate refractive indices for each color
n_red = 1.33    # This should be a float box input 
n_green = 1.33  # This should be a float box input
n_blue = 1.33   # This should be a float box input
light_ray_count = 400 # This should be an int slider

# Creates the folder for these parameters
folder = os.path.join(os.path.dirname(__file__), f'data/x{x_min}x{x_max}y{y_min}y{y_max}s{surface_mesh}o{octaves}sc{scale}h{height}zh{z_height}rs{random_seed}rm{rays_mesh}_colored')
if not os.path.exists(folder):
    os.makedirs(folder)

# Check if the surface already exists, otherwise create it
surface_path = os.path.join(folder, 'surface.npy')
if os.path.exists(surface_path):
    surface = np.load(surface_path)
else:
    x = np.linspace(x_min, x_max, surface_mesh)
    y = np.linspace(y_min, y_max, surface_mesh)
    grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    surface = np.array([[point[0], point[1], 0.5 * np.sin(4 * point[0]) * np.sin(point[1]) + z_height] for point in grid])
    np.save(surface_path, surface)

# Define the rays 
x_ray = np.linspace(x_min, x_max, rays_mesh)
y_ray = np.linspace(y_min, y_max, rays_mesh)
ray_coordinates = np.array(np.meshgrid(x_ray, y_ray)).T.reshape(-1, 2)
ray_3coordinates = el.create_perlin_noise_grid(ray_coordinates, octaves=octaves, scale=scale, height=height, z_height=z_height, random_seed=random_seed)
light_rays = np.array(rays_angle * len(ray_3coordinates)).reshape(-1, 3)

# Calculate the normals
normals_path = os.path.join(folder, 'normals.npy')
if os.path.exists(normals_path):
    normals = np.load(normals_path)
else:
    normals = el.compute_normal_vectors_combined(surface, ray_coordinates, n_jobs=-1, batch_size=100)
    np.save(normals_path, normals)

# Calculate the refracted rays for each color
refracted_red = el.refract_light(light_rays, normals, n_1, n_red)

# Calculate the intersections with the screen for each color
intersection_red = el.compute_intersection_with_plane(refracted_red, ray_3coordinates, z_screen)

# Save intersections to files
np.save(os.path.join(folder, 'intersection_red.npy'), intersection_red)

# Plot each colour combined
el.plot_interactions(intersection_red, kde=False)

# Generate 3D plotly visualizations for each color
#el.plot_interactions_3D_plotly(surface, ray_3coordinates, light_rays, intersection_red, z_screen, light_ray_count)