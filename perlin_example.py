import numpy as np
import os
import functions.eliott as el

# Setup the Perlin parameters (Note these will be saved in the data folder)
x_min = -5
x_max = 5
y_min = -5
y_max = 5
surface_mesh = 1000
octaves = 3
scale = 0.1
height = 0.5
z_height = 2
random_seed = 42
rays_mesh = 100

# Setup the screen parameters
z_screen = 0
rays_angle = [0, 0, -1]
n_1 = 1
n_2 = 1.5
light_ray_count = 20

# Creates the folder for these parameters
folder = os.path.join(os.path.dirname(__file__), f'data/x{x_min}x{x_max}y{y_min}y{y_max}s{surface_mesh}o{octaves}sc{scale}h{height}zh{z_height}rs{random_seed}rm{rays_mesh}')
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
    surface = el.create_perlin_noise_grid(grid, octaves=octaves, scale=scale, height=height, z_height=z_height, random_seed=random_seed, save=surface_path)

# Define the rays 
x_ray = np.linspace(x_min, x_max, rays_mesh)
y_ray = np.linspace(y_min, y_max, rays_mesh)
ray_coordinates = np.array(np.meshgrid(x_ray, y_ray)).T.reshape(-1, 2)
ray_3coordinates = el.create_perlin_noise_grid(ray_coordinates, octaves=octaves, scale=scale, height=height, z_height=z_height, random_seed=random_seed)
light_rays = np.array(rays_angle * len(ray_3coordinates)).reshape(-1, 3)

# Calculate the rays path
normals_path = os.path.join(folder, 'normals.npy')
if os.path.exists(normals_path):
    normals = np.load(normals_path)
else:
    normals = el.compute_normal_vectors(surface, ray_coordinates)
    np.save(normals_path, normals)
refracted = el.refract_light(light_rays, normals, n_1, n_2)
intersection = el.compute_intersection_with_plane(refracted, ray_3coordinates, z_screen)

# Plot the intersection
plot_save = os.path.join(folder, 'caustic.pdf')
el.plot_interactions(intersection, kde=True, save=plot_save)

# Plot the complete 3D interaction
plot_save = os.path.join(folder, '3D_caustic.pdf')
el.plot_interactions_3D(surface, ray_3coordinates, light_rays, intersection, z_screen, light_ray_count, save=plot_save)