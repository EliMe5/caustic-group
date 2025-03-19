import numpy as np
import matplotlib.pyplot as plt
import functions.eliott as el

# Define the function grid
x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
surface = el.create_perlin_noise_grid(grid, octaves=3, scale=0.1, height=0.5, z_height=2, random_seed=42)

print(surface)
# Plot the surface as a mesh
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
Z = surface[:, 2].reshape(X.shape)
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.show()
