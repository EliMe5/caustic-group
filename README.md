# Interactive Caustics Simulation Dashboard

This application provides an interactive interface for simulating and visualizing light refraction through surfaces, generating caustic patterns with different wavelengths (colors).

## Features

- **Interactive Parameter Controls**: Adjust surface properties, ray parameters, and refractive indices in real-time
- **Dynamic Visualization**: See both 3D ray paths and 2D caustic patterns
- **Save Functionality**: Save simulation results with custom filenames and proper configuration management

## How to Use

1. **Set Parameters**: Adjust the sliders and input fields on the left panel to configure your simulation
   - Surface parameters control the Perlin noise surface generation
   - Ray parameters control the light sources and refractive properties
   - Advanced parameters adjust boundaries and random seeds

2. **Render**: Click the "Render" button to run the simulation with your chosen parameters
   - A loading indicator will appear while the calculation is in progress
   - Once complete, the 3D visualization and RGB caustic pattern will update

3. **Save Results**: Enter a filename and click the "Save" button to save your results
   - All data will be stored in a folder with your chosen name inside the `data` directory
   - Parameters will be saved as a JSON file
   - Intersection data will be saved as NumPy arrays
   - Both PDF and HTML visualizations will be generated

## Parameter Descriptions

### Surface Parameters
- **Surface Mesh Resolution**: Number of points to use when generating the surface (higher = more detailed but slower)
- **Perlin Noise Octaves**: Number of octaves for the Perlin noise (affects surface complexity)
- **Perlin Noise Scale**: Scale factor for the noise (smaller = more detailed surface)
- **Surface Height**: Height of the surface bumps
- **Z Height Maximum**: Maximum height of the surface

### Ray Parameters
- **Rays Mesh Resolution**: Number of light rays to simulate (higher = more accurate caustics but slower)
- **Theta Angle**: Horizontal angle of incident light (0-360°)
- **Phi Angle**: Vertical angle of incident light (0-90°, where 90° is straight down)
- **Refractive Indices**: Different values for red, green and blue light (creates dispersion effects)
- **Light Ray Count**: Number of rays to display in the 3D visualization

### Advanced Parameters
- **X/Y Min/Max**: Boundaries of the simulation area
- **Z Screen**: Height of the plane where caustics are projected
- **Random Seed**: Seed for reproducible noise generation

## Requirements

- Python 3.6+
- Dash
- Plotly
- NumPy
- SciPy
- Dash Bootstrap Components

## Installation

1. Clone this repository
2. Install required packages: `pip install dash dash-bootstrap-components plotly numpy scipy`
3. Run the app: `python app.py`
4. Open your browser to `http://127.0.0.1:8050/`

## Notes

- Calculations for high-resolution surfaces and many rays can be compute-intensive
- The simulation data is saved with proper organization
- For best visual results, use a larger difference between refractive indices
