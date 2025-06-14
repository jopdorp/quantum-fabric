#!/usr/bin/env python3
"""
3D Quantum visualization using Plotly offline instead of Dash.
This bypasses potential Dash 3D rendering issues.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import time
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_hybrid_molecular_simulation import create_atom_simulation, AtomConfig
from config import SIZE_X, SIZE_Y, SIZE_Z

class OfflinePlotlyQuantumViz:
    """3D quantum visualization using Plotly offline mode."""
    
    def __init__(self):
        """Initialize the visualization."""
        print("Creating 3D quantum simulation...")
        
        # Create a hydrogen atom at the center
        hydrogen_config = AtomConfig(
            atomic_number=1, 
            position=(SIZE_X//2, SIZE_Y//2, SIZE_Z//2)
        )
        
        self.simulation = create_atom_simulation(hydrogen_config)
        print(f"Simulation created with {len(self.simulation.nuclei)} nuclei and {len(self.simulation.electrons)} electrons")
        
        self.frame_count = 0
        
    def extract_3d_data(self, mode='probability', threshold=0.00001):
        """Extract 3D data from the simulation."""
        try:
            # Get wavefunction
            wavefunction = self.simulation.get_combined_wavefunction()
            
            if isinstance(wavefunction, type(None)):
                print("No wavefunction available")
                return [], [], [], [], 'Hot', 'Error', 0
            
            print(f"Wavefunction shape: {wavefunction.shape}")
            print(f"Wavefunction dtype: {wavefunction.dtype}")
            
            # Convert to numpy first for all operations
            if hasattr(wavefunction, 'detach'):
                wavefunction_np = wavefunction.detach().cpu().numpy()
            else:
                wavefunction_np = np.array(wavefunction)
            
            print(f"Wavefunction range: {np.min(wavefunction_np)} to {np.max(wavefunction_np)}")
            
            # Process based on mode
            if mode == 'probability':
                data = np.abs(wavefunction_np)**2
                colorscale = 'Hot'
                title_suffix = 'Probability Density'
            elif mode == 'real':
                data = np.real(wavefunction_np)
                colorscale = 'RdBu'
                title_suffix = 'Real Part'
            elif mode == 'imaginary':
                data = np.imag(wavefunction_np)
                colorscale = 'RdBu'
                title_suffix = 'Imaginary Part'
            elif mode == 'phase':
                data = np.angle(wavefunction_np)
                colorscale = 'HSV'
                title_suffix = 'Phase'
            else:
                data = np.abs(wavefunction_np)**2
                colorscale = 'Hot'
                title_suffix = 'Probability Density'
            print(f"Data range after processing: {data.min():.6f} to {data.max():.6f}")
            
            # Subsample for performance (every 4th point)
            step = 4
            data_sub = data[::step, ::step, ::step]
            print(f"Subsampled data shape: {data_sub.shape}")
            print(f"Subsampled data range: {data_sub.min():.6f} to {data_sub.max():.6f}")
            
            max_val = np.abs(data_sub).max()
            print(f"Max absolute value: {max_val:.6f}")
            
            if max_val > 0:
                print(f"Threshold: {threshold:.8f}")
                mask = np.abs(data_sub) > threshold
                points_above = np.sum(mask)
                print(f"Points above threshold: {points_above}")
                
                if points_above == 0:
                    effective_threshold = max_val * 0.0001
                    mask = np.abs(data_sub) > effective_threshold
                    print(f"Using lower threshold: {effective_threshold:.8f}")
                    print(f"Points with lower threshold: {np.sum(mask)}")
                
                # Get coordinates of significant points
                x_coords, y_coords, z_coords = np.where(mask)
                values = data_sub[mask]
                
                # Scale coordinates back and normalize to smaller space
                x_coords = x_coords * step
                y_coords = y_coords * step
                z_coords = z_coords * step
                
                # Normalize to 0-10 space like the working test
                x_coords = (x_coords / 255.0) * 10.0
                y_coords = (y_coords / 255.0) * 10.0
                z_coords = (z_coords / 255.0) * 10.0
                
                print(f"Final number of points: {len(values)}")
                print(f"Coordinate ranges: X[{x_coords.min():.1f}, {x_coords.max():.1f}], Y[{y_coords.min():.1f}, {y_coords.max():.1f}], Z[{z_coords.min():.1f}, {z_coords.max():.1f}]")
                print(f"Value range: [{values.min():.8f}, {values.max():.8f}]")
                
                # Limit number of points for performance
                max_points = 50000
                if len(values) > max_points:
                    indices = np.random.choice(len(values), max_points, replace=False)
                    x_coords = x_coords[indices]
                    y_coords = y_coords[indices]
                    z_coords = z_coords[indices]
                    values = values[indices]
                    print(f"Reduced to {len(values)} points for performance")
                
                return x_coords, y_coords, z_coords, values, colorscale, title_suffix, max_val
            else:
                print("No data found - max value is 0")
                return [], [], [], [], colorscale, title_suffix, 0
                
        except Exception as e:
            print(f"Error extracting 3D data: {e}")
            import traceback
            traceback.print_exc()
            return [], [], [], [], 'Hot', 'Error', 0
    
    def create_figure(self, mode='probability', threshold=0.00001):
        """Create a 3D figure."""
        # Evolve simulation
        self.simulation.evolve_step(self.frame_count)
        self.frame_count += 1
        
        # Extract data
        x, y, z, values, colorscale, title_suffix, max_val = self.extract_3d_data(mode, threshold)
        
        # Create figure
        fig = go.Figure()
        
        # Always add test points in the new coordinate system
        fig.add_trace(go.Scatter3d(
            x=[5], y=[5], z=[5],  # Center of 0-10 space
            mode='markers',
            marker=dict(size=25, color='lime', opacity=1.0, symbol='diamond'),
            name='Test Center Point'
        ))
        
        # Add quantum data if available
        if len(values) > 0:
            # Ensure numpy arrays
            x = np.asarray(x)
            y = np.asarray(y) 
            z = np.asarray(z)
            values = np.asarray(values)
            
            print(f"Data types: x={x.dtype}, y={y.dtype}, z={z.dtype}, values={values.dtype}")
            
            # Normalize colors
            color_values = values / values.max() if values.max() > 0 else values
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=15,  # Much larger points
                    color=color_values,
                    colorscale=colorscale,
                    showscale=True,
                    opacity=1.0,  # Full opacity
                    line=dict(width=2, color='black'),  # Thicker black outline
                    colorbar=dict(title=title_suffix)
                ),
                name='Wavefunction'
            ))
            print(f"Added {len(values)} quantum data points")
        
        # Add nuclei
        for i, nucleus in enumerate(self.simulation.nuclei):
            nucleus_x = nucleus.position[0].item() if hasattr(nucleus.position[0], 'item') else nucleus.position[0]
            nucleus_y = nucleus.position[1].item() if hasattr(nucleus.position[1], 'item') else nucleus.position[1]
            nucleus_z = nucleus.position[2].item() if len(nucleus.position) > 2 and hasattr(nucleus.position[2], 'item') else (nucleus.position[2] if len(nucleus.position) > 2 else 0)
            
            # Normalize nucleus position to 0-10 space
            nucleus_x = (nucleus_x / 255.0) * 10.0
            nucleus_y = (nucleus_y / 255.0) * 10.0
            nucleus_z = (nucleus_z / 255.0) * 10.0
            
            fig.add_trace(go.Scatter3d(
                x=[nucleus_x], y=[nucleus_y], z=[nucleus_z],
                mode='markers',
                marker=dict(size=30, color='red', symbol='circle', line=dict(width=3, color='darkred')),
                name=f'Nucleus {i+1}'
            ))
        
        # Set layout - use working test's simple approach
        fig.update_layout(
            title=f'3D Quantum Simulation - {title_suffix} (Frame {self.frame_count})',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='cube'
            ),
            template='plotly_dark'
        )
        
        return fig
    
    def run_single_frame(self, mode='probability', threshold=0.00001):
        """Create and save a single frame."""
        fig = self.create_figure(mode, threshold)
        filename = f'quantum_3d_frame_{self.frame_count}.html'
        pyo.plot(fig, filename=filename, auto_open=True)
        print(f"Saved {filename}")
        return filename
    
    def run_animation(self, frames=10, mode='probability', threshold=0.00001):
        """Create multiple frames and save them."""
        filenames = []
        for i in range(frames):
            print(f"\nGenerating frame {i+1}/{frames}")
            filename = self.run_single_frame(mode, threshold)
            filenames.append(filename)
            time.sleep(1)  # Brief pause
        
        print(f"\nGenerated {len(filenames)} frames:")
        for f in filenames:
            print(f"  {f}")
        
        return filenames

def main():
    """Main function."""
    print("Creating offline 3D quantum visualization...")
    viz = OfflinePlotlyQuantumViz()
    
    print("\nGenerating single frame...")
    viz.run_single_frame(mode='probability', threshold=0.00001)
    
    print("\nDone! Check the HTML file that opened in your browser.")

if __name__ == "__main__":
    main()
