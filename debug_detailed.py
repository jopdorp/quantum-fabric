#!/usr/bin/env python3
"""
Debug version of quantum visualization to understand exactly what's happening.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_hybrid_molecular_simulation import create_atom_simulation, AtomConfig
from config import SIZE_X, SIZE_Y, SIZE_Z

def debug_quantum_viz():
    """Debug the quantum visualization step by step."""
    print("=== QUANTUM VISUALIZATION DEBUG ===")
    
    # Create simulation
    print("1. Creating simulation...")
    hydrogen_config = AtomConfig(
        atomic_number=1, 
        position=(SIZE_X//2, SIZE_Y//2, SIZE_Z//2)
    )
    simulation = create_atom_simulation(hydrogen_config)
    print(f"   Grid size: {SIZE_X}x{SIZE_Y}x{SIZE_Z}")
    print(f"   Nucleus at: ({SIZE_X//2}, {SIZE_Y//2}, {SIZE_Z//2})")
    
    # Evolve one step
    print("2. Evolving simulation...")
    simulation.evolve_step(1)
    
    # Get wavefunction
    print("3. Getting wavefunction...")
    wavefunction = simulation.get_combined_wavefunction()
    if hasattr(wavefunction, 'detach'):
        wavefunction_np = wavefunction.detach().cpu().numpy()
    else:
        wavefunction_np = np.array(wavefunction)
    
    print(f"   Wavefunction shape: {wavefunction_np.shape}")
    print(f"   Wavefunction range: {wavefunction_np.min():.8f} to {wavefunction_np.max():.8f}")
    
    # Calculate probability density
    print("4. Calculating probability density...")
    prob_density = np.abs(wavefunction_np)**2
    print(f"   Probability range: {prob_density.min():.8f} to {prob_density.max():.8f}")
    print(f"   Probability sum: {prob_density.sum():.8f}")
    
    # Subsample
    print("5. Subsampling...")
    step = 4
    prob_sub = prob_density[::step, ::step, ::step]
    print(f"   Subsampled shape: {prob_sub.shape}")
    print(f"   Subsampled range: {prob_sub.min():.8f} to {prob_sub.max():.8f}")
    
    # Find significant points
    print("6. Finding significant points...")
    max_val = prob_sub.max()
    print(f"   Max value: {max_val:.8f}")
    
    # Try different thresholds
    thresholds = [0.00001, max_val * 0.1, max_val * 0.01, max_val * 0.001, max_val * 0.0001]
    for thresh in thresholds:
        mask = prob_sub > thresh
        count = np.sum(mask)
        print(f"   Threshold {thresh:.8f}: {count} points")
        if count > 0 and count < 1000:
            break
    
    # Use the working threshold
    final_threshold = thresh
    mask = prob_sub > final_threshold
    print(f"   Using threshold: {final_threshold:.8f}")
    print(f"   Final point count: {np.sum(mask)}")
    
    # Get coordinates
    x_coords, y_coords, z_coords = np.where(mask)
    values = prob_sub[mask]
    
    # Scale back
    x_coords = x_coords * step
    y_coords = y_coords * step  
    z_coords = z_coords * step
    
    print(f"7. Point locations:")
    print(f"   X range: [{x_coords.min()}, {x_coords.max()}]")
    print(f"   Y range: [{y_coords.min()}, {y_coords.max()}]")
    print(f"   Z range: [{z_coords.min()}, {z_coords.max()}]")
    print(f"   Value range: [{values.min():.8f}, {values.max():.8f}]")
    
    # Calculate optimal camera position
    print("8. Camera positioning:")
    center_x = (x_coords.min() + x_coords.max()) / 2
    center_y = (y_coords.min() + y_coords.max()) / 2
    center_z = (z_coords.min() + z_coords.max()) / 2
    print(f"   Data center: ({center_x:.1f}, {center_y:.1f}, {center_z:.1f})")
    
    # Calculate extent
    extent_x = x_coords.max() - x_coords.min()
    extent_y = y_coords.max() - y_coords.min()
    extent_z = z_coords.max() - z_coords.min()
    max_extent = max(extent_x, extent_y, extent_z)
    print(f"   Data extent: {extent_x:.1f} x {extent_y:.1f} x {extent_z:.1f}")
    print(f"   Max extent: {max_extent:.1f}")
    
    # Create figure with focused view
    print("9. Creating visualization...")
    fig = go.Figure()
    
    # Add the quantum data
    if len(values) > 0:
        # Normalize colors
        color_values = values / values.max()
        
        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='markers',
            marker=dict(
                size=25,  # Very large points
                color=color_values,
                colorscale='Hot',
                showscale=True,
                opacity=1.0,
                line=dict(width=3, color='white'),  # White outline for contrast
                colorbar=dict(title='Probability Density')
            ),
            name=f'Quantum Data ({len(values)} points)'
        ))
        print(f"   Added {len(values)} quantum data points")
    
    # Add nucleus
    nucleus_pos = simulation.nuclei[0].position
    nucleus_x = nucleus_pos[0].item() if hasattr(nucleus_pos[0], 'item') else nucleus_pos[0]
    nucleus_y = nucleus_pos[1].item() if hasattr(nucleus_pos[1], 'item') else nucleus_pos[1]
    nucleus_z = nucleus_pos[2].item() if len(nucleus_pos) > 2 and hasattr(nucleus_pos[2], 'item') else (nucleus_pos[2] if len(nucleus_pos) > 2 else 0)
    
    fig.add_trace(go.Scatter3d(
        x=[nucleus_x], y=[nucleus_y], z=[nucleus_z],
        mode='markers',
        marker=dict(size=50, color='red', symbol='circle', line=dict(width=5, color='darkred')),
        name='Nucleus'
    ))
    print(f"   Added nucleus at ({nucleus_x:.1f}, {nucleus_y:.1f}, {nucleus_z:.1f})")
    
    # Add reference points at corners of data region
    corners_x = [x_coords.min(), x_coords.max(), x_coords.min(), x_coords.max()]
    corners_y = [y_coords.min(), y_coords.min(), y_coords.max(), y_coords.max()]
    corners_z = [z_coords.min(), z_coords.max(), z_coords.min(), z_coords.max()]
    
    fig.add_trace(go.Scatter3d(
        x=corners_x, y=corners_y, z=corners_z,
        mode='markers',
        marker=dict(size=20, color='lime', symbol='diamond'),
        name='Data Region Corners'
    ))
    print(f"   Added corner markers")
    
    # Set focused view on the data with generous padding
    data_padding = max(max_extent * 0.5, 20)  # 50% padding or at least 20 units
    x_range = [x_coords.min() - data_padding, x_coords.max() + data_padding]
    y_range = [y_coords.min() - data_padding, y_coords.max() + data_padding]
    z_range = [z_coords.min() - data_padding, z_coords.max() + data_padding]
    
    fig.update_layout(
        title=f'DEBUG: Quantum Visualization (Threshold: {final_threshold:.8f})',
        scene=dict(
            xaxis=dict(title='X', range=x_range, showgrid=True, gridcolor='white'),
            yaxis=dict(title='Y', range=y_range, showgrid=True, gridcolor='white'), 
            zaxis=dict(title='Z', range=z_range, showgrid=True, gridcolor='white'),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=2.0, y=2.0, z=2.0),  # Further out for better view
                center=dict(x=center_x, y=center_y, z=center_z),  # Look at data center
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='rgb(0, 0, 50)'  # Dark blue for contrast
        ),
        template='plotly_white',  # Try white template for max contrast
        showlegend=True,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    print(f"10. Camera setup:")
    print(f"    Axis ranges: X{x_range}, Y{y_range}, Z{z_range}")
    print(f"    Camera center: ({center_x:.1f}, {center_y:.1f}, {center_z:.1f})")
    print(f"    Camera eye: (2.0, 2.0, 2.0) relative to center")
    
    # Save
    filename = 'debug_quantum_detailed.html'
    pyo.plot(fig, filename=filename, auto_open=True)
    print(f"11. Saved {filename}")
    
    print("\n=== SUMMARY ===")
    print(f"Threshold used: {final_threshold:.8f}")
    print(f"Data points: {len(values)} quantum + 1 nucleus + 4 corners = {len(values) + 5} total")
    print(f"Data concentrated in: {max_extent:.1f} unit region")
    print(f"Camera: focused on data center ({center_x:.1f}, {center_y:.1f}, {center_z:.1f})")
    print(f"Expected to see: Large colored spheres clustered together")

if __name__ == "__main__":
    debug_quantum_viz()
