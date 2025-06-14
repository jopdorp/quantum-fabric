#!/usr/bin/env python3
"""
Debug version of quantum visualization to isolate the issue.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo

def create_minimal_test():
    """Create a minimal version with fake quantum data to test visibility."""
    
    # Create fake quantum data similar to what the real simulation produces
    n_points = 123  # Same number as real simulation
    
    # Create points clustered around center like a hydrogen orbital
    center = [128, 128, 128]
    spread = 12  # Points spread around center (116-140 range)
    
    x = np.random.normal(center[0], spread, n_points)
    y = np.random.normal(center[1], spread, n_points)
    z = np.random.normal(center[2], spread, n_points)
    
    # Values similar to real quantum data
    values = np.random.uniform(0.00001, 0.00127, n_points)
    color_values = values / values.max()
    
    print(f"Test data created:")
    print(f"  Points: {len(values)}")
    print(f"  X range: [{x.min():.1f}, {x.max():.1f}]")
    print(f"  Y range: [{y.min():.1f}, {y.max():.1f}]")
    print(f"  Z range: [{z.min():.1f}, {z.max():.1f}]")
    print(f"  Value range: [{values.min():.6f}, {values.max():.6f}]")
    
    # Create figure
    fig = go.Figure()
    
    # Add the test center point (should be very visible)
    fig.add_trace(go.Scatter3d(
        x=[128], y=[128], z=[128],
        mode='markers',
        marker=dict(size=50, color='lime', opacity=1.0, symbol='diamond'),
        name='CENTER TEST POINT - SHOULD BE VISIBLE'
    ))
    
    # Add corner test points
    corners = [[50, 50, 50], [200, 200, 200], [50, 200, 50], [200, 50, 200]]
    for i, (cx, cy, cz) in enumerate(corners):
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode='markers',
            marker=dict(size=30, color=['red', 'blue', 'green', 'yellow'][i], opacity=1.0),
            name=f'Corner {i+1} ({cx},{cy},{cz})'
        ))
    
    # Add quantum data with MAXIMUM visibility
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=20,  # Very large points
            color=color_values,
            colorscale='Hot',
            showscale=True,
            opacity=1.0,  # Full opacity
            line=dict(width=3, color='black'),  # Thick black outline
            colorbar=dict(title='Probability Density')
        ),
        name='Quantum Data (Fake but Similar)'
    ))
    
    # Add nucleus
    fig.add_trace(go.Scatter3d(
        x=[128], y=[128], z=[128],
        mode='markers',
        marker=dict(size=40, color='red', symbol='circle', 
                   line=dict(width=4, color='darkred')),
        name='Nucleus (Hydrogen)'
    ))
    
    # Set layout with maximum visibility settings
    fig.update_layout(
        title='DEBUG: Quantum Visualization Test - Should Show Multiple Colored Points',
        scene=dict(
            xaxis=dict(title='X Position', range=[0, 255], showgrid=True, gridcolor='white'),
            yaxis=dict(title='Y Position', range=[0, 255], showgrid=True, gridcolor='white'),
            zaxis=dict(title='Z Position', range=[0, 255], showgrid=True, gridcolor='white'),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=128, y=128, z=128),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='rgb(0, 0, 0)'  # Pure black background
        ),
        template='plotly_white',  # Try white template for contrast
        showlegend=True,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Save and open
    filename = 'debug_quantum_test.html'
    pyo.plot(fig, filename=filename, auto_open=True)
    print(f"\nSaved {filename}")
    print("This should show:")
    print("  - LARGE lime diamond at center (128,128,128)")
    print("  - 4 colored corner points")
    print("  - ~123 hot-colored points clustered around center")
    print("  - Red nucleus at center")
    print("  - White grid lines")
    print("\nIf you see NOTHING, the issue is with:")
    print("  1. Browser/WebGL support")
    print("  2. Plotly library issue")
    print("  3. File corruption during save")

if __name__ == "__main__":
    create_minimal_test()
