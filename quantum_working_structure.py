#!/usr/bin/env python3
"""
Fixed quantum visualization using the same structure as the working test_plotly_3d.py
"""

import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import dash
from dash import dcc, html, Input, Output
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_hybrid_molecular_simulation import create_atom_simulation, AtomConfig
from config import SIZE_X, SIZE_Y, SIZE_Z

def create_quantum_figure(evolution_steps=50):
    """Create quantum visualization using the EXACT same structure as working test."""
    print("Creating quantum simulation...")
    
    # Create a hydrogen atom at the center with 2p orbital (n=2, l=1, m=0)
    hydrogen_config = AtomConfig(
        atomic_number=1, 
        position=(SIZE_X//2, SIZE_Y//2, SIZE_Z//2),
        electron_configs=[(2, 1, 0)]  # n=2, l=1, m=0 (2p orbital)
    )
    
    simulation = create_atom_simulation(hydrogen_config)
    
    # Do several evolution steps to see dynamics
    evolution_steps = 50
    print(f"Evolving simulation for {evolution_steps} steps...")
    for step in range(evolution_steps):
        simulation.evolve_step(step)
        if (step + 1) % 10 == 0:  # Print progress every 10 steps
            print(f"  Completed evolution step {step + 1}/{evolution_steps}")
    
    # Get wavefunction data
    wavefunction = simulation.get_combined_wavefunction()
    if hasattr(wavefunction, 'detach'):
        wavefunction_np = wavefunction.detach().cpu().numpy()
    else:
        wavefunction_np = np.array(wavefunction)
    
    # Calculate probability density
    data = np.abs(wavefunction_np)**2
    print(f"Wavefunction shape: {data.shape}")
    print(f"Data range: {data.min():.8f} to {data.max():.8f}")
    
    # Check total probability (should be ~1 for normalized wavefunction)
    total_probability = np.sum(data)
    print(f"Total probability (all space): {total_probability:.6f}")
    print(f"Grid spacing factor: Each voxel represents {1.0} unit³")
    
    # For proper normalization, we might need to account for grid spacing
    # If the simulation uses proper units, total should be close to 1
    
    # Subsample and get points above threshold - SAME AS WORKING TEST APPROACH
    step = 4
    data_sub = data[::step, ::step, ::step]
    threshold = data_sub.max() * 0.01  # 1% of max value
    
    mask = data_sub > threshold
    x_coords, y_coords, z_coords = np.where(mask)
    values = data_sub[mask]
    
    # Check what fraction of total probability we're displaying
    displayed_probability = np.sum(values)
    total_subsampled_probability = np.sum(data_sub)
    display_fraction = displayed_probability / total_subsampled_probability if total_subsampled_probability > 0 else 0
    
    print(f"Subsampled total probability: {total_subsampled_probability:.6f}")
    print(f"Displayed probability (above threshold): {displayed_probability:.6f}")
    print(f"Displaying {display_fraction*100:.1f}% of subsampled probability")
    
    # KEEP ORIGINAL COORDINATE SPACE (0-255) but use working structure
    x_coords = x_coords * step
    y_coords = y_coords * step  
    z_coords = z_coords * step
    
    print(f"Found {len(values)} points")
    print(f"Coordinate ranges: X[{x_coords.min():.1f}, {x_coords.max():.1f}], Y[{y_coords.min():.1f}, {y_coords.max():.1f}], Z[{z_coords.min():.1f}, {z_coords.max():.1f}]")
    
    # Use log scale for colors but map to 0.4-1.0 range for more yellow!
    epsilon = 1e-10
    log_values = np.log10(values + epsilon) * 10
    
    # Get actual min/max of our data instead of fixed range
    actual_log_min = log_values.min()
    actual_log_max = log_values.max()
    
    # Map to 0.6-1.0 range instead of 0.4-1.0 to get even more yellow/bright colors
    color_values = 0.6 + 0.4 * ((log_values - actual_log_min) / (actual_log_max - actual_log_min))
    color_values = np.clip(color_values, 0.6, 1.0)  # Ensure 0.6-1.0 range for very bright colors
    
    # Make point sizes with more aggressive scaling (small points get MUCH smaller)
    # Use exponential scaling: smaller values drop off much more rapidly
    normalized_log = (log_values - actual_log_min) / (actual_log_max - actual_log_min)
    # Apply exponential curve: small values become tiny, large values stay big
    exp_scaled = normalized_log ** 3  # Cube makes small values much smaller
    size_values = 0.5 + 25 * exp_scaled  # Range from 0.5 to 25.5 pixels
    size_values = np.clip(size_values, 0.5, 25)
    
    print(f"Original values range: [{values.min():.8f}, {values.max():.8f}]")
    print(f"Log values range: [{log_values.min():.2f}, {log_values.max():.2f}]")
    print(f"Color values range: [{color_values.min():.3f}, {color_values.max():.3f}]")
    print(f"Size values range: [{size_values.min():.1f}, {size_values.max():.1f}]")
    
    # Create figure EXACTLY like working test
    fig = go.Figure()
    
    # Add quantum points with log-scaled colors
    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(
            size=size_values,  # Variable size based on probability
            color=color_values,
            colorscale='Hot',  # Hot colorscale for more yellow
            opacity=0.8,  # Fixed opacity for all points
            line=dict(width=0),  # No outline
            showscale=True,
            colorbar=dict(
                title="Log Scale Probability",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0.0001", "0.001", "0.01", "0.1", "1.0"]
            )
        ),
        name='Quantum Points (Log Scale)'
    ))
    
    # Add nucleus at original center (128,128,128)
    nucleus_x = 128.0  # Center of 0-255 space
    nucleus_y = 128.0
    nucleus_z = 128.0
    
    fig.add_trace(go.Scatter3d(
        x=[nucleus_x], y=[nucleus_y], z=[nucleus_z],
        mode='markers',
        marker=dict(size=20, color='red'),  # SAME as working test
        name='Nucleus'
    ))
    
    # Use working test layout but with explicit ranges for full 0-255 space
    fig.update_layout(
        title='Quantum 3D - Full 0-255 Space',
        scene=dict(
            xaxis=dict(title='X', range=[0, 255]),
            yaxis=dict(title='Y', range=[0, 255]),
            zaxis=dict(title='Z', range=[0, 255]),
            aspectmode='cube'
        ),
        template='plotly_dark'
    )
    
    return fig

def test_dash_quantum():
    """Test Dash quantum 3D rendering with continuous evolution."""
    print("\nTesting Dash Quantum 3D...")
    
    # Create simulation ONCE outside callback - store in global scope
    global quantum_simulation, quantum_frame_count
    print("Creating persistent quantum simulation...")
    hydrogen_config = AtomConfig(
        atomic_number=1, 
        position=(SIZE_X//2, SIZE_Y//2, SIZE_Z//2),
        electron_configs=[(2, 1, 0)]  # n=2, l=1, m=0 (2p orbital)
    )
    quantum_simulation = create_atom_simulation(hydrogen_config)
    quantum_frame_count = 0
    
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Real-Time Quantum Evolution", style={'textAlign': 'center', 'color': 'white'}),
        html.P("Watch the quantum wavefunction evolve step by step!", style={'textAlign': 'center', 'color': 'white'}),
        
        html.Div([
            html.Label("Threshold (% of max):", style={'color': 'white', 'margin-right': '10px'}),
            dcc.Slider(
                id='threshold-slider',
                min=0.1, max=5.0, value=1.0, step=0.1,
                marks={i: f"{i}%" for i in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]},
                tooltip={'placement': 'bottom', 'always_visible': True}
            )
        ], style={'margin': '20px', 'color': 'white'}),
        
        dcc.Graph(id='dash-quantum-3d', style={'height': '80vh'}),
        dcc.Interval(id='interval', interval=1000, n_intervals=0),  # Update every 1 second
        html.Div(id='frame-info', style={'textAlign': 'center', 'color': 'white', 'margin': '10px'})
    ], style={'backgroundColor': '#1f1f1f'})
    
    @app.callback(
        [Output('dash-quantum-3d', 'figure'),
         Output('frame-info', 'children')],
        [Input('interval', 'n_intervals'),
         Input('threshold-slider', 'value')]
    )
    def update_quantum_figure(n, threshold_percent):
        global quantum_simulation, quantum_frame_count
        
        # STEP the existing simulation
        quantum_simulation.evolve_step(quantum_frame_count)
        quantum_frame_count += 1
        
        print(f"Dash update {n} - Frame: {quantum_frame_count}, Threshold: {threshold_percent}%")
        
        # Get wavefunction data from stepped simulation
        wavefunction = quantum_simulation.get_combined_wavefunction()
        if hasattr(wavefunction, 'detach'):
            wavefunction_np = wavefunction.detach().cpu().numpy()
        else:
            wavefunction_np = np.array(wavefunction)
        
        # Calculate probability density
        data = np.abs(wavefunction_np)**2
        
        # Subsample and apply threshold
        step = 4
        data_sub = data[::step, ::step, ::step]
        threshold = data_sub.max() * (threshold_percent / 100)  # Use slider value
        
        mask = data_sub > threshold
        x_coords, y_coords, z_coords = np.where(mask)
        values = data_sub[mask]
        
        # Scale coordinates
        x_coords = x_coords * step
        y_coords = y_coords * step  
        z_coords = z_coords * step
        
        if len(values) > 0:
            # Use the same color and size scaling as the static version
            epsilon = 1e-10
            log_values = np.log10(values + epsilon) * 10
            
            actual_log_min = log_values.min()
            actual_log_max = log_values.max()
            
            # Your color mapping
            color_values = 0.6 + 0.4 * ((log_values - actual_log_min) / (actual_log_max - actual_log_min))
            color_values = np.clip(color_values, 0.6, 1.0)
            
            # Apply same aggressive size scaling as static version
            normalized_log = (log_values - actual_log_min) / (actual_log_max - actual_log_min)
            exp_scaled = normalized_log ** 3  # Cube makes small values much smaller
            size_values = 0.5 + 25 * exp_scaled
            size_values = np.clip(size_values, 0.5, 25)
        else:
            color_values = []
            size_values = []
        
        # Create figure
        fig = go.Figure()
        
        # Add quantum points
        if len(values) > 0:
            fig.add_trace(go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='markers',
                marker=dict(
                    size=size_values,
                    color=color_values,
                    colorscale='Hot',
                    opacity=0.8,
                    line=dict(width=0),  # No outline
                    showscale=True,
                    colorbar=dict(
                        title="Log Scale Probability",
                        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                        ticktext=["0.0001", "0.001", "0.01", "0.1", "1.0"]
                    )
                ),
                name=f'Quantum Points ({len(values)} points)'
            ))
        
        # Add nucleus
        fig.add_trace(go.Scatter3d(
            x=[128], y=[128], z=[128],
            mode='markers',
            marker=dict(size=20, color='red'),
            name='Nucleus'
        ))
        
        # Layout
        fig.update_layout(
            title=f'Real-Time Quantum Evolution - Frame {quantum_frame_count}, Threshold {threshold_percent}%',
            scene=dict(
                xaxis=dict(title='X', range=[0, 255]),
                yaxis=dict(title='Y', range=[0, 255]),
                zaxis=dict(title='Z', range=[0, 255]),
                aspectmode='cube'
            ),
            template='plotly_dark',
            showlegend=True
        )
        
        frame_info = f"Frame: {quantum_frame_count} | Points: {len(values)} | Threshold: {threshold_percent}%"
        return fig, frame_info
    
    print("Starting Real-Time Quantum Evolution server on http://localhost:8053")
    print("You should see:")
    print("  - Quantum simulation that evolves step-by-step")
    print("  - Interactive threshold slider")
    print("  - Frame counter showing evolution progress")
    print("Press Ctrl+C to stop when done testing")
    
    try:
        app.run(debug=True, port=8053, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\nReal-time quantum visualization stopped.")

def main():
    """Main function with both static and Dash options."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'dash':
        test_dash_quantum()
    else:
        print("Usage:")
        print("  python quantum_working_structure.py        # Generate static HTML")
        print("  python quantum_working_structure.py dash   # Run interactive Dash server")
        print()
        
        print("Testing quantum visualization with working test structure...")
        fig = create_quantum_figure()
        
        # Save using SAME method as working test
        pyo.plot(fig, filename='quantum_working_structure.html', auto_open=True)
        print("Saved quantum_working_structure.html")
        print("You should see:")
        print("  - Quantum probability points around (128,128,128)")
        print("  - Red nucleus at center (128,128,128)")  
        print("  - Blue point at corner (0,0,0)")
        print("  - Green point at corner (255,255,255)")
        print("  - Same simple structure as working test_plotly_3d.py but in 0-255 space")

if __name__ == "__main__":
    main()
