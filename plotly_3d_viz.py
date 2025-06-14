#!/usr/bin/env python3
"""
Plotly-based 3D Quantum Simulation                html.Di                html.Div([
                    html.Label("Point Size:", style={'color': 'white'}),
                    dcc.Slider(
                        id='point-size',
                        min=1, max=10, value=3, step=0.5,
                        marks={i: str(i) for i in range(1, 11)},
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'})               html.Label("Threshold:", style={'c                if np.sum(mask) == 0:
                    # If no points above threshold, use an extremely low threshold
                    effective_threshold = max_val * 0.0001  # 0.01% of max
                    mask = np.abs(data_sub) > effective_threshold
                    print(f"Using ultra-low threshold: {effective_threshold:.8f}")
                    print(f"Points with ultra-low threshold: {np.sum(mask)}") 'white'}),
                    dcc.Slider(
                        id='threshold',
                        min=0.0000001, max=0.001, value=0.00001, step=0.0000001,
                        marks={i*0.0002: f"{i*0.0002:.6f}" for i in range(0, 6)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'}),er

Real-time interactive 3D visualization of quantum wavefunctions using Plotly.
Works well in both browsers and Jupyter notebooks, no X11/Wayland issues.
"""

import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import numpy as np
import torch
from threading import Thread
import time
from typing import Optional

from unified_hybrid_molecular_simulation import create_atom_simulation, AtomConfig
from config import SIZE_X, SIZE_Y, SIZE_Z

class PlotlyQuantumViz:
    def __init__(self, simulation, port=8050):
        self.simulation = simulation
        self.port = port
        self.app = Dash(__name__)
        self.frame_count = 0
        
        # Visualization parameters
        self.threshold_multiplier = 0.00001  # Ultra-sensitive default threshold for tiny data values
        self.subsample_factor = 4  # Every 4th point for performance
        self.max_points = 50000   # Maximum points to display
        
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup the Dash app layout."""
        self.app.layout = html.Div([
            html.H1("3D Quantum Simulation - Real Time", 
                   style={'textAlign': 'center', 'color': 'white'}),
            
            dcc.Graph(id='quantum-3d', style={'height': '80vh'}),
            
            dcc.Interval(id='interval', interval=200, n_intervals=0),  # 5 FPS
            
            html.Div([
                html.Div([
                    html.Label("Visualization Mode:", style={'color': 'white'}),
                    dcc.Dropdown(
                        id='viz-mode',
                        options=[
                            {'label': 'Probability Density', 'value': 'probability'},
                            {'label': 'Real Part', 'value': 'real'},
                            {'label': 'Imaginary Part', 'value': 'imaginary'},
                            {'label': 'Phase', 'value': 'phase'}
                        ],
                        value='probability',
                        style={'backgroundColor': '#2f2f2f', 'color': 'black'}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label("Threshold:", style={'color': 'white'}),
                    dcc.Slider(
                        id='threshold',
                        min=0.00001, max=0.01, value=0.0001, step=0.00001,
                        marks={i/100000: f"{i/100000:.5f}" for i in range(0, 6, 1)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label("Point Size:", style={'color': 'white'}),
                    dcc.Slider(
                        id='point-size',
                        min=1, max=10, value=3, step=1,
                        marks={i: str(i) for i in range(1, 11, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'})
            ], style={'backgroundColor': '#1f1f1f', 'padding': '10px'}),
            
            html.Div(id='stats', style={'color': 'white', 'textAlign': 'center', 'margin': '10px'})
        ], style={'backgroundColor': '#1f1f1f'})
    
    def extract_3d_data(self, mode='probability', threshold=0.15, point_size=3):
        """Extract 3D volumetric data from wavefunction."""
        try:
            psi = self.simulation.get_combined_wavefunction()
            if isinstance(psi, torch.Tensor):
                psi = psi.detach().cpu().numpy()
            
            print(f"Wavefunction shape: {psi.shape}")
            print(f"Wavefunction dtype: {psi.dtype}")
            print(f"Wavefunction range: {psi.min():.6f} to {psi.max():.6f}")
            
            if mode == 'probability':
                data = np.abs(psi)**2
                colorscale = 'Hot'
                title_suffix = "Probability Density"
            elif mode == 'real':
                data = np.real(psi)
                colorscale = 'RdBu'
                title_suffix = "Real Part"
            elif mode == 'imaginary':
                data = np.imag(psi)
                colorscale = 'RdBu'
                title_suffix = "Imaginary Part"
            elif mode == 'phase':
                data = np.angle(psi)
                colorscale = 'HSV'
                title_suffix = "Phase"
            
            print(f"Data range after processing: {data.min():.6f} to {data.max():.6f}")
            
            # Subsample for performance
            step = self.subsample_factor
            data_sub = data[::step, ::step, ::step]
            
            print(f"Subsampled data shape: {data_sub.shape}")
            print(f"Subsampled data range: {data_sub.min():.6f} to {data_sub.max():.6f}")
            
            # Apply threshold
            max_val = np.max(np.abs(data_sub))
            print(f"Max absolute value: {max_val:.6f}")
            
            if max_val > 0:
                # Use a much lower threshold to see more data
                effective_threshold = threshold * max_val
                mask = np.abs(data_sub) > effective_threshold
                
                print(f"Threshold: {effective_threshold:.8f}")
                print(f"Points above threshold: {np.sum(mask)}")
                
                if np.sum(mask) == 0:
                    # If no points above threshold, use a much lower threshold
                    effective_threshold = max_val * 0.0001  # 0.01% of max
                    mask = np.abs(data_sub) > effective_threshold
                    print(f"Using lower threshold: {effective_threshold:.8f}")
                    print(f"Points with lower threshold: {np.sum(mask)}")
                
                # Get coordinates of significant points
                x_coords, y_coords, z_coords = np.where(mask)
                values = data_sub[mask]
                
                # Scale coordinates back
                x_coords = x_coords * step
                y_coords = y_coords * step
                z_coords = z_coords * step
                
                print(f"Final number of points: {len(values)}")
                print(f"Coordinate ranges: X[{x_coords.min():.1f}, {x_coords.max():.1f}], Y[{y_coords.min():.1f}, {y_coords.max():.1f}], Z[{z_coords.min():.1f}, {z_coords.max():.1f}]")
                print(f"Value range: [{values.min():.8f}, {values.max():.8f}]")
                
                # Limit number of points for performance
                if len(values) > self.max_points:
                    indices = np.random.choice(len(values), self.max_points, replace=False)
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
    
    def setup_callbacks(self):
        """Setup Dash callbacks for interactivity."""
        @self.app.callback(
            [Output('quantum-3d', 'figure'),
             Output('stats', 'children')],
            [Input('interval', 'n_intervals'),
             Input('viz-mode', 'value'),
             Input('threshold', 'value'),
             Input('point-size', 'value')]
        )
        def update_3d_plot(n, mode, threshold, point_size):
            try:
                # Evolve simulation
                self.simulation.evolve_step(self.frame_count)
                self.frame_count += 1
                
                # Extract data
                x, y, z, values, colorscale, title_suffix, max_val = self.extract_3d_data(
                    mode, threshold, point_size
                )
                
                # Create 3D scatter plot
                fig = go.Figure()
                
                # Always add test points to verify 3D rendering
                fig.add_trace(go.Scatter3d(
                    x=[128], y=[128], z=[128],
                    mode='markers',
                    marker=dict(size=25, color='lime', opacity=1.0, symbol='diamond'),
                    name='Test Center Point',
                    hovertemplate='<b>Test Point</b><br>X: 128<br>Y: 128<br>Z: 128<extra></extra>'
                ))
                
                # Add corner test points to verify 3D space
                fig.add_trace(go.Scatter3d(
                    x=[50, 200, 50, 200], 
                    y=[50, 50, 200, 200], 
                    z=[50, 200, 50, 200],
                    mode='markers',
                    marker=dict(size=15, color='red', opacity=1.0, symbol='square'),
                    name='Corner Test Points',
                    hovertemplate='<b>Corner Test</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
                ))
                
                if len(values) > 0:
                    # Ensure all coordinates and values are numpy arrays and finite
                    x = np.asarray(x)
                    y = np.asarray(y)
                    z = np.asarray(z)
                    values = np.asarray(values)
                    
                    print(f"Data types: x={x.dtype}, y={y.dtype}, z={z.dtype}, values={values.dtype}")
                    print(f"Any NaN in x? {np.isnan(x).any()}  y? {np.isnan(y).any()}  z? {np.isnan(z).any()}  values? {np.isnan(values).any()}")
                    print(f"Any Inf in x? {np.isinf(x).any()}  y? {np.isinf(y).any()}  z? {np.isinf(z).any()}  values? {np.isinf(values).any()}")
                    
                    # Normalize color values for better visibility
                    color_values = values / values.max() if values.max() > 0 else values
                    
                    print(f"Color values range: [{color_values.min():.6f}, {color_values.max():.6f}]")
                    
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='markers',
                        marker=dict(
                            size=10,  # Much larger points
                            color=color_values,
                            colorscale=colorscale,
                            showscale=True,
                            opacity=1.0,  # Full opacity
                            sizemode='diameter',
                            line=dict(width=2, color='white'),  # Thick white outline
                            colorbar=dict(title=title_suffix)
                        ),
                        name='Wavefunction',
                        hovertemplate='<b>Position:</b><br>' +
                                    'X: %{x:.2f}<br>' +
                                    'Y: %{y:.2f}<br>' +
                                    'Z: %{z:.2f}<br>' +
                                    f'<b>{title_suffix}:</b> %{{marker.color:.6f}}<extra></extra>'
                    ))
                    print(f"Added scatter trace with {len(values)} points")
                    print(f"Point size: {max(point_size, 3)}, Opacity: 0.9")
                    print(f"Sample X values: {x[:5]}")
                    print(f"Sample Y values: {y[:5]}")
                    print(f"Sample Z values: {z[:5]}")
                    print(f"Sample color values: {color_values[:5]}")
                else:
                    print("No points to add to scatter plot")
                
                # Add nuclei as larger spheres
                for i, nucleus in enumerate(self.simulation.nuclei):
                    nucleus_x = nucleus.position[0].item() if hasattr(nucleus.position[0], 'item') else nucleus.position[0]
                    nucleus_y = nucleus.position[1].item() if hasattr(nucleus.position[1], 'item') else nucleus.position[1]
                    nucleus_z = nucleus.position[2].item() if len(nucleus.position) > 2 and hasattr(nucleus.position[2], 'item') else (nucleus.position[2] if len(nucleus.position) > 2 else 0)
                    
                    print(f"Nucleus {i+1} position: ({nucleus_x:.1f}, {nucleus_y:.1f}, {nucleus_z:.1f})")
                    
                    fig.add_trace(go.Scatter3d(
                        x=[nucleus_x], y=[nucleus_y], z=[nucleus_z],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='red',
                            symbol='circle',
                            line=dict(width=2, color='darkred')
                        ),
                        name=f'Nucleus {i+1} (Z={nucleus.atomic_number})',
                        hovertemplate=f'<b>Nucleus {i+1}</b><br>' +
                                    f'Atomic Number: {nucleus.atomic_number}<br>' +
                                    'Position: (%{x}, %{y}, %{z})<extra></extra>'
                    ))
                
                # Calculate axis ranges based on data (grid is 256x256x256)
                # Initialize defaults first
                x_range = [0, 255]
                y_range = [0, 255]
                z_range = [0, 255]
                x_center, y_center, z_center = 128, 128, 128
                
                if len(values) > 0:
                    x_range = [max(0, x.min() - 20), min(255, x.max() + 20)]
                    y_range = [max(0, y.min() - 20), min(255, y.max() + 20)] 
                    z_range = [max(0, z.min() - 20), min(255, z.max() + 20)]
                    
                    # Center for camera
                    x_center = (x_range[0] + x_range[1]) / 2
                    y_center = (y_range[0] + y_range[1]) / 2
                    z_center = (z_range[0] + z_range[1]) / 2
                
                print(f"Axis ranges: X{x_range}, Y{y_range}, Z{z_range}")
                print(f"Camera center: ({x_center:.1f}, {y_center:.1f}, {z_center:.1f})")
                
                # Update layout with explicit 3D scene configuration
                fig.update_layout(
                    title=f'3D Quantum Simulation - {title_suffix} (Frame {self.frame_count})',
                    scene=dict(
                        xaxis=dict(title='X Position', range=[0, 255], showgrid=True, gridcolor='gray'),
                        yaxis=dict(title='Y Position', range=[0, 255], showgrid=True, gridcolor='gray'),
                        zaxis=dict(title='Z Position', range=[0, 255], showgrid=True, gridcolor='gray'),
                        aspectmode='cube',  # Keep cube aspect ratio for proper 3D
                        camera=dict(
                            eye=dict(x=1.8, y=1.8, z=1.8),  # Position camera properly for 256^3 space
                            center=dict(x=128, y=128, z=128),  # Look at center of grid
                            up=dict(x=0, y=0, z=1)  # Z is up
                        ),
                        bgcolor='rgb(5, 5, 15)',
                        # Add axis labels and ticks
                        xaxis_tickmode='linear',
                        yaxis_tickmode='linear', 
                        zaxis_tickmode='linear'
                    ),
                    template='plotly_dark',
                    showlegend=True,
                    autosize=True,
                    height=800,  # Set explicit height
                    margin=dict(l=0, r=0, t=50, b=0),  # Minimize margins
                    # Force WebGL if available
                    config=dict(
                        displayModeBar=True,
                        plotlyServerURL="https://plot.ly"
                    )
                )
                
                # Stats
                total_prob = self.simulation.get_combined_wavefunction()
                if isinstance(total_prob, torch.Tensor):
                    total_prob = torch.sum(torch.abs(total_prob)**2).item()
                else:
                    total_prob = np.sum(np.abs(total_prob)**2)
                
                stats_text = f"Frame: {self.frame_count} | Points: {len(values)} | " + \
                           f"Max {title_suffix}: {max_val:.6f} | Total Probability: {total_prob:.6f}"
                
                return fig, stats_text
                
            except Exception as e:
                print(f"Error in callback: {e}")
                # Return empty figure on error
                fig = go.Figure()
                fig.update_layout(
                    title='Error in visualization',
                    template='plotly_dark'
                )
                return fig, f"Error: {str(e)}"
    
    def run(self, debug=True):
        """Start the visualization server."""
        print(f"Starting 3D quantum visualization server...")
        print(f"Open your browser to: http://localhost:{self.port}")
        print(f"Controls:")
        print(f"  - Mouse: rotate, zoom, pan")
        print(f"  - Dropdowns/sliders: change visualization parameters")
        print(f"  - Ctrl+C to stop")
        
        self.app.run(debug=debug, port=self.port, host='0.0.0.0')


def main():
    """Main function to run the 3D quantum visualization."""
    print("Creating 3D quantum simulation...")
    
    # Create a hydrogen atom at the center
    hydrogen_config = AtomConfig(
        atomic_number=1, 
        position=(SIZE_X//2, SIZE_Y//2, SIZE_Z//2)
    )
    
    simulation = create_atom_simulation(hydrogen_config)
    
    print("Starting web-based 3D visualization...")
    viz = PlotlyQuantumViz(simulation)
    viz.run()


if __name__ == "__main__":
    main()
