#!/usr/bin/env python3
"""
Minimal 3D quantum visualization test to isolate the rendering issue.
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Minimal 3D Test", style={'textAlign': 'center'}),
    dcc.Graph(id='test-3d', style={'height': '80vh'}),
    dcc.Interval(id='interval', interval=2000, n_intervals=0),  # 2 second updates
])

@app.callback(
    Output('test-3d', 'figure'),
    Input('interval', 'n_intervals')
)
def update_plot(n):
    print(f"Update {n}")
    
    # Create fake quantum data similar to what our real simulation produces
    np.random.seed(42)  # Reproducible
    
    # Generate points around center (128, 128, 128) like our quantum simulation
    center = 128
    n_points = 1000
    
    # Create a 3D Gaussian distribution around the center
    x = np.random.normal(center, 20, n_points).astype(int)
    y = np.random.normal(center, 20, n_points).astype(int)
    z = np.random.normal(center, 20, n_points).astype(int)
    
    # Clip to valid range
    x = np.clip(x, 80, 176)
    y = np.clip(y, 80, 176)
    z = np.clip(z, 80, 176)
    
    # Create values similar to our quantum data
    values = np.random.exponential(0.001, n_points)
    
    print(f"Generated {len(values)} points")
    print(f"Coordinate ranges: X[{x.min()}, {x.max()}], Y[{y.min()}, {y.max()}], Z[{z.min()}, {z.max()}]")
    print(f"Value range: [{values.min():.6f}, {values.max():.6f}]")
    
    # Create figure exactly like our main visualization
    fig = go.Figure()
    
    # Add the exact same test point as our main viz
    fig.add_trace(go.Scatter3d(
        x=[128], y=[128], z=[128],
        mode='markers',
        marker=dict(size=15, color='lime', opacity=1.0, symbol='diamond'),
        name='Test Center Point',
        hovertemplate='<b>Test Point</b><br>X: 128<br>Y: 128<br>Z: 128<extra></extra>'
    ))
    
    # Add quantum-like data with exact same configuration as main viz
    color_values = values / values.max() if values.max() > 0 else values
    
    fig.add_trace(go.Scatter3d(
        x=x.tolist(),
        y=y.tolist(), 
        z=z.tolist(),
        mode='markers',
        marker=dict(
            size=3,  # Same as main viz
            color=color_values.tolist(),
            colorscale='Hot',  # Same as main viz
            showscale=True,
            opacity=0.9,  # Same as main viz
            sizemode='diameter',
            line=dict(width=0.5, color='white'),
            colorbar=dict(title='Probability Density')
        ),
        name='Fake Quantum Data',
        hovertemplate='<b>Position:</b><br>' +
                    'X: %{x:.2f}<br>' +
                    'Y: %{y:.2f}<br>' +
                    'Z: %{z:.2f}<br>' +
                    '<b>Value:</b> %{marker.color:.6f}<extra></extra>'
    ))
    
    # Add nucleus at center (red sphere like main viz)
    fig.add_trace(go.Scatter3d(
        x=[128], y=[128], z=[128],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='circle',
            line=dict(width=2, color='darkred')
        ),
        name='Nucleus',
        hovertemplate='<b>Nucleus</b><br>Position: (128, 128, 128)<extra></extra>'
    ))
    
    # Exact same layout as main visualization
    x_range = [80, 176]
    y_range = [80, 176] 
    z_range = [80, 176]
    x_center, y_center, z_center = 128, 128, 128
    
    fig.update_layout(
        title='Minimal 3D Test - Fake Quantum Data',
        scene=dict(
            xaxis=dict(title='X Position', range=x_range, showgrid=True, gridcolor='gray'),
            yaxis=dict(title='Y Position', range=y_range, showgrid=True, gridcolor='gray'),
            zaxis=dict(title='Z Position', range=z_range, showgrid=True, gridcolor='gray'),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.8),
                center=dict(x=x_center, y=y_center, z=z_center),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='rgb(5, 5, 15)',
            xaxis_tickmode='linear',
            yaxis_tickmode='linear', 
            zaxis_tickmode='linear'
        ),
        template='plotly_dark',
        showlegend=True,
        autosize=True,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    print(f"Figure created with {len(fig.data)} traces")
    for i, trace in enumerate(fig.data):
        print(f"  Trace {i}: {trace.name} with {len(trace.x) if hasattr(trace, 'x') else 0} points")
    
    return fig

if __name__ == "__main__":
    print("Starting minimal 3D test on http://localhost:8051")
    print("This uses fake data but exact same Plotly configuration as main visualization")
    app.run(debug=True, port=8051, host='0.0.0.0')
