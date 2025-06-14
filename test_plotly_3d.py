#!/usr/bin/env python3
"""
Simple test to verify Plotly 3D rendering works in your environment.
Tests both offline Plotly and Dash versions.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import dash
from dash import dcc, html, Input, Output

def create_test_figure():
    """Create a simple 3D scatter plot to test rendering."""
    # Create simple test data
    n_points = 100
    x = np.random.uniform(0, 10, n_points)
    y = np.random.uniform(0, 10, n_points) 
    z = np.random.uniform(0, 10, n_points)
    colors = np.random.uniform(0, 1, n_points)
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            colorscale='Viridis',
            opacity=1.0,
            showscale=True
        ),
        name='Test Points'
    ))
    
    # Add a big red point at origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=20, color='red'),
        name='Origin'
    ))
    
    # Add a big blue point at corner
    fig.add_trace(go.Scatter3d(
        x=[10], y=[10], z=[10],
        mode='markers',
        marker=dict(size=20, color='blue'),
        name='Corner'
    ))
    
    # Update layout
    fig.update_layout(
        title='Simple 3D Test - If you see points, 3D works!',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            aspectmode='cube'
        ),
        template='plotly_dark'
    )
    
    return fig

def test_plotly_offline():
    """Test Plotly offline HTML generation."""
    print("Testing Plotly offline...")
    fig = create_test_figure()
    
    # Save as HTML and show
    pyo.plot(fig, filename='test_plotly_offline.html', auto_open=True)
    print("Saved test_plotly_offline.html")
    print("You should see:")
    print("  - Random colored points scattered in 3D")
    print("  - Big red point at origin (0,0,0)")  
    print("  - Big blue point at corner (10,10,10)")
    print("  - Ability to rotate, zoom, pan")

def test_dash_3d():
    """Test Dash 3D rendering."""
    print("\nTesting Dash 3D...")
    
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Dash 3D Test", style={'textAlign': 'center', 'color': 'white'}),
        html.P("If you see 3D points below, Dash 3D works!", style={'textAlign': 'center', 'color': 'white'}),
        dcc.Graph(id='dash-3d-test', figure=create_test_figure(), style={'height': '80vh'}),
        dcc.Interval(id='interval', interval=3000, n_intervals=0),  # Update every 3 seconds
    ])
    
    @app.callback(
        Output('dash-3d-test', 'figure'),
        Input('interval', 'n_intervals')
    )
    def update_figure(n):
        print(f"Dash update {n}")
        return create_test_figure()
    
    print("Starting Dash server on http://localhost:8052")
    print("You should see the same 3D visualization as the HTML file")
    print("Press Ctrl+C to stop when done testing")
    
    try:
        app.run(debug=True, port=8052, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\nDash test stopped.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'dash':
        test_dash_3d()
    else:
        print("Usage:")
        print("  python test_plotly_3d.py        # Test Plotly offline HTML")
        print("  python test_plotly_3d.py dash   # Test Dash 3D server")
        print()
        test_plotly_offline()
