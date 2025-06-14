#!/usr/bin/env python3
"""
Alternative quantum visualization using matplotlib for 3D rendering.
This bypasses PyQt OpenGL issues entirely.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Import our simulation components
sys.path.append('.')

def create_matplotlib_quantum_viz():
    """Create quantum visualization using matplotlib instead of OpenGL."""
    print("Creating matplotlib-based quantum visualization...")
    
    try:
        from unified_hybrid_molecular_simulation import create_atom_simulation, AtomConfig
        from config import SIZE_X, SIZE_Y, SIZE_Z
        
        # Create hydrogen atom with 2p orbital
        hydrogen_config = AtomConfig(
            atomic_number=1, 
            position=(SIZE_X//2, SIZE_Y//2, SIZE_Z//2),
            electron_configs=[(2, 1, 0)]  # n=2, l=1, m=0 (2p orbital)
        )
        
        simulation = create_atom_simulation(hydrogen_config)
        
        # Get quantum data
        wavefunction = simulation.get_combined_wavefunction()
        if hasattr(wavefunction, 'detach'):
            wavefunction_np = wavefunction.detach().cpu().numpy()
        else:
            wavefunction_np = np.array(wavefunction)
        
        # Calculate probability density
        data = np.abs(wavefunction_np)**2
        
        print(f"Quantum data shape: {data.shape}")
        print(f"Data range: {data.min():.8f} to {data.max():.8f}")
        
        # Subsample for visualization
        step = 6
        data_sub = data[::step, ::step, ::step]
        threshold = data_sub.max() * 0.01  # 1% threshold
        
        mask = data_sub > threshold
        x_coords, y_coords, z_coords = np.where(mask)
        values = data_sub[mask]
        
        # Limit points for performance
        max_points = 1000
        if len(values) > max_points:
            indices = np.random.choice(len(values), max_points, replace=False)
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]
            z_coords = z_coords[indices]
            values = values[indices]
        
        # Convert to world coordinates
        center = SIZE_X // 2
        x_coords = (x_coords * step - center) * 0.1
        y_coords = (y_coords * step - center) * 0.1
        z_coords = (z_coords * step - center) * 0.1
        
        print(f"Found {len(values)} quantum points for visualization")
        
        # Create matplotlib 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot quantum points with color based on probability
        norm_values = (values - values.min()) / (values.max() - values.min()) if values.max() > values.min() else np.ones_like(values)
        
        scatter = ax.scatter(x_coords, y_coords, z_coords, 
                           c=norm_values, 
                           cmap='hot', 
                           s=20 + norm_values * 50,  # Size based on probability
                           alpha=0.8,
                           edgecolors='black',
                           linewidths=0.5)
        
        # Add nucleus at origin
        ax.scatter([0], [0], [0], c='red', s=200, marker='*', label='Nucleus')
        
        # Add coordinate axes
        axis_length = 5
        ax.plot([0, axis_length], [0, 0], [0, 0], 'r-', linewidth=3, label='X axis')
        ax.plot([0, 0], [0, axis_length], [0, 0], 'g-', linewidth=3, label='Y axis')
        ax.plot([0, 0], [0, 0], [0, axis_length], 'b-', linewidth=3, label='Z axis')
        
        # Set labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title('Hydrogen Atom 2p Orbital - Electron Probability Density\n(Matplotlib 3D Visualization)')
        
        # Add colorbar
        plt.colorbar(scatter, shrink=0.5, aspect=20, label='Probability Density')
        
        # Add legend
        ax.legend()
        
        # Set equal aspect ratio
        max_range = max(
            x_coords.max() - x_coords.min(),
            y_coords.max() - y_coords.min(),
            z_coords.max() - z_coords.min()
        ) / 2
        
        mid_x = (x_coords.max() + x_coords.min()) * 0.5
        mid_y = (y_coords.max() + y_coords.min()) * 0.5
        mid_z = (z_coords.max() + z_coords.min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add animation
        def animate(frame):
            ax.view_init(elev=20, azim=frame * 2)
            return [scatter]
        
        anim = animation.FuncAnimation(fig, animate, frames=180, interval=50, blit=False)
        
        # Add controls text
        plt.figtext(0.02, 0.02, 
                   "Interactive Controls:\n"
                   "• Mouse: Rotate view\n"
                   "• Scroll: Zoom\n"
                   "• Auto-rotation active",
                   fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        print("\nMatplotlib visualization ready!")
        print("Controls:")
        print("  - Mouse drag: Rotate view")
        print("  - Mouse scroll: Zoom")
        print("  - Auto-rotation: Enabled")
        print("  - Close window to exit")
        
        plt.tight_layout()
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"Error creating quantum visualization: {e}")
        import traceback
        traceback.print_exc()
        
        # Create fallback test visualization
        print("\nCreating fallback test visualization...")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create test data
        n_points = 500
        theta = np.linspace(0, 2*np.pi, n_points)
        phi = np.linspace(0, np.pi, n_points//5)
        
        x, y, z, colors = [], [], [], []
        for i, t in enumerate(theta):
            for j, p in enumerate(phi):
                r = 2 + 0.5 * np.sin(3*t) * np.cos(2*p)
                x.append(r * np.sin(p) * np.cos(t))
                y.append(r * np.sin(p) * np.sin(t))
                z.append(r * np.cos(p))
                colors.append(0.5 + 0.5 * np.sin(t + p))
        
        scatter = ax.scatter(x, y, z, c=colors, cmap='hot', s=30, alpha=0.7)
        ax.scatter([0], [0], [0], c='red', s=100, marker='*')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Test 3D Visualization (Fallback)')
        
        plt.colorbar(scatter)
        plt.show()
        
        return False

def main():
    print("=== Matplotlib Quantum Visualization ===")
    print("This bypasses PyQt OpenGL issues by using matplotlib for 3D rendering.")
    
    success = create_matplotlib_quantum_viz()
    
    if success:
        print("Matplotlib visualization completed successfully!")
    else:
        print("Used fallback test visualization.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
