#!/usr/bin/env python3
"""
Alternative quantum visualization using matplotlib for 3D rendering with real-time evolution.
This bypasses PyQt OpenGL issues entirely and provides smooth real-time quantum stepping.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

# Import our simulation components
sys.path.append('.')

class RealTimeQuantumViz:
    def __init__(self):
        self.simulation = None
        self.evolution_step = 0
        self.max_evolution_steps = 200
        self.threshold = 0.01
        self.auto_evolve = True
        self.step_interval = 100  # milliseconds
        
    def init_simulation(self):
        """Initialize the quantum simulation."""
        print("Initializing quantum simulation...")
        
        from unified_hybrid_molecular_simulation import create_atom_simulation, AtomConfig
        from config import SIZE_X, SIZE_Y, SIZE_Z
        
        # Create hydrogen atom with 2p orbital
        hydrogen_config = AtomConfig(
            atomic_number=1, 
            position=(SIZE_X//2, SIZE_Y//2, SIZE_Z//2),
            electron_configs=[(2, 1, 0)]  # n=2, l=1, m=0 (2p orbital)
        )
        
        self.simulation = create_atom_simulation(hydrogen_config)
        self.SIZE_X, self.SIZE_Y, self.SIZE_Z = SIZE_X, SIZE_Y, SIZE_Z
        
    def get_quantum_data(self):
        """Get current quantum data from simulation."""
        if not self.simulation:
            return [], [], [], []
            
        # Get wavefunction data
        wavefunction = self.simulation.get_combined_wavefunction()
        if hasattr(wavefunction, 'detach'):
            wavefunction_np = wavefunction.detach().cpu().numpy()
        else:
            wavefunction_np = np.array(wavefunction)
        
        # Calculate probability density
        data = np.abs(wavefunction_np)**2
        
        # Subsample for visualization
        step = 6
        data_sub = data[::step, ::step, ::step]
        threshold_value = data_sub.max() * self.threshold
        
        mask = data_sub > threshold_value
        x_coords, y_coords, z_coords = np.where(mask)
        values = data_sub[mask]
        
        # Limit points for performance
        max_points = 800
        if len(values) > max_points:
            indices = np.random.choice(len(values), max_points, replace=False)
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]
            z_coords = z_coords[indices]
            values = values[indices]
        
        # Convert to world coordinates
        center = self.SIZE_X // 2
        x_coords = (x_coords * step - center) * 0.08
        y_coords = (y_coords * step - center) * 0.08
        z_coords = (z_coords * step - center) * 0.08
        
        return x_coords, y_coords, z_coords, values
        
    def create_visualization(self):
        """Create the real-time matplotlib visualization."""
        print("Creating real-time matplotlib quantum visualization...")
        
        # Initialize simulation
        self.init_simulation()
        
        # Create figure and 3D axis
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Get initial quantum data
        x_coords, y_coords, z_coords, values = self.get_quantum_data()
        
        if len(values) > 0:
            # Normalize values for color mapping
            norm_values = (values - values.min()) / (values.max() - values.min()) if values.max() > values.min() else np.ones_like(values)
            
            # Create scatter plot WITHOUT black edges
            self.scatter = self.ax.scatter(x_coords, y_coords, z_coords, 
                                         c=norm_values, 
                                         cmap='hot', 
                                         s=15 + norm_values * 40,  # Size based on probability
                                         alpha=0.9,
                                         edgecolors='none')  # Remove black edges!
        else:
            # Empty plot if no data
            self.scatter = self.ax.scatter([], [], [], c=[], cmap='hot')
        
        # Add nucleus at origin (no black edge)
        self.nucleus = self.ax.scatter([0], [0], [0], c='red', s=150, marker='*', 
                                     label='Nucleus', alpha=1.0, edgecolors='none')
        
        # Add coordinate axes
        axis_length = 4
        self.ax.plot([0, axis_length], [0, 0], [0, 0], 'r-', linewidth=2, alpha=0.7, label='X axis')
        self.ax.plot([0, 0], [0, axis_length], [0, 0], 'g-', linewidth=2, alpha=0.7, label='Y axis')
        self.ax.plot([0, 0], [0, 0], [0, axis_length], 'b-', linewidth=2, alpha=0.7, label='Z axis')
        
        # Set labels and title
        self.ax.set_xlabel('X Position', fontsize=12)
        self.ax.set_ylabel('Y Position', fontsize=12)
        self.ax.set_zlabel('Z Position', fontsize=12)
        self.ax.set_title('Real-Time Hydrogen Atom Quantum Evolution\n(2p Orbital - No OpenGL)', fontsize=14)
        
        # Set reasonable axis limits
        limit = 8
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-limit, limit)
        
        # Add colorbar
        self.cbar = plt.colorbar(self.scatter, shrink=0.6, aspect=20, label='Probability Density')
        
        # Add controls
        self.add_controls()
        
        print(f"Initial quantum data: {len(x_coords)} points")
        print("Real-time controls added!")
        
    def add_controls(self):
        """Add interactive controls for real-time evolution."""
        # Make room for controls
        plt.subplots_adjust(bottom=0.2)
        
        # Threshold slider
        ax_threshold = plt.axes([0.1, 0.1, 0.3, 0.03])
        self.threshold_slider = Slider(ax_threshold, 'Threshold', 0.001, 0.1, 
                                     valinit=self.threshold, valstep=0.001, valfmt='%.3f')
        self.threshold_slider.on_changed(self.update_threshold)
        
        # Play/Pause button
        ax_play = plt.axes([0.45, 0.1, 0.1, 0.04])
        self.play_button = Button(ax_play, 'Pause')
        self.play_button.on_clicked(self.toggle_play)
        
        # Step button
        ax_step = plt.axes([0.57, 0.1, 0.1, 0.04])
        self.step_button = Button(ax_step, 'Step')
        self.step_button.on_clicked(self.manual_step)
        
        # Speed slider
        ax_speed = plt.axes([0.7, 0.1, 0.25, 0.03])
        self.speed_slider = Slider(ax_speed, 'Speed', 10, 500, 
                                 valinit=self.step_interval, valstep=10, valfmt='%d ms')
        self.speed_slider.on_changed(self.update_speed)
        
        # Add status text
        plt.figtext(0.02, 0.02, 
                   "Real-Time Quantum Evolution:\n"
                   "• Threshold: Adjust visibility\n"
                   "• Play/Pause: Control evolution\n"
                   "• Step: Manual evolution\n"
                   "• Speed: Animation speed\n"
                   "• Mouse: Rotate/zoom view",
                   fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                   
    def update_threshold(self, val):
        """Update probability threshold."""
        self.threshold = val
        self.update_plot()
        
    def toggle_play(self, event):
        """Toggle play/pause."""
        self.auto_evolve = not self.auto_evolve
        self.play_button.label.set_text('Play' if not self.auto_evolve else 'Pause')
        plt.draw()
        
    def manual_step(self, event):
        """Manual evolution step."""
        self.evolve_step()
        self.update_plot()
        
    def update_speed(self, val):
        """Update animation speed."""
        self.step_interval = int(val)
        
    def evolve_step(self):
        """Evolve the simulation one step."""
        if self.simulation:
            self.simulation.evolve_step(self.evolution_step)
            self.evolution_step = (self.evolution_step + 1) % self.max_evolution_steps
            
    def update_plot(self):
        """Update the 3D plot with current quantum data."""
        # Get new quantum data
        x_coords, y_coords, z_coords, values = self.get_quantum_data()
        
        # Remove old scatter plot
        self.scatter.remove()
        
        if len(values) > 0:
            # Normalize values for color mapping
            norm_values = (values - values.min()) / (values.max() - values.min()) if values.max() > values.min() else np.ones_like(values)
            
            # Create new scatter plot WITHOUT black edges
            self.scatter = self.ax.scatter(x_coords, y_coords, z_coords, 
                                         c=norm_values, 
                                         cmap='hot', 
                                         s=15 + norm_values * 40,
                                         alpha=0.9,
                                         edgecolors='none')  # No black edges!
        else:
            # Empty plot if no data
            self.scatter = self.ax.scatter([], [], [], c=[], cmap='hot')
        
        # Update title with step info
        self.ax.set_title(f'Real-Time Hydrogen Quantum Evolution (Step {self.evolution_step})\n'
                         f'Points: {len(x_coords)}, Threshold: {self.threshold:.3f}', fontsize=14)
        
    def animate(self, frame):
        """Animation function for real-time updates."""
        if self.auto_evolve:
            self.evolve_step()
            self.update_plot()
        
        # Gentle rotation
        self.ax.view_init(elev=20 + 10*np.sin(frame*0.02), azim=frame*0.5)
        
        return [self.scatter]
        
    def start_visualization(self):
        """Start the real-time visualization."""
        # Create animation
        self.anim = animation.FuncAnimation(self.fig, self.animate, 
                                          interval=self.step_interval, 
                                          blit=False, cache_frame_data=False)
        
        print("\nReal-time quantum visualization started!")
        print("Controls:")
        print("  - Threshold slider: Adjust point visibility")
        print("  - Play/Pause button: Control real-time evolution") 
        print("  - Step button: Manual evolution steps")
        print("  - Speed slider: Adjust animation speed")
        print("  - Mouse: Rotate and zoom view")
        print("  - Close window to exit")
        
        plt.tight_layout()
        plt.show()
        
        print("\nReal-time quantum visualization started!")
        print("Controls:")
        print("  - Threshold slider: Adjust point visibility")
        print("  - Play/Pause button: Control real-time evolution") 
        print("  - Step button: Manual evolution steps")
        print("  - Speed slider: Adjust animation speed")
        print("  - Mouse: Rotate and zoom view")
        print("  - Close window to exit")
        
        plt.tight_layout()
        plt.show()

def create_matplotlib_quantum_viz():
    """Create and start the real-time quantum visualization."""
    try:
        viz = RealTimeQuantumViz()
        viz.create_visualization()
        viz.start_visualization()
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
        
        # No black edges in fallback either
        scatter = ax.scatter(x, y, z, c=colors, cmap='hot', s=30, alpha=0.7, edgecolors='none')
        ax.scatter([0], [0], [0], c='red', s=100, marker='*', edgecolors='none')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Test 3D Visualization (Fallback)')
        
        plt.colorbar(scatter)
        plt.show()
        
        return False

def main():
    print("=== Real-Time Matplotlib Quantum Visualization ===")
    print("This bypasses PyQt OpenGL issues and provides real-time quantum evolution.")
    
    success = create_matplotlib_quantum_viz()
    
    if success:
        print("Real-time quantum visualization completed successfully!")
    else:
        print("Used fallback test visualization.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
