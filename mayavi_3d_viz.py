#!/usr/bin/env python3
"""
Mayavi-based 3D Quantum Simulation Visualizer

Real-time interactive 3D visualization of quantum wavefunctions using Mayavi.
Supports volume rendering, isosurfaces, and nucleus visualization.
"""

from mayavi import mlab
import numpy as np
import torch
from threading import Thread
import time
from typing import Optional

from unified_hybrid_molecular_simulation import create_atom_simulation, AtomConfig
from config import SIZE_X, SIZE_Y, SIZE_Z

class MayaviQuantumViz:
    def __init__(self, simulation):
        self.simulation = simulation
        self.running = False
        self.frame_count = 0
        self.paused = False
        self.visualization_mode = 'volume'  # 'volume', 'isosurface', 'both'
        
        # Visualization parameters
        self.threshold_multiplier = 0.15
        self.opacity = 0.6
        self.colormap = 'hot'
        
    def setup_scene(self):
        """Setup the 3D visualization scene."""
        # Create figure with dark background
        self.fig = mlab.figure(size=(1200, 800), bgcolor=(0.1, 0.1, 0.2))
        mlab.title('3D Quantum Simulation - Real Time', height=0.9, size=0.3, color=(1, 1, 1))
        
        # Get initial data
        psi = self.simulation.get_combined_wavefunction()
        if isinstance(psi, torch.Tensor):
            prob_density = torch.abs(psi)**2
            prob_density = prob_density.detach().cpu().numpy()
        else:
            prob_density = np.abs(psi)**2
        
        print(f"Wavefunction shape: {psi.shape}")
        print(f"Probability density range: {prob_density.min():.6f} to {prob_density.max():.6f}")
        
        # Volume rendering for electron probability cloud
        self.volume_source = mlab.pipeline.scalar_field(prob_density)
        
        if self.visualization_mode in ['volume', 'both']:
            self.volume = mlab.pipeline.volume(self.volume_source, 
                                             vmin=0, vmax=prob_density.max()*0.8)
            # Set volume rendering properties
            self.volume.trait_set(volume_mapper_type='texture_3d')
            
        # Isosurface for clearer structure
        if self.visualization_mode in ['isosurface', 'both']:
            threshold = self.threshold_multiplier * prob_density.max()
            self.iso = mlab.contour3d(prob_density, contours=[threshold], 
                                     opacity=self.opacity, colormap=self.colormap)
        
        # Add nuclei as glowing spheres
        self.nuclei_objects = []
        for i, nucleus in enumerate(self.simulation.nuclei):
            # Main nucleus sphere
            sphere = mlab.points3d([nucleus.x], [nucleus.y], [nucleus.z],
                                  scale_factor=12, color=(0.8, 0.2, 0.2),
                                  resolution=20, name=f'Nucleus_{i}')
            
            # Glow effect around nucleus  
            glow = mlab.points3d([nucleus.x], [nucleus.y], [nucleus.z],
                               scale_factor=20, color=(1, 0.5, 0.5),
                               opacity=0.3, resolution=20, name=f'Glow_{i}')
            
            self.nuclei_objects.extend([sphere, glow])
            
            # Add nucleus label
            mlab.text3d(nucleus.x + 15, nucleus.y, nucleus.z, 
                       f'Z={nucleus.atomic_number}', scale=8, color=(1, 1, 1))
        
        # Add coordinate axes
        axes_extent = [0, SIZE_X, 0, SIZE_Y, 0, SIZE_Z]
        mlab.axes(extent=axes_extent, color=(0.7, 0.7, 0.7), line_width=2)
        
        # Add colorbar
        if hasattr(self, 'iso'):
            mlab.colorbar(self.iso, title='Probability Density', orientation='vertical')
        elif hasattr(self, 'volume'):
            mlab.colorbar(self.volume, title='Probability Density', orientation='vertical')
        
        # Set up nice camera view
        mlab.view(45, 60, distance='auto', focalpoint='auto')
        
        # Add some lighting
        self.fig.scene.light_manager.lights[0].intensity = 0.8
        
        print("3D scene setup complete!")
        
    def update_visualization(self):
        """Update the visualization with current simulation state."""
        try:
            # Get current wavefunction
            psi = self.simulation.get_combined_wavefunction()
            if isinstance(psi, torch.Tensor):
                prob_density = torch.abs(psi)**2
                prob_density = prob_density.detach().cpu().numpy()
            else:
                prob_density = np.abs(psi)**2
            
            # Update volume rendering
            if hasattr(self, 'volume'):
                self.volume.mlab_source.scalars = prob_density
                
            # Update isosurface with adaptive threshold
            if hasattr(self, 'iso'):
                threshold = self.threshold_multiplier * prob_density.max()
                self.iso.mlab_source.scalars = prob_density
                self.iso.contour.contours = [threshold]
            
            # Update nuclei positions (if they move)
            for i, nucleus in enumerate(self.simulation.nuclei):
                if i * 2 < len(self.nuclei_objects):
                    # Update main sphere
                    self.nuclei_objects[i*2].mlab_source.set(
                        x=[nucleus.x], y=[nucleus.y], z=[nucleus.z]
                    )
                    # Update glow
                    if i*2+1 < len(self.nuclei_objects):
                        self.nuclei_objects[i*2+1].mlab_source.set(
                            x=[nucleus.x], y=[nucleus.y], z=[nucleus.z]
                        )
                        
        except Exception as e:
            print(f"Error updating visualization: {e}")
    
    def animation_loop(self):
        """Main animation loop."""
        print("Starting animation loop...")
        while self.running:
            start_time = time.time()
            
            if not self.paused:
                try:
                    # Evolve simulation
                    self.simulation.evolve_step(self.frame_count)
                    self.frame_count += 1
                    
                    # Update visualization
                    self.update_visualization()
                    
                except Exception as e:
                    print(f"Error in animation loop: {e}")
                    break
            
            # Control frame rate (aim for ~20 FPS for smoother visualization)
            elapsed = time.time() - start_time
            target_dt = 1.0 / 20.0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
                
            if self.frame_count % 10 == 0:  # Print every 10 frames
                actual_fps = 1.0 / (time.time() - start_time)
                print(f"Frame {self.frame_count}, FPS: {actual_fps:.1f}")
        
        print("Animation loop stopped.")
    
    def setup_keyboard_controls(self):
        """Setup keyboard controls for the visualization."""
        def on_key_press(vtk_obj, event):
            key = vtk_obj.GetKeyCode()
            
            if key == 'p' or key == ' ':  # Pause/unpause
                self.paused = not self.paused
                print(f"Animation {'paused' if self.paused else 'resumed'}")
                
            elif key == 'r':  # Reset camera
                mlab.view(45, 60, distance='auto')
                print("Camera view reset")
                
            elif key == 's':  # Save screenshot
                filename = f"quantum_3d_frame_{self.frame_count:05d}.png"
                mlab.savefig(filename, size=(1920, 1080))
                print(f"Saved screenshot: {filename}")
                
            elif key == 'v':  # Switch visualization mode
                if self.visualization_mode == 'volume':
                    self.visualization_mode = 'isosurface'
                elif self.visualization_mode == 'isosurface':
                    self.visualization_mode = 'both'
                else:
                    self.visualization_mode = 'volume'
                print(f"Visualization mode: {self.visualization_mode}")
                
            elif key == '+' or key == '=':  # Increase threshold
                self.threshold_multiplier = min(1.0, self.threshold_multiplier * 1.1)
                print(f"Threshold multiplier: {self.threshold_multiplier:.3f}")
                
            elif key == '-' or key == '_':  # Decrease threshold
                self.threshold_multiplier = max(0.01, self.threshold_multiplier * 0.9)
                print(f"Threshold multiplier: {self.threshold_multiplier:.3f}")
                
            elif key == 'h':  # Help
                print("\\nKeyboard Controls:")
                print("  p/space - Pause/unpause animation")
                print("  r - Reset camera view")
                print("  s - Save screenshot")  
                print("  v - Switch visualization mode")
                print("  +/- - Adjust threshold")
                print("  h - Show this help")
                print("  q - Quit")
                
            elif key == 'q':  # Quit
                self.stop()
                
        # Attach keyboard handler
        self.fig.scene.interactor.add_observer('KeyPressEvent', on_key_press)
    
    def start(self):
        """Start the real-time visualization."""
        print("Setting up 3D scene...")
        self.setup_scene()
        
        print("Setting up keyboard controls...")
        self.setup_keyboard_controls()
        
        print("\\nKeyboard Controls:")
        print("  p/space - Pause/unpause")
        print("  r - Reset camera")
        print("  s - Save screenshot")
        print("  h - Help")
        print()
        
        print("Starting animation...")
        self.running = True
        
        # Start animation in background thread
        self.anim_thread = Thread(target=self.animation_loop, daemon=True)
        self.anim_thread.start()
        
        # Show scene (this blocks until window is closed)
        print("Opening 3D visualization window...")
        mlab.show()
        
    def stop(self):
        """Stop the animation."""
        print("Stopping animation...")
        self.running = False
        mlab.close(all=True)


def main():
    """Main function to run the 3D quantum visualization."""
    print("Creating 3D quantum simulation...")
    
    # Create a hydrogen atom at the center
    hydrogen_config = AtomConfig(
        atomic_number=1, 
        position=(SIZE_X//2, SIZE_Y//2, SIZE_Z//2)
    )
    
    simulation = create_atom_simulation(hydrogen_config)
    
    print("Starting 3D visualization...")
    viz = MayaviQuantumViz(simulation)
    viz.start()


if __name__ == "__main__":
    main()
