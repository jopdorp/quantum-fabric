#!/usr/bin/env python3
"""
Pygame + OpenGL quantum visualization - more robust windowing system.
This should avoid the GTK/libdecor issues we're seeing.
"""

import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import torch
import time
import numpy as np

# Import our simulation components
sys.path.append('.')

class PygameQuantumViz:
    def __init__(self):
        self.width = 1200
        self.height = 800
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = -20.0
        self.threshold = 0.03
        self.evolution_step = 0
        self.max_evolution_steps = 200
        self.auto_evolve = True
        self.points = np.array([])
        self.colors = np.array([])
        self.point_sizes = np.array([])
        self.simulation = None
        self.last_evolution = time.time()
        self.evolution_interval = 0.1  # seconds
        
    def init_pygame(self):
        """Initialize pygame and OpenGL."""
        print("Initializing Pygame + OpenGL...")
        
        # Initialize pygame
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Quantum Visualization - Pygame + OpenGL")
        
        print("Pygame initialized successfully")
        
        # Initialize OpenGL
        glClearColor(0.1, 0.1, 0.2, 1.0)  # Dark blue background
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
        print("OpenGL initialized successfully")
        
    def init_simulation(self):
        """Initialize quantum simulation."""
        print("Initializing quantum simulation...")
        from unified_hybrid_molecular_simulation import create_atom_simulation, AtomConfig
        from config import SIZE_X, SIZE_Y, SIZE_Z
        
        # Create hydrogen atom with 2p orbital
        hydrogen_config = AtomConfig(
            atomic_number=1, 
            position=(SIZE_X//2, SIZE_Y//2, SIZE_Z//2),
            electron_configs=[(3, 2, 1)]  # n=2, l=1, m=0 (2p orbital)
        )
        
        self.simulation = create_atom_simulation(hydrogen_config)
        
        # Add momentum to the electron by applying a phase gradient
        from config import get_coordinate_tensors
        X, Y, Z = get_coordinate_tensors()
        
        # Create momentum phase shift: e^(i * p · r) where p is momentum vector
        momentum = torch.tensor([0.1, 0.0, 0.1])  # momentum in x, y, z directions
        
        # Calculate phase shift across the grid
        if Z is not None:
            # 3D case
            phase_shift = momentum[0] * X + momentum[1] * Y + momentum[2] * Z
        else:
            # 2D case - only use x and y components
            phase_shift = momentum[0] * X + momentum[1] * Y
        
        # Apply momentum by multiplying wavefunction by phase factor
        self.simulation.unified_wavefunction *= torch.exp(1j * phase_shift * 0.01)  # Small momentum
        
        self.SIZE_X, self.SIZE_Y, self.SIZE_Z = SIZE_X, SIZE_Y, SIZE_Z
        self.update_quantum_data()
        print(f"Simulation initialized with {len(self.points)} quantum points")
        
    def update_quantum_data(self):
        """Update quantum data from simulation."""
        if not self.simulation:
            print("No simulation available for visualization")
            return
            
        # Get wavefunction data
        wavefunction = self.simulation.get_combined_wavefunction()
        
        # Calculate probability density
        data = torch.abs(wavefunction)**2
        
        # Handle both 2D and 3D data
        if len(data.shape) == 2:
            # 2D data - add a dummy Z dimension
            data = data.unsqueeze(2)  # Add Z dimension
            is_2d = True
        else:
            is_2d = False
        
        # Subsample for visualization
        step = 6
        data_sub = data[::step, ::step, ::step]
        threshold_value = data_sub.max() * self.threshold
        
        mask = data_sub > threshold_value
        x_coords, y_coords, z_coords = torch.where(mask)
        values = data_sub[mask]
        
        # For 2D data, flatten Z coordinates
        if is_2d:
            z_coords = torch.zeros_like(x_coords)
        
        print(f"Found {len(values)} points above threshold")
        
        # Limit points for performance - use topk instead of argsort for better performance
        max_points = 1000
        if len(values) > max_points:
            # topk is much faster than argsort when we only need the top k values
            top_values, indices = torch.topk(values, max_points)
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]
            z_coords = z_coords[indices]
            values = top_values
            print(f"Reduced to {max_points} points for performance")
        
        # Convert to world coordinates using vectorized operations
        center = self.SIZE_X // 2
        scale = 0.05
        
        if len(values) > 0:
            # Vectorized coordinate conversion
            positions = torch.stack([x_coords, y_coords, z_coords], dim=1).float()
            positions = (positions * step - center) * scale
            self.points = positions.cpu().numpy()
            
            # Vectorized color mapping (hot colormap)
            val_min = values.min()
            val_max = values.max()
            val_range = val_max - val_min if val_max > val_min else 1.0
            val_norm = (values - val_min) / val_range
            
            # Vectorized hot colormap calculation
            colors = torch.zeros(len(values), 3)
            
            # Red component
            colors[:, 0] = torch.where(val_norm < 0.33, val_norm * 3, 1.0)
            
            # Green component  
            colors[:, 1] = torch.where(val_norm < 0.33, 0.0,
                                     torch.where(val_norm < 0.66, (val_norm - 0.33) * 3, 1.0))
            
            # Blue component
            colors[:, 2] = torch.where(val_norm < 0.66, 0.0, (val_norm - 0.66) * 3)
            
            self.colors = colors.cpu().numpy()
            
            # Pre-calculate point sizes for batched rendering
            intensities = colors.mean(dim=1)
            self.point_sizes = (2.0 + 8.0 * intensities).cpu().numpy()
        else:
            self.points = np.array([])
            self.colors = np.array([])
            self.point_sizes = np.array([])
                
    def draw_scene(self):
        """Draw the quantum visualization scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Camera transformations
        glTranslatef(0.0, 0.0, self.zoom)
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        
        # Draw coordinate axes
        self.draw_axes()
        
        # Draw nucleus
        self.draw_nucleus()
        
        # Draw quantum points
        self.draw_quantum_points()
        
        pygame.display.flip()
        
    def draw_axes(self):
        """Draw coordinate axes."""
        glLineWidth(3.0)
        glBegin(GL_LINES)
        
        # X axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(3.0, 0.0, 0.0)
        
        # Y axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 3.0, 0.0)
        
        # Z axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 3.0)
        
        glEnd()
        glLineWidth(1.0)
        
    def draw_nucleus(self):
        """Draw nucleus as a red point."""
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.2, 0.2)
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()
        glPointSize(1.0)
        
    def draw_quantum_points(self):
        """Draw quantum probability points with optimized batching."""
        if len(self.points) == 0:
            return
            
        # Group points by size to minimize OpenGL state changes
        size_groups = {}
        for i in range(len(self.points)):
            if i < len(self.colors):
                color = self.colors[i]
                # Calculate point size based on color intensity (brightness)
                intensity = (color[0] + color[1] + color[2]) / 3.0
                point_size = int(2.0 + 8.0 * intensity)  # Quantize to reduce groups
            else:
                point_size = 4
                
            if point_size not in size_groups:
                size_groups[point_size] = {'points': [], 'colors': []}
            size_groups[point_size]['points'].append(self.points[i])
            if i < len(self.colors):
                size_groups[point_size]['colors'].append(self.colors[i])
            else:
                size_groups[point_size]['colors'].append([1.0, 1.0, 0.0])
        
        # Draw each size group in batch
        for size, group in size_groups.items():
            glPointSize(float(size))
            glBegin(GL_POINTS)
            for j, point in enumerate(group['points']):
                color = group['colors'][j]
                glColor3f(color[0], color[1], color[2])
                glVertex3f(point[0], point[1], point[2])
            glEnd()
            
        glPointSize(1.0)  # Reset point size
        
    def handle_events(self):
        """Handle pygame events."""
        mouse_pressed = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.auto_evolve = not self.auto_evolve
                    print(f"Auto evolution: {'ON' if self.auto_evolve else 'OFF'}")
                elif event.key == pygame.K_s:
                    # Manual step
                    self.evolve_simulation()
                elif event.key == pygame.K_UP:
                    self.threshold *= 1.2
                    self.update_quantum_data()
                    print(f"Threshold: {self.threshold:.4f}")
                elif event.key == pygame.K_DOWN:
                    self.threshold /= 1.2
                    self.update_quantum_data()
                    print(f"Threshold: {self.threshold:.4f}")
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pressed = True
                
        # Handle mouse movement for rotation
        if pygame.mouse.get_pressed()[0]:  # Left mouse button
            mouse_rel = pygame.mouse.get_rel()
            self.rotation_y += mouse_rel[0] * 0.5
            self.rotation_x += mouse_rel[1] * 0.5
        else:
            pygame.mouse.get_rel()  # Reset relative mouse movement
            
        # Handle mouse wheel for zoom
        keys = pygame.key.get_pressed()
        if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
            self.zoom += 0.5
        if keys[pygame.K_MINUS]:
            self.zoom -= 0.5
            
        return True
        
    def evolve_simulation(self):
        """Evolve the simulation one step."""
        if self.simulation:
            self.simulation.evolve_step(self.evolution_step)
            self.evolution_step = (self.evolution_step + 1) % self.max_evolution_steps
            self.update_quantum_data()
            
    def run(self):
        """Main game loop with 60 FPS rendering."""
        print("Starting 60 FPS quantum visualization...")
        print("Controls:")
        print("  - Mouse drag: Rotate view")
        print("  - +/- keys: Zoom in/out")
        print("  - Space: Toggle auto evolution")
        print("  - S: Manual evolution step")
        print("  - Up/Down arrows: Adjust threshold")
        print("  - ESC: Exit")
        
        clock = pygame.time.Clock()
        running = True
        
        # Separate simulation and rendering timing
        last_sim_time = time.time()
        sim_interval = 0.1  # Simulate every 100ms (10 FPS)
        
        while running:
            # Always handle events for 60 FPS responsiveness
            running = self.handle_events()
            
            # Auto evolution at lower frequency
            current_time = time.time()
            if self.auto_evolve and (current_time - last_sim_time) > sim_interval:
                self.evolve_simulation()
                last_sim_time = current_time
            
            # Always draw scene at 60 FPS for smooth interaction
            self.draw_scene()
            
            # Maintain 60 FPS
            clock.tick(60)
            
        pygame.quit()

def main():
    print("=== Pygame + OpenGL Quantum Visualization ===")
    print("This uses pygame for robust windowing and OpenGL for 3D rendering.")
    
    try:
        viz = PygameQuantumViz()
        viz.init_pygame()
        viz.init_simulation()
        viz.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
