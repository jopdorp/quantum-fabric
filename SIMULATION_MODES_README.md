# Quantum Molecular Simulation - 2D/3D Mode Support

## Overview

This codebase now supports both 2D and 3D quantum molecular simulations. The system automatically adapts to the chosen mode, providing optimal performance and accuracy for each case.

## Key Features Restored/Added

### 2D Mode (High Performance)
- **Grid Size**: 1024x1024 (high resolution)
- **Memory Usage**: ~16 MB for complex wavefunctions
- **Performance**: Fast, suitable for rapid testing and development
- **Physics**: 2D quantum mechanics with proper angular momentum treatment
- **Use Case**: Quick prototyping, algorithm testing, educational demonstrations

### 3D Mode (Realistic Physics)
- **Grid Size**: 256x256x256 (optimized for memory)
- **Memory Usage**: ~256 MB for complex wavefunctions
- **Performance**: Slower but more realistic
- **Physics**: Full 3D quantum mechanics with spherical harmonics
- **Use Case**: Research, accurate molecular modeling, publication-quality results

## Fixed Components

### 1. Configuration System (`config.py`)
- Added `SIMULATION_MODE` global variable
- Created `set_simulation_mode()` function
- Automatic grid size and coordinate tensor setup
- Device-aware tensor creation (Intel XPU/CUDA/CPU)

### 2. Atomic Orbital Generation (`hybrid_molecular_simulation.py`)
- Updated `create_atom_electron()` to handle both 2D and 3D
- Proper angular momentum treatment for each mode
- Backward compatibility maintained

### 3. Force Calculations (`hybrid_molecular_simulation.py`)
- Fixed `compute_nuclear_forces()` for 2D/3D compatibility
- Updated `compute_force_from_density()` for proper dimensionality
- Automatic force vector sizing based on nucleus position dimensions

### 4. Unified Simulation Framework (`unified_hybrid_molecular_simulation.py`)
- Enhanced `AtomConfig` class for automatic mode detection
- Updated `create_atom_simulation()` for both modes
- Fixed coordinate system handling throughout

### 5. Test System (`hydrogen_molecule_sim.py`)
- Interactive mode selection
- Command-line argument support
- Real-time visualization integration
- Comparison tools for 2D vs 3D

## Usage Examples

### Command Line
```bash
# Quick 2D test
python hydrogen_molecule_sim.py 2D

# Full 3D simulation
python hydrogen_molecule_sim.py 3D

# Interactive mode selection
python hydrogen_molecule_sim.py INTERACTIVE

# Real-time visualization (if display available)
python hydrogen_molecule_sim.py VIZ2D
python hydrogen_molecule_sim.py VIZ3D

# Compare modes
python hydrogen_molecule_sim.py COMPARE
```

### Interactive Mode Options
1. **2D mode** - Fast simulation with video output
2. **3D mode** - Realistic simulation with video output  
3. **2D with visualization** - Real-time pygame rendering
4. **3D with visualization** - Real-time pygame rendering
5. **2D vs 3D comparison** - Run both modes sequentially
6. **Bonding vs Antibonding in 2D** - Compare molecular orbital types
7. **Bonding vs Antibonding in 3D** - Compare molecular orbital types

### Programmatic Usage
```python
from config import set_simulation_mode
from unified_hybrid_molecular_simulation import AtomConfig, create_molecule_simulation

# Set to 2D mode
set_simulation_mode("2D")

# Create atoms (positions adapt to current mode)
hydrogen_configs = [
    AtomConfig(atomic_number=1, position=(100, 200)),  # 2D position
    AtomConfig(atomic_number=1, position=(150, 200))   # 2D position
]

# Simulation automatically uses 2D physics
simulation = create_molecule_simulation(hydrogen_configs)
```

## Performance Comparison

| Mode | Grid Size | Memory | Speed | Accuracy |
|------|-----------|--------|-------|----------|
| 2D   | 1024²     | ~16 MB | Fast  | Good for testing |
| 3D   | 256³      | ~256 MB| Slow  | Research quality |

## Technical Details

### 2D Physics Adaptations
- Angular momentum reduced to azimuthal quantum number only
- Radial wavefunctions preserved from 3D theory
- Nuclear forces calculated in 2D plane
- Proper normalization for 2D integrals

### 3D Physics Features
- Full spherical harmonics for angular momentum
- 3D nuclear motion and forces
- Complete electron-electron interactions
- Realistic molecular orbital shapes

### Automatic Mode Detection
The system automatically detects the simulation mode from:
1. Global `SIMULATION_MODE` setting
2. AtomConfig position dimensions
3. Coordinate tensor dimensions
4. Wavefunction tensor shapes

## Validation Status

✅ **2D Mode**: Fully tested and working
- Hydrogen molecule bonding/antibonding orbitals
- Proper electron dynamics
- Video generation
- Interactive visualization

✅ **3D Mode**: Fully tested and working  
- Full 3D quantum mechanics
- Realistic molecular shapes
- Proper force calculations
- Memory-optimized performance

✅ **Mode Switching**: Seamless transitions between modes
- No code changes required
- Automatic parameter adjustment
- Preserved simulation quality

## Troubleshooting

### Memory Issues
- Use 2D mode for large systems or limited memory
- Reduce grid size in config.py if needed
- Monitor GPU memory usage on Intel XPU/CUDA

### Performance Issues
- 2D mode is ~16x faster than 3D for the same resolution
- Use batched electron propagation (enabled by default)
- Enable vectorized repulsion calculations

### Visualization Issues
- Pygame visualization requires display/X11 forwarding
- Fallback to video-only mode if visualization fails
- Use VNC or local display for interactive features

## Future Enhancements

1. **Adaptive Resolution**: Automatic grid size selection based on system size
2. **Hybrid Mode**: 2D+1 approximation for certain systems
3. **GPU Memory Management**: Automatic batching for large systems
4. **Real-time Parameter Tuning**: Interactive adjustment during simulation
