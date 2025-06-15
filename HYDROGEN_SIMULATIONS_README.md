# Hydrogen Simulations - 2D/3D Mode Support

This document describes the hydrogen simulation scripts that now support both 2D and 3D simulation modes.

## Available Simulations

### 1. Hydrogen Molecule Simulation (`hydrogen_molecule_sim.py`)

Simulates H2 molecule formation with bonding and antibonding orbitals.

**Interactive Mode:**
```bash
python hydrogen_molecule_sim.py
```

**Command Line Mode:**
```bash
python hydrogen_molecule_sim.py 2D          # 2D simulation only
python hydrogen_molecule_sim.py 3D          # 3D simulation only  
python hydrogen_molecule_sim.py VIZ2D       # 2D with real-time visualization
python hydrogen_molecule_sim.py VIZ3D       # 3D with real-time visualization
```

**Features:**
- Bonding vs antibonding molecular orbitals
- Real-time 3D visualization with pygame
- Nuclear motion and electron-electron interactions
- Video output for both modes

### 2. Hydrogen Atom Simulation (`hydrogen_atom_sim.py`)

Demonstrates hydrogen atom orbital progression through different quantum states.

**Interactive Mode:**
```bash
python hydrogen_atom_sim.py
```

**Command Line Mode:**
```bash
python hydrogen_atom_sim.py 2D          # 2D simulation only
python hydrogen_atom_sim.py 3D          # 3D simulation only
python hydrogen_atom_sim.py VIZ2D       # 2D with real-time visualization
python hydrogen_atom_sim.py VIZ3D       # 3D with real-time visualization
python hydrogen_atom_sim.py COMPARE     # Both 2D and 3D comparison
```

**Orbital Progression:**
1. 1s orbital (400 steps)
2. 2p orbital (400 steps) 
3. 3d orbital (400 steps)
4. 2p orbital with dynamics (800 steps)

## Mode Differences

### 2D Mode
- **Grid Size**: 1024×1024 (high resolution)
- **Memory Usage**: ~16 MB
- **Performance**: Fast
- **Use Case**: Good for testing, visualization, and quick results

### 3D Mode  
- **Grid Size**: 256×256×256 (reduced for memory)
- **Memory Usage**: ~256 MB
- **Performance**: Slower but more realistic
- **Use Case**: Accurate 3D quantum behavior

## Real-time Visualization

Both simulations support real-time 3D visualization using pygame and OpenGL.

**Controls:**
- **Mouse drag**: Rotate view
- **+/- keys**: Zoom in/out
- **Space**: Toggle auto evolution
- **S**: Manual evolution step
- **Up/Down arrows**: Adjust visualization threshold
- **ESC**: Exit

## Technical Implementation

### Mode Switching
- Uses `config.set_simulation_mode()` to switch between 2D and 3D
- Dynamically creates appropriate coordinate tensors
- Automatically adjusts grid sizes and memory usage

### Coordinate Tensors
- **2D**: X, Y tensors (1024×1024)
- **3D**: X, Y, Z tensors (256×256×256)
- Retrieved via `config.get_coordinate_tensors()`

### Physics Engine
- Unified hybrid molecular simulation framework
- FFT-based wave propagation for both 2D and 3D
- Handles electron-electron repulsion and nuclear forces
- Real-time quantum evolution

## Output Files

- **Videos**: `*_2d.avi` and `*_3d.avi` for each mode
- **Format**: AVI with real/imaginary/phase/probability frames
- **Resolution**: Automatically scales with grid size

## Hardware Support

- **Intel XPU**: Primary acceleration (Intel Arc Graphics)
- **CUDA**: Secondary GPU acceleration
- **CPU**: Fallback mode

All simulations automatically detect and use the best available hardware.
