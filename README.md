# Quantum Wavefunction Simulator

A comprehensive quantum mechanics simulation engine that solves the time-dependent Schrödinger equation for atomic and molecular systems using advanced numerical methods.

## Overview

This project implements a real-time quantum wavefunction simulator capable of modeling hydrogen atoms, hydrogen molecules, and multi-electron systems. The simulation uses the **split-step Fourier method** to propagate wavefunctions in time while accounting for various physical interactions including:

- Nuclear-electron Coulomb attraction
- Electron-electron repulsion (mean-field approximation)
- Nuclear-nuclear interactions (Coulomb repulsion + strong force)
- Quantum tunneling and orbital dynamics

## Physics and Mathematical Foundation

### 1. Time-Dependent Schrödinger Equation

The fundamental equation governing quantum mechanical evolution:

$$i\hbar \frac{\partial \psi(\mathbf{r}, t)}{\partial t} = \hat{H} \psi(\mathbf{r}, t)$$

Where:
- $\psi(\mathbf{r}, t)$ is the wavefunction
- $\hat{H}$ is the Hamiltonian operator
- $\hbar$ is the reduced Planck constant

The Hamiltonian consists of kinetic and potential energy terms:

$$\hat{H} = \hat{T} + \hat{V} = -\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r}, t)$$

### 2. Split-Step Fourier Method

The simulation employs the **split-step Fourier method** for time evolution, which factorizes the evolution operator:

$$\psi(t + \Delta t) = e^{-i\hat{H}\Delta t/\hbar} \psi(t) \approx e^{-i\hat{V}\Delta t/2\hbar} e^{-i\hat{T}\Delta t/\hbar} e^{-i\hat{V}\Delta t/2\hbar} \psi(t)$$

This second-order splitting scheme ensures numerical stability and conservation of probability.

#### Implementation Steps:

1. **Potential Evolution (Half Step)**: 
   $$\psi_1 = e^{-i V(\mathbf{r}) \Delta t / 2\hbar} \psi(t)$$

2. **Kinetic Evolution (Full Step)**:
   $$\tilde{\psi}_2 = \mathcal{F}[\psi_1]$$
   $$\tilde{\psi}_3 = e^{-i \hbar k^2 \Delta t / 2m} \tilde{\psi}_2$$
   $$\psi_2 = \mathcal{F}^{-1}[\tilde{\psi}_3]$$

3. **Potential Evolution (Half Step)**:
   $$\psi(t + \Delta t) = e^{-i V(\mathbf{r}) \Delta t / 2\hbar} \psi_2$$

Where $\mathcal{F}$ denotes the Fast Fourier Transform and $k$ is the momentum vector in frequency space.

### 3. Potentials and Forces

The simulation models three types of interactions with distinct physical origins:

#### 3.1 Electron-Nuclear Potential
The potential experienced by electrons near nuclei combines attractive and repulsive components:

$$V_{\text{nucleus}}(r) = -\frac{k_e Z e^2}{r} + A \frac{e^{-r/r_0}}{r + \epsilon}$$

Where:
- **Coulomb attraction**: $-k_e Z e^2/r$ (attractive for electrons)
- **Quantum repulsion**: $A e^{-r/r_0}/(r + \epsilon)$ (short-range repulsion)
- $k_e$ is Coulomb's constant, $Z$ is nuclear charge
- $A$ is repulsion strength (`NUCLEAR_REPULSION_STRENGTH`)
- $r_0$ is nuclear core radius (`NUCLEAR_CORE_RADIUS`)
- $\epsilon = 0.1$ prevents singularities

The repulsion term models quantum effects:
- **Pauli exclusion principle**: Prevents electron collapse into nucleus
- **Quantum uncertainty**: Finite nuclear size effects
- **Core electron screening**: In multi-electron atoms

#### 3.2 Electron-Electron Interactions
Mean-field approximation for electron repulsion:

$$V_{\text{ee}}(\mathbf{r}) = \alpha \int \frac{|\psi_j(\mathbf{r}')|^2}{|\mathbf{r} - \mathbf{r}'|} d^3\mathbf{r}'$$

Implemented via convolution with a Gaussian kernel:

$$V_{\text{ee}}(\mathbf{r}) \approx \alpha \cdot G_\sigma * |\psi_j(\mathbf{r})|^2$$

Where $G_\sigma$ is a Gaussian filter approximating the $1/r$ Coulomb interaction.

#### 3.3 Nuclear-Nuclear Forces (Molecules)
Inter-nuclear forces combine electromagnetic and strong interactions:

**Coulomb repulsion** (long-range):
$$\mathbf{F}_{\text{Coulomb}} = k_c \frac{Z_1 Z_2 (\mathbf{R}_2 - \mathbf{R}_1)}{|\mathbf{R}_2 - \mathbf{R}_1|^3}$$

**Strong nuclear force** (short-range attractive):
$$\mathbf{F}_{\text{strong}} = -A_s e^{-r/r_s} \frac{\mathbf{R}_2 - \mathbf{R}_1}{|\mathbf{R}_2 - \mathbf{R}_1|} \quad \text{for } r < r_s$$

### 4. Nuclear Dynamics

Nuclei are treated as classical particles with total force:

$$\mathbf{F}_{\text{total}} = \mathbf{F}_{\text{electronic}} + \mathbf{F}_{\text{nuclear}}$$

**Electronic force** from electron density:
$$\mathbf{F}_{\text{electronic}} = -\int |\psi(\mathbf{r})|^2 \nabla V_{\text{nucleus}}(\mathbf{r}) d^3\mathbf{r}$$

Discretized as:
$$\mathbf{F}_{\text{electronic}} = \sum_{i,j} |\psi_{i,j}|^2 \frac{\mathbf{r}_{i,j} - \mathbf{R}_{\text{nucleus}}}{|\mathbf{r}_{i,j} - \mathbf{R}_{\text{nucleus}}|^3}$$

**Nuclear force** from other nuclei (Section 3.3 forces applied directly).

### 5. Hydrogen Orbitals

The simulation can initialize exact hydrogen-like orbitals using quantum numbers $(n, l, m)$:

$$\psi_{nlm}(r, \theta, \phi) = R_{nl}(r) Y_l^m(\theta, \phi)$$

#### 5.1 Radial Component
$$R_{nl}(r) = \sqrt{\left(\frac{2}{na_0}\right)^3 \frac{(n-l-1)!}{2n(n+l)!}} e^{-r/na_0} \left(\frac{2r}{na_0}\right)^l L_{n-l-1}^{2l+1}\left(\frac{2r}{na_0}\right)$$

Where $L_{n-l-1}^{2l+1}$ are the associated Laguerre polynomials and $a_0$ is the Bohr radius.

**Energy eigenvalues** for hydrogen-like atoms:
$$E_n = -\frac{Z^2 e^2}{8\pi\epsilon_0 a_0 n^2} = -\frac{13.6 \text{ eV} \cdot Z^2}{n^2}$$

#### 5.2 Angular Component
$$Y_l^m(\theta, \phi) = \sqrt{\frac{2l+1}{4\pi}\frac{(l-|m|)!}{(l+|m|)!}} P_l^{|m|}(\cos\theta) e^{im\phi}$$

Where $P_l^{|m|}$ are the associated Legendre polynomials.

#### 5.3 Specific Orbital Forms

**1s orbital** ($n=1, l=0, m=0$):
$$\psi_{100} = \frac{1}{\sqrt{\pi a_0^3}} e^{-r/a_0}$$

**2p orbitals** ($n=2, l=1$):
$$\psi_{21\pm1} = \frac{1}{2\sqrt{6\pi a_0^3}} \frac{r}{a_0} e^{-r/2a_0} \sin\theta e^{\pm i\phi}$$
$$\psi_{210} = \frac{1}{2\sqrt{6\pi a_0^3}} \frac{r}{a_0} e^{-r/2a_0} \cos\theta$$

#### 5.4 2D Projection Implementation

For computational efficiency, the simulation projects 3D orbitals onto 2D grids:

- **s orbitals**: $|\psi|^2 \propto e^{-2r/na_0}$ (spherically symmetric)
- **p orbitals**: Dumbbell shapes with $\cos^2\theta$ or $\sin^2\theta$ angular dependence
- **d orbitals**: Four-lobed patterns from $Y_2^m$ angular functions

**Normalization condition**: $\int |\psi_{nlm}(\mathbf{r})|^2 d^3\mathbf{r} = 1$

## Numerical Methods and Optimizations

### 1. Absorbing Boundary Conditions

To prevent artificial reflections at computational boundaries:

$$\psi_{\text{absorbed}}(\mathbf{r}) = \psi(\mathbf{r}) \cdot \exp(-\alpha (r - r_{\text{start}})^3) \quad \text{for } r > r_{\text{start}}$$

The cubic exponential provides smooth absorption while preserving probability conservation.

### 2. Low-Pass Filtering

High-frequency noise is removed using Fourier space filtering:

$$\tilde{\psi}_{\text{filtered}}(\mathbf{k}) = \tilde{\psi}(\mathbf{k}) \cdot H(|\mathbf{k}| < k_{\text{cutoff}})$$

Where $H$ is the Heaviside step function.

### 3. Computational Optimizations

- **Caching**: FFT grids and kinetic energy operators are pre-computed and cached
- **Memory efficiency**: Streaming video output to avoid memory limitations
- **Vectorization**: All operations use NumPy vectorized functions
- **Adaptive time stepping**: Configurable $\Delta t$ for stability vs. performance

## Key Physical Constants and Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| `POTENTIAL_STRENGTH` | $V_0$ | 0.6 | Nuclear attraction strength |
| `NUCLEAR_REPULSION_STRENGTH` | $A$ | 0.5 | Quantum repulsion magnitude |
| `NUCLEAR_CORE_RADIUS` | $r_0$ | 2.0 | Nuclear size parameter |
| `COULOMB_STRENGTH` | $k_c$ | 2.0 | Inter-nuclear Coulomb strength |
| `STRONG_FORCE_STRENGTH` | $A_s$ | 0.1 | Nuclear binding strength |
| `STRONG_FORCE_RANGE` | $r_s$ | 3.0 | Nuclear force range |
| `TIME_DELTA` | $\Delta t$ | 4 | Time evolution step |
| `ELECTRON_REPULSION_STRENGTH` | $\alpha$ | 0.08 | Mean-field coupling |

## Applications and Simulations

### 1. Hydrogen Atom (`hydrogen_atom.py`)
- Single electron in nuclear potential
- Visualization of orbital transitions
- Real-time wavefunction evolution
- Support for all quantum number combinations

### 2. Hydrogen Molecule (`simple-hydrogen-molecule.py`)
- Two-electron, two-nucleus system
- Molecular bond formation dynamics
- Nuclear motion with Born-Oppenheimer approximation
- Electron correlation effects

## 6. Multi-Atom Electron Creation Functions

The simulator provides functions for creating electron wavefunctions for arbitrary atomic systems beyond hydrogen using effective nuclear charge scaling and quantum mechanical principles.

### 6.1 Core Functions

**`create_atom_electron()`**: Creates electrons for any atomic system with proper nuclear charge scaling using **effective nuclear charge scaling** based on Slater's rules:

$$Z_{\text{eff}} = Z \cdot \alpha \quad \text{where } \alpha \approx 0.8$$

The orbital radius scales as: $r_{\text{scaled}} = r_{\text{base}}/Z_{\text{eff}}^{1/2}$

**`create_orbital_electron()`**: Creates hydrogen-like orbitals with quantum number scaling: $r_{\text{eff}} = r_{\text{base}} \cdot n^{1.5}$

### 6.2 Angular Momentum and Node Structure

**Angular momentum**: For $l > 0$, $m \neq 0$: $\psi(\mathbf{r}) = \psi_{\text{radial}}(r) \cdot e^{im\phi}$

**Radial nodes**: For $n > 1$: $\psi(r) \rightarrow \psi(r) \cdot [1 + 0.3 \cos(\pi r (n-1)/r_{\text{eff}})]$

### 6.3 Supported Atomic Systems

| Element | Z | $Z_{\text{eff}}$ | Orbital Scaling | Ground State |
|---------|---|------------------|-----------------|--------------|
| Carbon  | 6 | 4.8             | $0.46 r_{\text{base}}$ | $1s^2 2s^2 2p^2$ |
| Oxygen  | 8 | 6.4             | $0.39 r_{\text{base}}$ | $1s^2 2s^2 2p^4$ |
| Lithium | 3 | 2.4             | $0.65 r_{\text{base}}$ | $1s^2 2s^1$ |

### 6.4 Momentum Initialization

For excited states ($n > 1$), random momentum breaks symmetry: $\psi(\mathbf{r}) \rightarrow \psi(\mathbf{r}) \cdot e^{i\mathbf{p}_{\text{random}} \cdot \mathbf{r}/\hbar}$ where $|\mathbf{p}_{\text{random}}| \sim 0.1/n$.

## Quantum Mechanical Phenomena Demonstrated

1. **Wave-particle duality**: Wavefunction spreading and localization
2. **Tunneling**: Barrier penetration effects
3. **Orbital hybridization**: Formation of molecular orbitals
4. **Electron correlation**: Mean-field treatment of many-body effects
5. **Nuclear dynamics**: Born-Oppenheimer molecular dynamics
6. **Quantum interference**: Wavefunction phase relationships

## Technical Implementation

### Core Modules:

- **`physics.py`**: Core physics engines and propagation algorithms
- **`hydrogen_utils.py`**: Hydrogen orbital generation and quantum numbers
- **`particles.py`**: Wavepacket creation and manipulation
- **`frame_utils.py`**: Numerical filtering and boundary conditions
- **`config.py`**: Physical constants and simulation parameters
- **`video_utils.py`**: Real-time visualization and video output

### Simulation Flow:

1. **Initialization**: Create initial wavefunctions (orbitals or wavepackets)
2. **Force Calculation**: Compute nuclear forces from electron densities
3. **Nuclear Update**: Evolve nuclear positions (classical mechanics)
4. **Potential Construction**: Build total potential energy landscape
5. **Wavefunction Propagation**: Split-step Fourier evolution
6. **Filtering**: Apply boundary conditions and noise reduction
7. **Visualization**: Generate real-time video output

## Performance and Scaling

- **Grid Resolution**: 512×512 default (configurable)
- **Time Steps**: 3000 default iterations
- **Memory Usage**: Optimized streaming to handle large simulations
- **Computational Complexity**: $O(N^2 \log N)$ per time step (FFT-dominated)

## Future Extensions

1. **Spin-orbit coupling**: Add relativistic corrections
2. **External fields**: Electric and magnetic field interactions
3. **Quantum gates**: Implement quantum computing operations
4. **Three-dimensional**: Full 3D wavefunction evolution
5. **GPU acceleration**: CUDA/OpenCL implementations
6. **Machine learning**: Neural network potential energy surfaces

---

This simulator provides a comprehensive platform for exploring quantum mechanical phenomena with state-of-the-art numerical methods, suitable for research, education, and visualization of atomic and molecular quantum dynamics.
