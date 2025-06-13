#!/usr/bin/env python3
"""
Benchmark script to compare performance between:
1. Hybrid Molecular Simulation (individual electron wavefunctions)
2. Unified Molecular Simulation (combined electron wavefunction)
"""

import time
import torch
import numpy as np
from typing import Dict, List

# Import both simulation approaches
from hybrid_molecular_simulation import (
    HybridMolecularSimulation, MolecularElectron, MolecularNucleus, 
    create_atom_electron, X, Y
)
from unified_hybrid_molecular_simulation import (
    UnifiedHybridMolecularSimulation, ElectronInfo
)
from config import SIZE, SCALE, center_x, center_y


class BenchmarkTimer:
    """Simple timer context manager for benchmarking."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def create_hybrid_simulation() -> HybridMolecularSimulation:
    """Create hybrid simulation for benchmarking with carbon atoms (6 electrons each)."""
    bond_length = 1.5 * SCALE  # Slightly longer for carbon-carbon bond
    nucleus1_x = center_x - bond_length/2
    nucleus2_x = center_x + bond_length/2
    nucleus_y = center_y
    
    nuclei = [
        MolecularNucleus(nucleus1_x, nucleus_y, atomic_number=6, atom_id=0),  # Carbon
        MolecularNucleus(nucleus2_x, nucleus_y, atomic_number=6, atom_id=1)   # Carbon
    ]
    
    electrons = []
    
    # Create 6 electrons for first carbon atom (1s, 2s, 2p orbitals)
    print("Creating 6 electrons for carbon atom 1...")
    electrons.extend([
        # 1s electrons (2 electrons)
        MolecularElectron(create_atom_electron(X, Y, nucleus1_x, nucleus_y, (1, 0, 0), 
                         atomic_number=6, scale=SCALE/10), atom_id=0, electron_name="C1_1s1"),
        MolecularElectron(create_atom_electron(X, Y, nucleus1_x, nucleus_y, (1, 0, 0), 
                         atomic_number=6, scale=SCALE/10), atom_id=0, electron_name="C1_1s2"),
        # 2s electrons (2 electrons)
        MolecularElectron(create_atom_electron(X, Y, nucleus1_x, nucleus_y, (2, 0, 0), 
                         atomic_number=6, scale=SCALE/10), atom_id=0, electron_name="C1_2s1"),
        MolecularElectron(create_atom_electron(X, Y, nucleus1_x, nucleus_y, (2, 0, 0), 
                         atomic_number=6, scale=SCALE/10), atom_id=0, electron_name="C1_2s2"),
        # 2p electrons (2 electrons in different orbitals)
        MolecularElectron(create_atom_electron(X, Y, nucleus1_x, nucleus_y, (2, 1, 0), 
                         atomic_number=6, scale=SCALE/10), atom_id=0, electron_name="C1_2px"),
        MolecularElectron(create_atom_electron(X, Y, nucleus1_x, nucleus_y, (2, 1, 1), 
                         atomic_number=6, scale=SCALE/10), atom_id=0, electron_name="C1_2py"),
    ])
    
    # Create 6 electrons for second carbon atom
    print("Creating 6 electrons for carbon atom 2...")
    electrons.extend([
        # 1s electrons (2 electrons)
        MolecularElectron(create_atom_electron(X, Y, nucleus2_x, nucleus_y, (1, 0, 0), 
                         atomic_number=6, scale=SCALE/10), atom_id=1, electron_name="C2_1s1"),
        MolecularElectron(create_atom_electron(X, Y, nucleus2_x, nucleus_y, (1, 0, 0), 
                         atomic_number=6, scale=SCALE/10), atom_id=1, electron_name="C2_1s2"),
        # 2s electrons (2 electrons)
        MolecularElectron(create_atom_electron(X, Y, nucleus2_x, nucleus_y, (2, 0, 0), 
                         atomic_number=6, scale=SCALE/10), atom_id=1, electron_name="C2_2s1"),
        MolecularElectron(create_atom_electron(X, Y, nucleus2_x, nucleus_y, (2, 0, 0), 
                         atomic_number=6, scale=SCALE/10), atom_id=1, electron_name="C2_2s2"),
        # 2p electrons (2 electrons in different orbitals)
        MolecularElectron(create_atom_electron(X, Y, nucleus2_x, nucleus_y, (2, 1, 0), 
                         atomic_number=6, scale=SCALE/10), atom_id=1, electron_name="C2_2px"),
        MolecularElectron(create_atom_electron(X, Y, nucleus2_x, nucleus_y, (2, 1, 1), 
                         atomic_number=6, scale=SCALE/10), atom_id=1, electron_name="C2_2py"),
    ])
    
    return HybridMolecularSimulation(
        nuclei=nuclei,
        electrons=electrons,  # 12 electrons total (6 per carbon atom)
        electron_repulsion_strength=0.08,  # Reduced for many-electron system
        nuclear_motion_enabled=True,
        damping_factor=0.999
    )


def create_unified_simulation() -> UnifiedHybridMolecularSimulation:
    """Create unified simulation for benchmarking with carbon atoms (all 12 electrons unified)."""
    bond_length = 1.5 * SCALE  # Same as hybrid simulation
    nucleus1_x = center_x - bond_length/2
    nucleus2_x = center_x + bond_length/2
    nucleus_y = center_y
    
    nuclei = [
        MolecularNucleus(nucleus1_x, nucleus_y, atomic_number=6, atom_id=0),  # Carbon
        MolecularNucleus(nucleus2_x, nucleus_y, atomic_number=6, atom_id=1)   # Carbon
    ]
    
    # Create all individual electron wavefunctions first
    print("Creating 12 individual electron wavefunctions for unification...")
    electron_wavefunctions = []
    
    # Carbon 1 electrons
    electron_wavefunctions.extend([
        create_atom_electron(X, Y, nucleus1_x, nucleus_y, (1, 0, 0), atomic_number=6, scale=SCALE/10),  # 1s1
        create_atom_electron(X, Y, nucleus1_x, nucleus_y, (1, 0, 0), atomic_number=6, scale=SCALE/10),  # 1s2
        create_atom_electron(X, Y, nucleus1_x, nucleus_y, (2, 0, 0), atomic_number=6, scale=SCALE/10),  # 2s1
        create_atom_electron(X, Y, nucleus1_x, nucleus_y, (2, 0, 0), atomic_number=6, scale=SCALE/10),  # 2s2
        create_atom_electron(X, Y, nucleus1_x, nucleus_y, (2, 1, 0), atomic_number=6, scale=SCALE/10),  # 2px
        create_atom_electron(X, Y, nucleus1_x, nucleus_y, (2, 1, 1), atomic_number=6, scale=SCALE/10),  # 2py
    ])
    
    # Carbon 2 electrons
    electron_wavefunctions.extend([
        create_atom_electron(X, Y, nucleus2_x, nucleus_y, (1, 0, 0), atomic_number=6, scale=SCALE/10),  # 1s1
        create_atom_electron(X, Y, nucleus2_x, nucleus_y, (1, 0, 0), atomic_number=6, scale=SCALE/10),  # 1s2
        create_atom_electron(X, Y, nucleus2_x, nucleus_y, (2, 0, 0), atomic_number=6, scale=SCALE/10),  # 2s1
        create_atom_electron(X, Y, nucleus2_x, nucleus_y, (2, 0, 0), atomic_number=6, scale=SCALE/10),  # 2s2
        create_atom_electron(X, Y, nucleus2_x, nucleus_y, (2, 1, 0), atomic_number=6, scale=SCALE/10),  # 2px
        create_atom_electron(X, Y, nucleus2_x, nucleus_y, (2, 1, 1), atomic_number=6, scale=SCALE/10),  # 2py
    ])
    
    # Create unified wavefunction by summing all 12 electrons
    print("Unifying 12 electron wavefunctions...")
    unified_psi = torch.zeros_like(X, dtype=torch.complex64)
    for psi in electron_wavefunctions:
        unified_psi += psi
    
    # Normalize
    unified_psi = unified_psi / torch.sqrt(torch.tensor(12.0))  # 12 electrons
    norm = torch.sqrt(torch.sum(torch.abs(unified_psi)**2))
    if norm > 0:
        unified_psi = unified_psi / norm
    
    # Create electron info objects for all 12 electrons
    electron_infos = []
    for atom_id in [0, 1]:  # Two carbon atoms
        for i in range(6):  # 6 electrons per carbon
            electron_infos.append(ElectronInfo(atom_id=atom_id, electron_name=f"C{atom_id+1}_e{i+1}"))
    
    return UnifiedHybridMolecularSimulation(
        nuclei=nuclei,
        electron_infos=electron_infos,  # 12 electrons total (unified)
        unified_wavefunction=unified_psi,
        electron_repulsion_strength=0.08,  # Same as hybrid
        nuclear_motion_enabled=True,
        damping_factor=0.999
    )


def benchmark_simulation(simulation, name: str, num_steps: int = 100) -> Dict:
    """Benchmark a simulation for specified number of steps."""
    print(f"\n=== Benchmarking {name} ===")
    
    # Warm up GPU/CPU
    for _ in range(5):
        simulation.evolve_step(0)
    
    # Individual step timing
    step_times = []
    
    # Total evolution timing
    with BenchmarkTimer(f"{name} Total") as total_timer:
        for step in range(num_steps):
            with BenchmarkTimer(f"{name} Step {step}") as step_timer:
                simulation.evolve_step(step)
            step_times.append(step_timer.elapsed)
            
            if step % 10 == 0:
                print(f"  Step {step:3d}: {step_timer.elapsed*1000:.2f} ms")
    
    # Calculate statistics
    avg_step_time = np.mean(step_times)
    std_step_time = np.std(step_times)
    min_step_time = np.min(step_times)
    max_step_time = np.max(step_times)
    
    results = {
        'name': name,
        'total_time': total_timer.elapsed,
        'avg_step_time': avg_step_time,
        'std_step_time': std_step_time,
        'min_step_time': min_step_time,
        'max_step_time': max_step_time,
        'steps_per_second': num_steps / total_timer.elapsed,
        'num_steps': num_steps
    }
    
    print(f"  Total time: {total_timer.elapsed:.3f} seconds")
    print(f"  Average step: {avg_step_time*1000:.2f} ± {std_step_time*1000:.2f} ms")
    print(f"  Min/Max step: {min_step_time*1000:.2f} / {max_step_time*1000:.2f} ms")
    print(f"  Steps/second: {results['steps_per_second']:.1f}")
    
    return results


def run_benchmark(num_steps: int = 100):
    """Run performance benchmark comparing hybrid vs unified approaches."""
    
    print("=" * 60)
    print("CARBON MOLECULE SIMULATION PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Grid size: {SIZE}x{SIZE}")
    print(f"Number of steps: {num_steps}")
    print(f"PyTorch device: {torch.tensor(0.0).device}")
    print(f"Molecule: C2 (2 carbon atoms, 12 electrons total)")
    print(f"Hybrid approach: 12 individual electron wavefunctions")
    print(f"Unified approach: 1 combined wavefunction representing all 12 electrons")
    
    # Create simulations
    print("\nCreating simulations...")
    hybrid_sim = create_hybrid_simulation()
    unified_sim = create_unified_simulation()
    
    # Run benchmarks
    hybrid_results = benchmark_simulation(hybrid_sim, "Hybrid", num_steps)
    unified_results = benchmark_simulation(unified_sim, "Unified", num_steps)
    
    # Compare results
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    hybrid_total = hybrid_results['total_time']
    unified_total = unified_results['total_time']
    speedup = hybrid_total / unified_total
    
    print(f"Hybrid simulation:   {hybrid_total:.3f}s ({hybrid_results['steps_per_second']:.1f} steps/s)")
    print(f"Unified simulation:  {unified_total:.3f}s ({unified_results['steps_per_second']:.1f} steps/s)")
    print(f"Speedup factor:      {speedup:.2f}x {'(Unified faster)' if speedup > 1 else '(Hybrid faster)'}")
    
    hybrid_avg = hybrid_results['avg_step_time'] * 1000
    unified_avg = unified_results['avg_step_time'] * 1000
    step_speedup = hybrid_avg / unified_avg
    
    print(f"\nPer-step timing:")
    print(f"Hybrid average:      {hybrid_avg:.2f} ms/step")
    print(f"Unified average:     {unified_avg:.2f} ms/step")
    print(f"Step speedup:        {step_speedup:.2f}x {'(Unified faster)' if step_speedup > 1 else '(Hybrid faster)'}")
    
    # Memory usage comparison (rough estimate)
    print(f"\nMemory usage (rough estimate):")
    wavefunction_size = SIZE * SIZE * 8 * 2  # complex64 = 8 bytes, 2 for real+imag
    hybrid_memory = 12 * wavefunction_size  # 12 separate electron wavefunctions
    unified_memory = 1 * wavefunction_size   # 1 unified wavefunction
    memory_savings = (hybrid_memory - unified_memory) / hybrid_memory * 100
    
    print(f"Hybrid memory:       ~{hybrid_memory / (1024**2):.1f} MB (12 wavefunctions)")
    print(f"Unified memory:      ~{unified_memory / (1024**2):.1f} MB (1 wavefunction)")
    print(f"Memory savings:      {memory_savings:.1f}%")
    
    # Theoretical analysis
    print(f"\nTheoretical analysis:")
    print(f"- Hybrid: 12 FFT calls per step (one per electron)")
    print(f"- Unified: 1 FFT call per step (combined wavefunction)")
    print(f"- Expected speedup: ~12x (if FFT dominates computation)")
    print(f"- Actual speedup: {speedup:.2f}x")
    
    if speedup < 8.0:
        print(f"- Much lower than expected speedup suggests significant other bottlenecks")
    elif speedup > 15.0:
        print(f"- Higher than expected speedup suggests additional optimizations")
    else:
        print(f"- Speedup roughly matches theoretical expectation")
    
    return {
        'hybrid': hybrid_results,
        'unified': unified_results,
        'speedup': speedup,
        'step_speedup': step_speedup
    }


if __name__ == "__main__":
    # Run benchmark with different step counts
    print("Running molecular simulation benchmark...")
    
    # Quick benchmark
    results_quick = run_benchmark(num_steps=50)
    
    # Longer benchmark for more accurate timing
    print("\n" + "="*60)
    print("CARBON MOLECULE EXTENDED BENCHMARK")
    print("="*60)
    results_extended = run_benchmark(num_steps=200)
    
    print("\n" + "="*60)
    print("CARBON MOLECULE BENCHMARK SUMMARY")
    print("="*60)
    print(f"Quick test (50 steps):    {results_quick['speedup']:.2f}x speedup")
    print(f"Extended test (200 steps): {results_extended['speedup']:.2f}x speedup")
    print(f"Average speedup:          {(results_quick['speedup'] + results_extended['speedup'])/2:.2f}x")
    
    if results_extended['speedup'] > 6.0:
        print("\n✅ Unified approach shows MASSIVE performance improvement!")
        print("   Recommendation: Use unified simulation for dramatic speedup with many electrons")
    elif results_extended['speedup'] > 3.0:
        print("\n✅ Unified approach shows significant performance improvement!")
        print("   Recommendation: Use unified simulation for better performance")
    elif results_extended['speedup'] > 1.5:
        print("\n✅ Unified approach shows moderate performance improvement")
        print("   Recommendation: Use unified simulation for efficiency")
    else:
        print("\n⚠️  Lower than expected performance improvement")
        print("   Analysis: Other bottlenecks may be limiting the speedup")
