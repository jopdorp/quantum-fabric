# Wave-Based Computational Architecture: Approach Summary

## Core Concept
A self-modifying, signal-driven computational fabric where:
- **Signal = Data = Logic**: The wavefront carries both computation and reconfiguration
- **Flexible Timing**: Can operate asynchronously (signal-driven) or with optional clocking for FPGA compatibility
- **Spatial Computing**: Logic cells rewrite themselves and their neighbors based on signal patterns
- **Biological Inspiration**: Mimics protein folding, DNA computation, and cellular processes

## Key Innovations

### 1. Computational Wave
- Each logic cell processes data AND rewrites downstream logic
- Wave propagates configuration changes alongside computation
- Creates an "infinite pipeline" through spatial reconfiguration
- Self-sustaining computation that adapts to input patterns

### 2. Signal-Driven Reconfiguration
- Strong signals trigger local logic mutations
- Pattern recognition enables predictive reconfiguration
- Feedback loops allow learning and optimization
- Environmental triggers (like pH, temperature) can alter behavior

### 3. Applications

#### Protein Folding Simulation
- Each amino acid = logic cell with phi/psi angles
- Folding wave propagates torsion decisions
- Signal-triggered reconfiguration simulates binding events
- Real-time conformational changes through logic evolution

#### Cryptographic Factorization
- Wave-based modular exponentiation
- Period detection through signal interference patterns
- Potential polynomial-time scaling for large numbers
- Eliminates instruction fetch/decode overhead

## Mathematical Foundation

### Scaling Analysis
- **Classical CPU**: O(log³N) per modular exponentiation
- **Quantum (Shor)**: O(log³N) with QFT periodicity detection
- **Wave Architecture**: O(log²N) with spatial parallelism and signal-based period detection

### Periodicity Detection
For factoring N = pq, find order r where aʳ ≡ 1 (mod N):
- Multiple wavefronts run parallel trials
- Signal interference patterns reveal period r
- Physical alignment of waves indicates mathematical cycles
- No exponential trial search needed

### Resource Requirements
- **Logic**: O(B) stages where B = log₂N
- **Time**: O(B) per pipeline depth, O(B²) total
- **Space**: Fixed wavefront width, reuses logic spatially

## Implementation Advantages

1. **No Instruction Overhead**: Logic IS the program
2. **Massive Parallelism**: Spatial unrolling across fabric
3. **Adaptive Optimization**: Self-modifying based on data patterns
4. **Physical Efficiency**: Computation at signal propagation speed
5. **Biological Fidelity**: Mirrors natural processes like folding

## Challenges & Limitations

1. **FPGA Constraints**: Resource overhead of the overlay fabric and the complexity of designing efficient, rapidly switchable overlay configurations.
2. **Timing Closure**: Asynchronous design complexity
3. **Verification**: Hard to test self-modifying logic
4. **Tool Support**: Standard EDA tools assume fixed logic

## Optional Clocking

While the wave-based computational architecture is inherently asynchronous, optional clocking can be integrated to:

1. **Simplify Implementation**:
   - Clocked designs are easier to synthesize and debug using standard FPGA toolchains.
   - Timing closure becomes more predictable.

2. **Enhance Compatibility**:
   - Clocking allows integration with existing synchronous systems.
   - Enables the use of standard modules like BRAMs, DSPs, and FIFOs.

3. **Maintain Wave Behavior**:
   - The computational wave can still propagate logically, with each clock cycle representing a discrete step.
   - Signal-driven reconfiguration and feedback loops remain intact.

### Clocked vs Asynchronous Comparison

| Feature                  | Asynchronous Wave         | Clocked Overlay            |
|--------------------------|---------------------------|----------------------------|
| Signal Propagation       | Continuous, event-driven | Discrete, clock-driven     |
| Timing Closure           | Complex                  | Predictable                |
| Debugging                | Harder                   | Easier                     |
| Integration              | Requires custom logic    | Compatible with standard tools |

### Implementation Notes

- **Clocked Mode**:
  - Each wavefront step corresponds to a clock cycle.
  - Logic cells latch data and configuration on rising edges.

- **Asynchronous Mode**:
  - Signal propagation triggers computation and reconfiguration dynamically.
  - No global timing constraints.

This flexibility allows the architecture to adapt to different hardware environments and design goals.

## Achieving Dynamic Behavior with Overlays

**Dynamic Kernel Swapping**:
- Overlays allow different pre-compiled computational kernels to be rapidly loaded onto a flexible, pre-defined hardware fabric at runtime.
- This enables runtime adaptability and efficient resource utilization without traditional full or partial bitstream reprogramming.

**Bitstream Caching**:
- Pre-compile common overlay kernels/configurations and load them dynamically based on usage patterns.
- Reduces reconfiguration latency and improves performance.

**Runtime Schedulers**:
- Implement schedulers to manage overlay kernel swaps and optimize memory hierarchy.
- Dynamically prioritize hot entries for BRAM and cold entries for DDR storage.

### Open Source Frameworks for Implementation

**LiteX + Migen**:
- An open-source SoC builder using Python for softcore CPUs and coprocessors.
- Useful for creating the base system and integrating custom overlay fabrics or the logic blocks that run on them.

**Dynamatic**:
- An open-source high-level synthesis (HLS) compiler that translates LLVM IR to Verilog.
- Capable of generating reconfigurable hardware modules from higher-level descriptions, which can then be targeted to an overlay.

### Conceptual Similarities

**JIT Compilation for CPUs**:
- The wave-based architecture mirrors JIT compilation by dynamically synthesizing and loading hardware accelerators.

**TensorRT/XLA for GPUs**:
- Similar to runtime kernel optimization for GPUs, the wave-based approach adapts hardware logic based on workload patterns.

**eBPF JITs**:
- Analogous to how eBPF dynamically and safely extends kernel capabilities with efficient, JIT-compiled bytecode, the wave architecture aims to dynamically accelerate operations at the hardware level.

## Potential Impact

- **Cryptography**: Could challenge RSA security assumptions
- **Biology**: Real-time protein folding prediction
- **AI**: Hardware that learns and adapts like neural tissue
- **Physics**: Bridge between classical and quantum computation

---

This architecture represents a fundamental shift from von Neumann computing toward spatial, signal-driven computation that embodies rather than simulates natural processes.

## Overlay Concept

**Definition**:
An overlay is a virtual hardware layer implemented on top of the FPGA fabric. It abstracts the underlying hardware and provides a flexible, reusable framework for dynamic computation.

**Purpose**:
- Simplifies the implementation of wave-based architectures by predefining logic templates.
- Reduces the need for full bitstream reprogramming, enabling faster reconfiguration.
- Acts as a runtime-adaptive layer that can dynamically load and execute computational kernels.

**Advantages**:
- **Flexibility**: Allows dynamic adaptation to changing workloads without disrupting ongoing operations.
- **Efficiency**: Minimizes reconfiguration latency by using precompiled templates.

**Examples and Variants**:
- Overlays can range from simple template-based systems to more complex dynamically composed soft Coarse-Grained Reconfigurable Architectures (CGRAs).
- Projects like QUICKDough demonstrate caching and reuse of kernel configurations via overlays, while others explore reconfigurable Arithmetic Logic Units (ALUs) for common operations, aligning with the adaptive nature of the wave computing paradigm.
