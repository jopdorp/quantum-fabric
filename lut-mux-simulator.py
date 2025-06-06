import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Fabric Parameters ---
NUM_CELLS = 32  # Total number of logic cells
STEPS = 20      # Number of simulation steps

# --- Define logic functions ---
def nand(a, b):
    return ~(a & b) & 1

# --- Each cell stores its output and logic function ---
class LogicCell:
    def __init__(self, id, lut=nand):
        self.id = id
        self.lut = lut
        self.inputs = (0, 0)
        self.output = 0
        self.next_output = 0

    def compute(self):
        a, b = self.inputs
        self.next_output = self.lut(a, b)

    def update(self):
        self.output = self.next_output

# --- Build fabric of cells ---
cells = [LogicCell(i) for i in range(NUM_CELLS)]

# --- Define sparse routing graph ---
# This maps source cell ID -> list of destination cell ID and input index (0 or 1)
routing_graph = defaultdict(list)

# Example: long-distance, irregular wiring
np.random.seed(42)
for src in range(NUM_CELLS):
    for _ in range(2):  # each cell drives two others
        dst = np.random.randint(0, NUM_CELLS)
        inp_idx = np.random.choice([0, 1])
        routing_graph[src].append((dst, inp_idx))

# --- Initialize a wavefront (seed pulse) ---
cells[0].output = 1  # seed signal in first cell

# --- Simulation ---
history = np.zeros((STEPS, NUM_CELLS), dtype=int)

for t in range(STEPS):
    # Route signals: reset inputs
    for cell in cells:
        cell.inputs = (0, 0)
    
    # Propagate outputs through the routing graph
    for src_id, targets in routing_graph.items():
        for dst_id, input_idx in targets:
            current_input = list(cells[dst_id].inputs)
            current_input[input_idx] = cells[src_id].output
            cells[dst_id].inputs = tuple(current_input)

    # Compute and update
    for cell in cells:
        cell.compute()
    for cell in cells:
        cell.update()
        history[t, cell.id] = cell.output

# --- Plot activity over time ---
plt.figure(figsize=(10, 5))
plt.imshow(history.T, cmap="Greys", aspect="auto", interpolation="nearest")
plt.title("Signal Propagation Over Time (Sparse Global Routing)")
plt.xlabel("Time Step")
plt.ylabel("Logic Cell ID")
plt.colorbar(label="Signal (0 or 1)")
plt.show()