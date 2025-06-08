from config import GRID_WIDTH, GRID_HEIGHT, MAX_GATES_PER_CELL


def apply_spatial_gates_from_patch(psi, gate_patch_map):
    updated = psi.copy()
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if abs(psi[y, x]) > 1e-3:
                for i in range(MAX_GATES_PER_CELL):
                    gate_fn = gate_patch_map[y, x, i]
                    if gate_fn is not None:
                        updated[y, x] = gate_fn(updated[y, x])
    return updated