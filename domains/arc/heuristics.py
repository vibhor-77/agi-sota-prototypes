import numpy as np
from core.approximability import Heuristic

class PixelEditDistance(Heuristic):
    """
    Pillar 2: Approximability.
    Evaluating closeness to truth in combinatorial spatial search.
    """
    def evaluate(self, candidate_grid, target_grid) -> float:
        if candidate_grid.arr.shape != target_grid.arr.shape:
            return 1.0 + abs(candidate_grid.arr.size - target_grid.arr.size) / max(candidate_grid.arr.size, 1)
        
        matches = np.sum(candidate_grid.arr == target_grid.arr)
        total = target_grid.arr.size
        return 1.0 - (matches / total) # 0.0 is perfectly identical
