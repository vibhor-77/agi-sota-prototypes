import numpy as np
import random
from core.feedback import Environment

class Grid:
    def __init__(self, arr):
        self.arr = np.array(arr)
    def __eq__(self, other):
        if self.arr.shape != other.arr.shape: return False
        return np.array_equal(self.arr, other.arr)

class ArcObject:
    def __init__(self, mask, color):
        self.mask = mask
        self.color = color

class BoundingBox:
    def __init__(self, r_min, r_max, c_min, c_max):
        self.r_min, self.r_max = r_min, r_max
        self.c_min, self.c_max = c_min, c_max
    def __eq__(self, other):
        return (self.r_min == other.r_min and self.r_max == other.r_max and 
                self.c_min == other.c_min and self.c_max == other.c_max)

class ARCEnvironment(Environment):
    """
    Pillar 1: Feedback Loop.
    The agent executes synthesized program actions and observes the mutated spatial grid.
    """
    def __init__(self, initial_grid: Grid):
        self.current_grid = initial_grid
        
    def get_observation(self):
        return self.current_grid
        
    def execute_action(self, program_node):
        """Action is executing an AST node operation on the grid."""
        try:
            self.current_grid = program_node.evaluate({'input_grid': self.current_grid})
        except Exception:
            pass # Invalid operations just leave the state unchanged
        return self.current_grid

def generate_2d_arc_task(n_examples=3):
    """
    Task: Find the blue block, crop the grid to its bounding box, rotate 90 degrees, and paint it green.
    """
    examples = []
    for _ in range(n_examples):
        inp = np.zeros((8, 8), dtype=int)
        for _ in range(3):
            r, c = random.randint(0, 7), random.randint(0, 7)
            inp[r, c] = 2 # Red
            
        r_start, c_start = random.randint(1, 4), random.randint(1, 4)
        h, w = random.randint(2, 3), random.randint(1, 2)
        inp[r_start:r_start+h, c_start:c_start+w] = 1 # Blue
        
        blue_mask = (inp == 1)
        if not np.any(blue_mask): continue
            
        rows, cols = np.where(blue_mask)
        cropped = inp[np.min(rows):np.max(rows)+1, np.min(cols):np.max(cols)+1]
        
        rotated = np.rot90(cropped)
        out = np.copy(rotated)
        out[out == 1] = 3
        
        examples.append((Grid(inp), Grid(out)))
    return examples
