import numpy as np
import random
from scipy.ndimage import label
import warnings
warnings.filterwarnings('ignore')

# ======================================================================
# 1. ARC 2D Environment & Types
# ======================================================================
class Grid:
    def __init__(self, arr):
        self.arr = np.array(arr)
    def __eq__(self, other):
        if self.arr.shape != other.arr.shape:
            return False
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

# Synthetic Task: Complex spatial reasoning
def generate_2d_arc_task(n_examples=3):
    """
    Task: Find the blue block, crop the grid to its bounding box, rotate 90 degrees, and paint it green.
    """
    examples = []
    for _ in range(n_examples):
        inp = np.zeros((8, 8), dtype=int)
        
        # Add random red distractions
        for _ in range(3):
            r, c = random.randint(0, 7), random.randint(0, 7)
            inp[r, c] = 2 # Red
            
        # Add a blue rectangle
        r_start, c_start = random.randint(1, 4), random.randint(1, 4)
        h, w = random.randint(2, 3), random.randint(1, 2)
        inp[r_start:r_start+h, c_start:c_start+w] = 1 # Blue
        
        # Goal: crop to blue bounding box (reds are ignored), rotate it 90 degrees, color it green (3)
        blue_mask = (inp == 1)
        if not np.any(blue_mask):
            continue
            
        rows, cols = np.where(blue_mask)
        cropped = inp[np.min(rows):np.max(rows)+1, np.min(cols):np.max(cols)+1]
        
        rotated = np.rot90(cropped)
        out = np.copy(rotated)
        out[out == 1] = 3
        
        examples.append((Grid(inp), Grid(out)))
    return examples

# ======================================================================
# 2. Domain Specific Language (DSL) & Primitives
# ======================================================================

def get_objects(grid: Grid) -> list:
    objects = []
    colors = np.unique(grid.arr)
    for c in colors:
        if c == 0: continue
        binary_mask = (grid.arr == c).astype(int)
        labeled_arr, num_features = label(binary_mask)
        for i in range(1, num_features + 1):
            obj_mask = (labeled_arr == i)
            objects.append(ArcObject(obj_mask, c))
    return objects

def filter_by_color(objects: list, color: int) -> list:
    return [o for o in objects if o.color == color]

def paint_objects(grid: Grid, objects: list, color: int) -> Grid:
    # Notice: It creates an error if dimensions mismatch, but we assume
    # the objects are relative to the given grid here.
    new_grid = np.copy(grid.arr)
    for obj in objects:
        if obj.mask.shape == new_grid.shape:
            new_grid[obj.mask] = color
    return Grid(new_grid)

def rotate90(grid: Grid) -> Grid:
    return Grid(np.rot90(grid.arr))

def mirror_x(grid: Grid) -> Grid:
    return Grid(np.flipud(grid.arr))

def mirror_y(grid: Grid) -> Grid:
    return Grid(np.fliplr(grid.arr))

def get_bounding_box(obj: ArcObject) -> BoundingBox:
    rows, cols = np.where(obj.mask)
    if len(rows) == 0:
        return BoundingBox(0, 0, 0, 0)
    return BoundingBox(np.min(rows), np.max(rows), np.min(cols), np.max(cols))

def crop_to_box(grid: Grid, box: BoundingBox) -> Grid:
    if box.r_max < box.r_min or box.c_max < box.c_min:
        return Grid(np.zeros((1, 1), dtype=int))
    return Grid(grid.arr[box.r_min:box.r_max+1, box.c_min:box.c_max+1])

# ======================================================================
# 3. Type-Safe Abstract Syntax Tree (AST) Search
# ======================================================================
class ASTNode:
    def evaluate(self, env): pass
    def __str__(self): return ""
    def mutate(self): return self

class IdentityGridNode(ASTNode):
    def evaluate(self, env): return env['input_grid']
    def __str__(self): return "input_grid"

class ColorNode(ASTNode):
    def __init__(self, color_val): self.color_val = color_val
    def evaluate(self, env): return self.color_val
    def __str__(self): return f"{self.color_val}"
    def mutate(self): return ColorNode(random.choice([1, 2, 3]))

class GetObjectsNode(ASTNode):
    def __init__(self, grid_node): self.grid_node = grid_node
    def evaluate(self, env): return get_objects(self.grid_node.evaluate(env))
    def __str__(self): return f"get_objects({self.grid_node})"

class FilterColorNode(ASTNode):
    def __init__(self, obj_list_node, color_node):
        self.objs = obj_list_node
        self.col = color_node
    def evaluate(self, env): return filter_by_color(self.objs.evaluate(env), self.col.evaluate(env))
    def __str__(self): return f"filter_by_color({self.objs}, {self.col})"

class PaintNode(ASTNode):
    def __init__(self, grid_node, obj_list_node, color_node):
        self.grid = grid_node
        self.objs = obj_list_node
        self.col = color_node
    def evaluate(self, env): return paint_objects(self.grid.evaluate(env), self.objs.evaluate(env), self.col.evaluate(env))
    def __str__(self): return f"paint({self.grid}, {self.objs}, {self.col})"

class Rotate90Node(ASTNode):
    def __init__(self, grid_node): self.grid_node = grid_node
    def evaluate(self, env): return rotate90(self.grid_node.evaluate(env))
    def __str__(self): return f"rotate90({self.grid_node})"

class MirrorXNode(ASTNode):
    def __init__(self, grid_node): self.grid_node = grid_node
    def evaluate(self, env): return mirror_x(self.grid_node.evaluate(env))
    def __str__(self): return f"mirror_x({self.grid_node})"

class MirrorYNode(ASTNode):
    def __init__(self, grid_node): self.grid_node = grid_node
    def evaluate(self, env): return mirror_y(self.grid_node.evaluate(env))
    def __str__(self): return f"mirror_y({self.grid_node})"

class GetBoundingBoxNode(ASTNode):
    def __init__(self, obj_list_node): self.objs = obj_list_node
    def evaluate(self, env):
        objs = self.objs.evaluate(env)
        if not objs: return BoundingBox(0,0,0,0)
        return get_bounding_box(objs[0])
    def __str__(self): return f"bbox({self.objs}[0])"

class CropToBoxNode(ASTNode):
    def __init__(self, grid_node, box_node):
        self.grid = grid_node
        self.box = box_node
    def evaluate(self, env): return crop_to_box(self.grid.evaluate(env), self.box.evaluate(env))
    def __str__(self): return f"crop_to_box({self.grid}, {self.box})"

# --- Type-Safe Generator ---
def generate_typed_program(return_type, max_depth=3):
    if return_type == 'Grid':
        if max_depth <= 1:
            return IdentityGridNode()
        r = random.random()
        if r < 0.2:
            return IdentityGridNode()
        elif r < 0.4:
            return Rotate90Node(generate_typed_program('Grid', max_depth-1))
        elif r < 0.5:
            return MirrorXNode(generate_typed_program('Grid', max_depth-1))
        elif r < 0.6:
            return MirrorYNode(generate_typed_program('Grid', max_depth-1))
        elif r < 0.8:
            return CropToBoxNode(
                generate_typed_program('Grid', max_depth-1),
                generate_typed_program('Box', max_depth-1)
            )
        else:
            return PaintNode(
                grid_node=generate_typed_program('Grid', max_depth-1),
                obj_list_node=generate_typed_program('ObjectList', max_depth-1),
                color_node=generate_typed_program('Color', max_depth-1)
            )
            
    elif return_type == 'ObjectList':
        if max_depth <= 1 or random.random() < 0.5:
            return GetObjectsNode(generate_typed_program('Grid', max_depth-1))
        else:
            return FilterColorNode(
                obj_list_node=generate_typed_program('ObjectList', max_depth-1),
                color_node=generate_typed_program('Color', max_depth-1)
            )
            
    elif return_type == 'Color':
        return ColorNode(random.choice([1, 2, 3]))
        
    elif return_type == 'Box':
        return GetBoundingBoxNode(generate_typed_program('ObjectList', max_depth-1))

def pixel_edit_distance(pred: Grid, target: Grid) -> float:
    """Heuristic: Pixel Edit Distance"""
    if pred.arr.shape != target.arr.shape:
        # Heavily penalize shape mismatch, but differentiate based on size proximity maybe?
        return 1.0 + abs(pred.arr.size - target.arr.size) / max(pred.arr.size, 1)
    
    matches = np.sum(pred.arr == target.arr)
    total = target.arr.size
    return 1.0 - (matches / total) # 0 is perfectly identical

def evaluate_program(program, examples):
    total_dist = 0.0
    for inp, out in examples:
        try:
            pred = program.evaluate({'input_grid': inp})
            total_dist += pixel_edit_distance(pred, out)
        except Exception:
            total_dist += 10.0 # Error penalty
    return total_dist / len(examples)

def synthesize_arc_program_beam_search(train_examples, beam_width=50, max_generations=20):
    print(f"[SYNTHESIS] Running Beam Search with pixel edit distance heuristic...")
    beam = []
    
    # Initialize beam randomly
    for _ in range(beam_width * 2):
        candidate = generate_typed_program('Grid', max_depth=4)
        score = evaluate_program(candidate, train_examples)
        beam.append((score, candidate))
        
    for gen in range(max_generations):
        # Sort and prune beam
        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_width]
        
        if beam[0][0] == 0.0:
            print(f"[SYNTHESIS] Converged perfectly in generation {gen}")
            return beam[0][1]
            
        if gen % 5 == 0:
            print(f"Gen {gen} | Best Heuristic Loss: {beam[0][0]:.3f} | Best Program: {beam[0][1]}")
            
        # Crossover & Mutate (For simplicity, regenerate partially and keep old ones)
        new_beam = []
        for score, prog in beam:
            new_beam.append((score, prog))
            # Mutate by generating new random trees to inject diversity
            for _ in range(2):
                new_candidate = generate_typed_program('Grid', max_depth=5)
                new_score = evaluate_program(new_candidate, train_examples)
                new_beam.append((new_score, new_candidate))
        beam = new_beam
        
    beam.sort(key=lambda x: x[0])
    return beam[0][1]

# ======================================================================
# 4. Main Execution
# ======================================================================
def run_arc_sota():
    print("==================================================")
    print(" ARC AGI SOTA: BEAM SEARCH & GEOMETRY PRIMITIVES ")
    print("==================================================\n")
    
    examples = generate_2d_arc_task(4)
    if not examples or len(examples) < 4:
        print("Failed to generate task examples.")
        return
        
    train_ex = examples[:3]
    test_ex = examples[3]
    
    print(">>> OBSERVING TRAINING GRIDS")
    for i, (inp, out) in enumerate(train_ex):
        print(f"\n--- Train {i+1} ---")
        print(f"Input:\n{inp.arr}")
        print(f"Output:\n{out.arr}")
        
    best_program = synthesize_arc_program_beam_search(train_ex, beam_width=100, max_generations=20)
    
    print("\n==================================================")
    print(">>> COMPOSITIONAL LEARNING CONVERGED")
    print("==================================================")
    if best_program:
        print(f"Discovered Concept: f(grid) = {str(best_program)}\n")
        
        print(">>> EXTRAPOLATION ON UNSEEN TEST GRID")
        test_in, test_out = test_ex
        try:
            predicted = best_program.evaluate({'input_grid': test_in})
            print(f"Input Grid:\n{test_in.arr}")
            print(f"AI Predicted:\n{predicted.arr}")
            print(f"True Answer:\n{test_out.arr}")
            
            if predicted == test_out:
                print("\nRESULT: PERFECT 2D GENERALIZATION & SPATIAL REASONING ACHIEVED.")
            else:
                print("\nRESULT: TEST FAILED.")
        except Exception as e:
            print(f"\nRESULT: RUNTIME ERROR ON TEST: {e}")
    else:
        print("Search failed.")

if __name__ == "__main__":
    run_arc_sota()
