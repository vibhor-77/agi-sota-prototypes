import numpy as np
import random
from scipy.ndimage import label
from core.abstraction import ActionGrammar
from domains.arc.env import Grid, ArcObject, BoundingBox

# --- Primitives ---
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
    new_grid = np.copy(grid.arr)
    for obj in objects:
        if obj.mask.shape == new_grid.shape:
            new_grid[obj.mask] = color
    return Grid(new_grid)

def rotate90(grid: Grid) -> Grid:
    return Grid(np.rot90(grid.arr))

def flip_y(grid: Grid) -> Grid:
    return Grid(np.flipud(grid.arr))

def flip_x(grid: Grid) -> Grid:
    return Grid(np.fliplr(grid.arr))

def get_bounding_box(obj: ArcObject) -> BoundingBox:
    rows, cols = np.where(obj.mask)
    if len(rows) == 0: return BoundingBox(0, 0, 0, 0)
    return BoundingBox(np.min(rows), np.max(rows), np.min(cols), np.max(cols))

def crop_to_box(grid: Grid, box: BoundingBox) -> Grid:
    if box.r_max < box.r_min or box.c_max < box.c_min:
        return Grid(np.zeros((1, 1), dtype=int))
    return Grid(grid.arr[box.r_min:box.r_max+1, box.c_min:box.c_max+1])

def replace_color(grid: Grid, from_color: int, to_color: int) -> Grid:
    new_grid = np.copy(grid.arr)
    new_grid[new_grid == from_color] = to_color
    return Grid(new_grid)

def pad(grid: Grid, pad_width: int, color: int) -> Grid:
    arr = grid.arr
    return Grid(np.pad(arr, pad_width=pad_width, mode='constant', constant_values=color))

def fill_box(grid: Grid, box: BoundingBox, color: int) -> Grid:
    if box.r_max < box.r_min or box.c_max < box.c_min:
        return grid
    new_grid = np.copy(grid.arr)
    new_grid[box.r_min:box.r_max+1, box.c_min:box.c_max+1] = color
    return Grid(new_grid)

def transpose(grid: Grid) -> Grid:
    return Grid(grid.arr.T)

def tile(grid: Grid, n_rows: int, n_cols: int) -> Grid:
    return Grid(np.tile(grid.arr, (n_rows, n_cols)))

def overlay(base: Grid, top: Grid) -> Grid:
    """Composite two grids: non-zero pixels from top overwrite base."""
    if base.arr.shape != top.arr.shape:
        # Crop to minimum common shape
        min_r = min(base.arr.shape[0], top.arr.shape[0])
        min_c = min(base.arr.shape[1], top.arr.shape[1])
        b = base.arr[:min_r, :min_c].copy()
        t = top.arr[:min_r, :min_c]
    else:
        b = base.arr.copy()
        t = top.arr
    mask = t != 0
    b[mask] = t[mask]
    return Grid(b)

def count_color(grid: Grid, color: int) -> int:
    """Count pixels of a specific color."""
    return int(np.sum(grid.arr == color))

def most_common_color(grid: Grid) -> int:
    """Return the most common non-zero color in the grid."""
    flat = grid.arr.flatten()
    non_zero = flat[flat != 0]
    if len(non_zero) == 0:
        return 0
    values, counts = np.unique(non_zero, return_counts=True)
    return int(values[np.argmax(counts)])

def scale_up(grid: Grid, factor: int) -> Grid:
    """Scale grid by repeating each pixel factor×factor."""
    factor = max(1, min(factor, 5))  # Clamp to [1, 5]
    return Grid(np.repeat(np.repeat(grid.arr, factor, axis=0), factor, axis=1))

def stack_v(top: Grid, bottom: Grid) -> Grid:
    """Stack two grids vertically. Pads narrower grid with zeros."""
    max_c = max(top.arr.shape[1], bottom.arr.shape[1])
    t = np.pad(top.arr, ((0, 0), (0, max_c - top.arr.shape[1])), constant_values=0)
    b = np.pad(bottom.arr, ((0, 0), (0, max_c - bottom.arr.shape[1])), constant_values=0)
    return Grid(np.vstack([t, b]))

def stack_h(left: Grid, right: Grid) -> Grid:
    """Stack two grids horizontally. Pads shorter grid with zeros."""
    max_r = max(left.arr.shape[0], right.arr.shape[0])
    l = np.pad(left.arr, ((0, max_r - left.arr.shape[0]), (0, 0)), constant_values=0)
    r = np.pad(right.arr, ((0, max_r - right.arr.shape[0]), (0, 0)), constant_values=0)
    return Grid(np.hstack([l, r]))

def largest_object(grid: Grid) -> Grid:
    """Extract the largest connected non-zero component as a cropped grid."""
    objs = get_objects(grid)
    if not objs:
        return grid
    biggest = max(objs, key=lambda o: int(np.sum(o.mask)))
    rows, cols = np.where(biggest.mask)
    if len(rows) == 0:
        return grid
    cropped = grid.arr[np.min(rows):np.max(rows)+1, np.min(cols):np.max(cols)+1].copy()
    return Grid(cropped)

# --- Abstract Syntax Tree (Composability) ---
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

class FlipYNode(ASTNode):
    def __init__(self, grid_node): self.grid_node = grid_node
    def evaluate(self, env): return flip_y(self.grid_node.evaluate(env))
    def __str__(self): return f"flip_y({self.grid_node})"

class FlipXNode(ASTNode):
    def __init__(self, grid_node): self.grid_node = grid_node
    def evaluate(self, env): return flip_x(self.grid_node.evaluate(env))
    def __str__(self): return f"flip_x({self.grid_node})"

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

class ReplaceColorNode(ASTNode):
    def __init__(self, grid_node, from_col, to_col):
        self.grid = grid_node
        self.f_col = from_col
        self.t_col = to_col
    def evaluate(self, env): return replace_color(self.grid.evaluate(env), self.f_col.evaluate(env), self.t_col.evaluate(env))
    def __str__(self): return f"replace_color({self.grid}, {self.f_col}, {self.t_col})"

class PadNode(ASTNode):
    def __init__(self, grid_node, width_node, color_node):
        self.g = grid_node
        self.w = width_node
        self.c = color_node
    def evaluate(self, env): return pad(self.g.evaluate(env), self.w.evaluate(env), self.c.evaluate(env))
    def __str__(self): return f"pad({self.g}, {self.w}, {self.c})"

class FillBoxNode(ASTNode):
    def __init__(self, grid_node, box_node, color_node):
        self.g, self.b, self.c = grid_node, box_node, color_node
    def evaluate(self, env): return fill_box(self.g.evaluate(env), self.b.evaluate(env), self.c.evaluate(env))
    def __str__(self): return f"fill_box({self.g}, {self.b}, {self.c})"

class TransposeNode(ASTNode):
    def __init__(self, grid_node): self.grid_node = grid_node
    def evaluate(self, env): return transpose(self.grid_node.evaluate(env))
    def __str__(self): return f"transpose({self.grid_node})"

class TileNode(ASTNode):
    def __init__(self, grid_node, rows_node, cols_node):
        self.g, self.r, self.c = grid_node, rows_node, cols_node
    def evaluate(self, env): return tile(self.g.evaluate(env), self.r.evaluate(env), self.c.evaluate(env))
    def __str__(self): return f"tile({self.g}, {self.r}, {self.c})"

class OverlayNode(ASTNode):
    def __init__(self, base_node, top_node):
        self.base, self.top = base_node, top_node
    def evaluate(self, env): return overlay(self.base.evaluate(env), self.top.evaluate(env))
    def __str__(self): return f"overlay({self.base}, {self.top})"

class ScaleUpNode(ASTNode):
    def __init__(self, grid_node, factor_node):
        self.grid, self.factor = grid_node, factor_node
    def evaluate(self, env): return scale_up(self.grid.evaluate(env), self.factor.evaluate(env))
    def __str__(self): return f"scale_up({self.grid}, {self.factor})"

class StackVNode(ASTNode):
    def __init__(self, top_node, bottom_node):
        self.top_g, self.bottom_g = top_node, bottom_node
    def evaluate(self, env): return stack_v(self.top_g.evaluate(env), self.bottom_g.evaluate(env))
    def __str__(self): return f"stack_v({self.top_g}, {self.bottom_g})"

class StackHNode(ASTNode):
    def __init__(self, left_node, right_node):
        self.left_g, self.right_g = left_node, right_node
    def evaluate(self, env): return stack_h(self.left_g.evaluate(env), self.right_g.evaluate(env))
    def __str__(self): return f"stack_h({self.left_g}, {self.right_g})"

class LargestObjectNode(ASTNode):
    def __init__(self, grid_node): self.grid_node = grid_node
    def evaluate(self, env): return largest_object(self.grid_node.evaluate(env))
    def __str__(self): return f"largest_object({self.grid_node})"

class CountColorNode(ASTNode):
    def __init__(self, grid_node, color_node):
        self.grid, self.col = grid_node, color_node
    def evaluate(self, env): return count_color(self.grid.evaluate(env), self.col.evaluate(env))
    def __str__(self): return f"count_color({self.grid}, {self.col})"

class MostCommonColorNode(ASTNode):
    def __init__(self, grid_node): self.grid_node = grid_node
    def evaluate(self, env): return most_common_color(self.grid_node.evaluate(env))
    def __str__(self): return f"most_common_color({self.grid_node})"

# --- Evolutionary Operators ---
import copy

# Single-child Grid nodes that can be swapped with each other
_UNARY_GRID_NODES = [Rotate90Node, FlipYNode, FlipXNode, TransposeNode, LargestObjectNode]

def mutate_program(program, grammar):
    """
    Mutate an AST program by randomly altering one node.
    Always operates on a deep copy to prevent shared-reference bugs.
    """
    prog = copy.deepcopy(program)
    r = random.random()
    
    # Swap unary grid node type
    if r < 0.4 and isinstance(prog, tuple(_UNARY_GRID_NODES)):
        NewType = random.choice(_UNARY_GRID_NODES)
        return NewType(prog.grid_node)
    
    # Re-randomize a color/int child
    if r < 0.7:
        if hasattr(prog, 'col') and isinstance(prog.col, ColorNode):
            prog.col = ColorNode(random.choice([0,1,2,3,4,5,6,7,8,9]))
            return prog
        if hasattr(prog, 'c') and isinstance(prog.c, ColorNode):
            prog.c = ColorNode(random.choice([0,1,2,3,4,5,6,7,8,9]))
            return prog
        if hasattr(prog, 'f_col') and isinstance(prog.f_col, ColorNode):
            prog.f_col = ColorNode(random.choice([0,1,2,3,4,5,6,7,8,9]))
            return prog
        if hasattr(prog, 't_col') and isinstance(prog.t_col, ColorNode):
            prog.t_col = ColorNode(random.choice([0,1,2,3,4,5,6,7,8,9]))
            return prog
    
    # Replace entire subtree with fresh random
    return grammar.compose('Grid', max_depth=4)

def mutate_leaves(program):
    """
    Targeted leaf mutation: walk the AST and randomize exactly one ColorNode value.
    Preserves the program's structural skeleton for fine-grained parameter tuning.
    """
    prog = copy.deepcopy(program)
    
    # Collect all ColorNode references in the tree
    color_nodes = []
    stack = [prog]
    while stack:
        node = stack.pop()
        if isinstance(node, ColorNode):
            color_nodes.append(node)
        # Walk children
        for attr in ['grid_node', 'grid', 'g', 'base', 'top', 'col', 'c',
                     'f_col', 't_col', 'objs', 'box', 'b', 'w', 'r',
                     'factor', 'top_g', 'bottom_g', 'left_g', 'right_g']:
            child = getattr(node, attr, None)
            if child is not None and isinstance(child, ASTNode):
                stack.append(child)
    
    if color_nodes:
        # Mutate exactly one random color leaf
        target = random.choice(color_nodes)
        target.color_val = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    return prog

def crossover(parent_a, parent_b):
    """
    Combine two parent programs by inserting parent_b's grid subtree into parent_a.
    Uses deep copies to prevent circular references.
    """
    a = copy.deepcopy(parent_a)
    b = copy.deepcopy(parent_b)
    
    children = []
    if hasattr(a, 'grid_node'): children.append('grid_node')
    if hasattr(a, 'grid'): children.append('grid')
    if hasattr(a, 'g'): children.append('g')
    if hasattr(a, 'base'): children.append('base')
    if hasattr(a, 'top'): children.append('top')
    
    if children:
        attr_name = random.choice(children)
        setattr(a, attr_name, b)
        return a
    # Fallback: wrap parent_b in a random unary transform
    Wrapper = random.choice(_UNARY_GRID_NODES)
    return Wrapper(b)


class ARCGrammar(ActionGrammar):
    """
    Pillar 3: Abstraction & Composability.
    Provides the rules for generating nested functional expressions (Programs).
    19 primitives: rotate, flip_y/y, transpose, crop, paint, replace_color, pad, fill,
    tile, overlay, scale_up, stack_v, stack_h, largest_object, count_color, most_common_color.
    """
    @property
    def primitives(self):
        return [rotate90, flip_y, flip_x, transpose, get_objects, filter_by_color,
                crop_to_box, paint_objects, replace_color, pad, fill_box, tile, overlay,
                scale_up, stack_v, stack_h, largest_object, count_color, most_common_color]
        
    def compose(self, return_type, max_depth=3):
        return self._generate_typed_program(return_type, max_depth)
        
    def _generate_typed_program(self, return_type, max_depth=3):
        if return_type == 'Grid':
            if max_depth <= 1:
                return IdentityGridNode()
            # 19 grid-producing options
            choices = [
                lambda: IdentityGridNode(),
                lambda: Rotate90Node(self._generate_typed_program('Grid', max_depth-1)),
                lambda: FlipYNode(self._generate_typed_program('Grid', max_depth-1)),
                lambda: FlipXNode(self._generate_typed_program('Grid', max_depth-1)),
                lambda: TransposeNode(self._generate_typed_program('Grid', max_depth-1)),
                lambda: CropToBoxNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Box', max_depth-1)),
                lambda: ReplaceColorNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Color', max_depth-1), self._generate_typed_program('Color', max_depth-1)),
                lambda: PadNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Int', max_depth-1), self._generate_typed_program('Color', max_depth-1)),
                lambda: FillBoxNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Box', max_depth-1), self._generate_typed_program('Color', max_depth-1)),
                lambda: TileNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Int', max_depth-1), self._generate_typed_program('Int', max_depth-1)),
                lambda: OverlayNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Grid', max_depth-1)),
                lambda: PaintNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('ObjectList', max_depth-1), self._generate_typed_program('Color', max_depth-1)),
                lambda: ScaleUpNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Int', max_depth-1)),
                lambda: StackVNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Grid', max_depth-1)),
                lambda: StackHNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Grid', max_depth-1)),
                lambda: LargestObjectNode(self._generate_typed_program('Grid', max_depth-1)),
            ]
            return random.choice(choices)()
        elif return_type == 'ObjectList':
            if max_depth <= 1 or random.random() < 0.5:
                return GetObjectsNode(self._generate_typed_program('Grid', max_depth-1))
            else:
                return FilterColorNode(self._generate_typed_program('ObjectList', max_depth-1), self._generate_typed_program('Color', max_depth-1))
        elif return_type == 'Color':
            if max_depth > 1 and random.random() < 0.15:
                return MostCommonColorNode(self._generate_typed_program('Grid', max_depth-1))
            return ColorNode(random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        elif return_type == 'Box':
            return GetBoundingBoxNode(self._generate_typed_program('ObjectList', max_depth-1))
        elif return_type == 'Int':
            return ColorNode(random.choice([1, 2, 3]))

