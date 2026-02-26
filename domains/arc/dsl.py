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

def mirror_x(grid: Grid) -> Grid:
    return Grid(np.flipud(grid.arr))

def mirror_y(grid: Grid) -> Grid:
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

# --- Evolutionary Operators ---
import copy

# Single-child Grid nodes that can be swapped with each other
_UNARY_GRID_NODES = [Rotate90Node, MirrorXNode, MirrorYNode, TransposeNode]

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
    13 primitives: rotate, mirror_x/y, transpose, crop, paint, replace_color, pad, fill, tile, overlay.
    """
    @property
    def primitives(self):
        return [rotate90, mirror_x, mirror_y, transpose, get_objects, filter_by_color,
                crop_to_box, paint_objects, replace_color, pad, fill_box, tile, overlay]
        
    def compose(self, return_type, max_depth=3):
        return self._generate_typed_program(return_type, max_depth)
        
    def _generate_typed_program(self, return_type, max_depth=3):
        if return_type == 'Grid':
            if max_depth <= 1:
                return IdentityGridNode()
            # 13 grid-producing options with ~equal weight
            r = random.random()
            if r < 0.08: return IdentityGridNode()
            elif r < 0.16: return Rotate90Node(self._generate_typed_program('Grid', max_depth-1))
            elif r < 0.24: return MirrorXNode(self._generate_typed_program('Grid', max_depth-1))
            elif r < 0.32: return MirrorYNode(self._generate_typed_program('Grid', max_depth-1))
            elif r < 0.40: return TransposeNode(self._generate_typed_program('Grid', max_depth-1))
            elif r < 0.48: return CropToBoxNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Box', max_depth-1))
            elif r < 0.56: return ReplaceColorNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Color', max_depth-1), self._generate_typed_program('Color', max_depth-1))
            elif r < 0.64: return PadNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Int', max_depth-1), self._generate_typed_program('Color', max_depth-1))
            elif r < 0.72: return FillBoxNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Box', max_depth-1), self._generate_typed_program('Color', max_depth-1))
            elif r < 0.80: return TileNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Int', max_depth-1), self._generate_typed_program('Int', max_depth-1))
            elif r < 0.88: return OverlayNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('Grid', max_depth-1))
            else: return PaintNode(self._generate_typed_program('Grid', max_depth-1), self._generate_typed_program('ObjectList', max_depth-1), self._generate_typed_program('Color', max_depth-1))
        elif return_type == 'ObjectList':
            if max_depth <= 1 or random.random() < 0.5:
                return GetObjectsNode(self._generate_typed_program('Grid', max_depth-1))
            else:
                return FilterColorNode(self._generate_typed_program('ObjectList', max_depth-1), self._generate_typed_program('Color', max_depth-1))
        elif return_type == 'Color':
            return ColorNode(random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        elif return_type == 'Box':
            return GetBoundingBoxNode(self._generate_typed_program('ObjectList', max_depth-1))
        elif return_type == 'Int':
            return ColorNode(random.choice([1, 2, 3]))

