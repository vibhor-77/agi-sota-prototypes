"""
ARC Domain Adapter

Registers all 19 ARC primitives into the universal PrimitiveLibrary.
This thin wrapper is the ONLY domain-specific code needed to use
the UniversalSolver for ARC tasks.
"""
from core.library import PrimitiveLibrary
from domains.arc.dsl import (
    rotate90, mirror_x, mirror_y, transpose,
    get_objects, filter_by_color, crop_to_box, paint_objects,
    replace_color, pad, fill_box, tile, overlay,
    scale_up, stack_v, stack_h, largest_object,
    count_color, most_common_color
)
from domains.arc.env import Grid, BoundingBox


def create_arc_library() -> PrimitiveLibrary:
    """Create a PrimitiveLibrary populated with all ARC primitives."""
    lib = PrimitiveLibrary()
    
    # Register the input variable
    lib.register_variable('input_grid', 'Grid')
    
    # --- Unary Grid -> Grid primitives ---
    lib.register('rotate90', rotate90, ['Grid'], 'Grid')
    lib.register('mirror_x', mirror_x, ['Grid'], 'Grid')
    lib.register('mirror_y', mirror_y, ['Grid'], 'Grid')
    lib.register('transpose', transpose, ['Grid'], 'Grid')
    lib.register('largest_object', largest_object, ['Grid'], 'Grid')
    
    # --- Grid + params -> Grid ---
    lib.register('replace_color', replace_color, ['Grid', 'Color', 'Color'], 'Grid')
    lib.register('pad', pad, ['Grid', 'Int', 'Color'], 'Grid')
    lib.register('fill_box', fill_box, ['Grid', 'Box', 'Color'], 'Grid')
    lib.register('tile', tile, ['Grid', 'Int', 'Int'], 'Grid')
    lib.register('scale_up', scale_up, ['Grid', 'Int'], 'Grid')
    
    # --- Grid + Grid -> Grid ---
    lib.register('overlay', overlay, ['Grid', 'Grid'], 'Grid')
    lib.register('stack_v', stack_v, ['Grid', 'Grid'], 'Grid')
    lib.register('stack_h', stack_h, ['Grid', 'Grid'], 'Grid')
    
    # --- Grid + ObjectList -> Grid ---
    lib.register('paint_objects', paint_objects, ['Grid', 'ObjectList', 'Color'], 'Grid')
    lib.register('crop_to_box', crop_to_box, ['Grid', 'Box'], 'Grid')
    
    # --- Object operations ---
    lib.register('get_objects', get_objects, ['Grid'], 'ObjectList')
    lib.register('filter_by_color', filter_by_color, ['ObjectList', 'Color'], 'ObjectList')
    
    # --- Constant generators ---
    import random
    lib.register_constant_generator('Color',
        lambda: __import__('core.program', fromlist=['Constant']).Constant(
            random.choice([0,1,2,3,4,5,6,7,8,9]), 'Color'))
    lib.register_constant_generator('Int',
        lambda: __import__('core.program', fromlist=['Constant']).Constant(
            random.choice([1,2,3]), 'Int'))
    
    # --- Box type: derived from ObjectList ---
    def get_bbox(obj_list):
        """Get bounding box of the first object."""
        if not obj_list:
            return BoundingBox(0, 0, 0, 0)
        obj = obj_list[0]
        import numpy as np
        rows, cols = np.where(obj.mask)
        if len(rows) == 0:
            return BoundingBox(0, 0, 0, 0)
        return BoundingBox(int(np.min(rows)), int(np.max(rows)),
                          int(np.min(cols)), int(np.max(cols)))
    
    lib.register('bbox', get_bbox, ['ObjectList'], 'Box')
    
    return lib


def make_arc_eval_fn(train_examples):
    """
    Create an evaluation function for a specific ARC task.
    Returns a callable: program -> float (0.0 = perfect).
    """
    from domains.arc.heuristics import PixelEditDistance
    heuristic = PixelEditDistance()
    
    def eval_fn(program):
        total_dist = 0.0
        for inp, out in train_examples:
            try:
                pred = program.execute({'input_grid': inp})
                if pred is None:
                    return 10.0
                total_dist += heuristic.evaluate(pred, out)
            except Exception:
                return 10.0
        return total_dist / len(train_examples)
    
    return eval_fn
