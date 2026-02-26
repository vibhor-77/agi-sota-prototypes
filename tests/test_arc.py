import unittest
import numpy as np

from domains.arc.env import Grid, BoundingBox
from domains.arc.dsl import (
    rotate90, mirror_x, mirror_y, get_objects, filter_by_color, crop_to_box, paint_objects,
    ColorNode, GetObjectsNode, CropToBoxNode, Rotate90Node, IdentityGridNode
)
from domains.arc.heuristics import PixelEditDistance
from domains.arc.search import ARCBeamSearch

class TestARCDomain(unittest.TestCase):
    def setUp(self):
        self.grid = Grid([
            [0, 1, 0],
            [2, 1, 0],
            [0, 0, 0]
        ])

    def test_get_objects(self):
        objs = get_objects(self.grid)
        self.assertEqual(len(objs), 2)
        blue_objs = filter_by_color(objs, 1)
        self.assertEqual(len(blue_objs), 1)

    def test_rotate_mirror_crop_paint(self):
        rot = rotate90(self.grid)
        self.assertTrue(np.array_equal(rot.arr, np.array([[0, 0, 0], [1, 1, 0], [0, 2, 0]])))
        mir = mirror_x(self.grid)
        self.assertTrue(np.array_equal(mir.arr, np.array([[0, 0, 0], [2, 1, 0], [0, 1, 0]])))
        
        box = BoundingBox(0, 1, 0, 1)
        cropped = crop_to_box(self.grid, box)
        self.assertTrue(np.array_equal(cropped.arr, np.array([[0, 1], [2, 1]])))
        
        painted = paint_objects(self.grid, get_objects(self.grid), 3)
        self.assertTrue(np.array_equal(painted.arr, np.array([[0, 3, 0], [3, 3, 0], [0, 0, 0]])))

    def test_ast_evaluation(self):
        env = {'input_grid': self.grid}
        
        # Identity
        id_node = IdentityGridNode()
        self.assertTrue(np.array_equal(id_node.evaluate(env).arr, self.grid.arr))
        
        # Color
        c_node = ColorNode(3)
        self.assertEqual(c_node.evaluate(env), 3)
        
        # GetObjects
        get_node = GetObjectsNode(id_node)
        self.assertEqual(len(get_node.evaluate(env)), 2)
        
        # Rotate90 composed with Identity
        rot_node = Rotate90Node(id_node)
        self.assertTrue(np.array_equal(rot_node.evaluate(env).arr, np.array([[0, 0, 0], [1, 1, 0], [0, 2, 0]])))

    def test_pixel_edit_distance(self):
        heuristic = PixelEditDistance()
        pred = Grid([[1, 1], [0, 0]])
        self.assertEqual(heuristic.evaluate(pred, Grid([[1, 1], [0, 0]])), 0.0)
        self.assertEqual(heuristic.evaluate(pred, Grid([[1, 0], [0, 0]])), 0.25)
        
    def test_beam_search_convergence(self):
        # Trivial task: Output is always the exact same as input (Identity)
        agent = ARCBeamSearch()
        
        # Mute prints for testing
        import sys, os
        
        train_examples = [
            (Grid([[1, 0], [0, 1]]), Grid([[1, 0], [0, 1]])),
            (Grid([[2, 2], [0, 1]]), Grid([[2, 2], [0, 1]]))
        ]
        
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            best_prog = agent.search(train_examples, target=None, beam_width=10, max_generations=5)
            sys.stdout = sys.__stdout__
        
        # Ensure it found a program that scores 0.0 loss
        loss = agent._evaluate_program(best_prog, train_examples)
        self.assertEqual(loss, 0.0)

if __name__ == '__main__':
    unittest.main()
