import unittest
import numpy as np
from domains.arc.env import Grid, BoundingBox
from domains.arc.dsl import (
    rotate90, mirror_x, mirror_y, get_objects, filter_by_color, crop_to_box, paint_objects,
    ColorNode, GetObjectsNode, CropToBoxNode, Rotate90Node, IdentityGridNode
)

class TestARCDSL(unittest.TestCase):
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

    def test_rotate90(self):
        rot = rotate90(self.grid)
        self.assertTrue(np.array_equal(rot.arr, np.array([[0, 0, 0], [1, 1, 0], [0, 2, 0]])))

    def test_mirror_x(self):
        mir = mirror_x(self.grid)
        self.assertTrue(np.array_equal(mir.arr, np.array([[0, 0, 0], [2, 1, 0], [0, 1, 0]])))

    def test_mirror_y(self):
        mir = mirror_y(self.grid)
        self.assertTrue(np.array_equal(mir.arr, np.array([[0, 1, 0], [0, 1, 2], [0, 0, 0]])))

    def test_crop_to_box(self):
        box = BoundingBox(0, 1, 0, 1)
        cropped = crop_to_box(self.grid, box)
        self.assertTrue(np.array_equal(cropped.arr, np.array([[0, 1], [2, 1]])))

    def test_paint_objects(self):
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
        expected = np.array([[0, 0, 0], [1, 1, 0], [0, 2, 0]])
        self.assertTrue(np.array_equal(rot_node.evaluate(env).arr, expected))

if __name__ == '__main__':
    unittest.main()
