import unittest
import numpy as np
from domains.arc.env import Grid, BoundingBox
from domains.arc.dsl import (
    rotate90, mirror_x, mirror_y, get_objects, filter_by_color, crop_to_box, paint_objects,
    count_color, most_common_color, scale_up, stack_v, stack_h, largest_object,
    ColorNode, GetObjectsNode, CropToBoxNode, Rotate90Node, IdentityGridNode,
    ScaleUpNode, LargestObjectNode, StackVNode, StackHNode, MostCommonColorNode
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

    def test_count_color(self):
        self.assertEqual(count_color(self.grid, 0), 6)
        self.assertEqual(count_color(self.grid, 1), 2)
        self.assertEqual(count_color(self.grid, 2), 1)
        self.assertEqual(count_color(self.grid, 5), 0)

    def test_most_common_color(self):
        self.assertEqual(most_common_color(self.grid), 1)  # 2 blue pixels vs 1 red
        empty = Grid([[0, 0], [0, 0]])
        self.assertEqual(most_common_color(empty), 0)

    def test_scale_up(self):
        tiny = Grid([[1, 2], [3, 4]])
        scaled = scale_up(tiny, 2)
        expected = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        self.assertTrue(np.array_equal(scaled.arr, expected))
        # Clamped to max 5
        scaled5 = scale_up(Grid([[1]]), 10)
        self.assertEqual(scaled5.arr.shape, (5, 5))

    def test_stack_v(self):
        a = Grid([[1, 2], [3, 4]])
        b = Grid([[5, 6]])
        result = stack_v(a, b)
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(np.array_equal(result.arr, expected))

    def test_stack_h(self):
        a = Grid([[1], [2]])
        b = Grid([[3, 4], [5, 6]])
        result = stack_h(a, b)
        expected = np.array([[1, 3, 4], [2, 5, 6]])
        self.assertTrue(np.array_equal(result.arr, expected))

    def test_largest_object(self):
        grid = Grid([
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 2]
        ])
        result = largest_object(grid)
        # Largest is the 3-pixel blue object
        expected = np.array([[1, 0], [1, 1]])
        self.assertTrue(np.array_equal(result.arr, expected))

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

    def test_new_ast_nodes(self):
        env = {'input_grid': self.grid}
        id_node = IdentityGridNode()
        
        # LargestObjectNode
        lo = LargestObjectNode(id_node)
        result = lo.evaluate(env)
        self.assertEqual(result.arr.shape, (2, 1))  # The 2 blue pixels form a 2x1 column
        
        # ScaleUpNode
        su = ScaleUpNode(id_node, ColorNode(2))
        result = su.evaluate(env)
        self.assertEqual(result.arr.shape, (6, 6))
        
        # MostCommonColorNode
        mc = MostCommonColorNode(id_node)
        self.assertEqual(mc.evaluate(env), 1)

if __name__ == '__main__':
    unittest.main()
