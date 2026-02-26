import unittest
import numpy as np

from domains.arc.env import Grid, ArcObject, BoundingBox
from domains.arc.dsl import (
    rotate90, mirror_x, mirror_y, get_objects, filter_by_color, crop_to_box
)
from domains.arc.heuristics import PixelEditDistance

class TestARCDomain(unittest.TestCase):
    def setUp(self):
        self.grid = Grid([
            [0, 1, 0],
            [2, 1, 0],
            [0, 0, 0]
        ])

    def test_get_objects(self):
        objs = get_objects(self.grid)
        self.assertEqual(len(objs), 2) # Blue item (1) and Red item (2)
        
        blue_objs = filter_by_color(objs, 1)
        self.assertEqual(len(blue_objs), 1)
        self.assertEqual(blue_objs[0].color, 1)

    def test_rotate90(self):
        rot = rotate90(self.grid)
        expected = np.array([
            [0, 0, 0],
            [1, 1, 0],
            [0, 2, 0]
        ])
        self.assertTrue(np.array_equal(rot.arr, expected))

    def test_mirror_x(self):
        mir = mirror_x(self.grid)
        expected = np.array([
            [0, 0, 0],
            [2, 1, 0],
            [0, 1, 0]
        ])
        self.assertTrue(np.array_equal(mir.arr, expected))

    def test_crop_to_box(self):
        box = BoundingBox(0, 1, 0, 1) # Top left 2x2
        cropped = crop_to_box(self.grid, box)
        expected = np.array([
            [0, 1],
            [2, 1]
        ])
        self.assertTrue(np.array_equal(cropped.arr, expected))

    def test_pixel_edit_distance(self):
        heuristic = PixelEditDistance()
        pred = Grid([[1, 1], [0, 0]])
        target1 = Grid([[1, 1], [0, 0]])
        target2 = Grid([[1, 0], [0, 0]])
        
        # Perfect match
        self.assertEqual(heuristic.evaluate(pred, target1), 0.0)
        # 1 pixel off out of 4 (25% error)
        self.assertEqual(heuristic.evaluate(pred, target2), 0.25)

if __name__ == '__main__':
    unittest.main()
