import unittest
from domains.arc.env import Grid
from domains.arc.heuristics import PixelEditDistance

class TestARCHeuristics(unittest.TestCase):
    def test_pixel_edit_distance(self):
        heuristic = PixelEditDistance()
        pred = Grid([[1, 1], [0, 0]])
        self.assertEqual(heuristic.evaluate(pred, Grid([[1, 1], [0, 0]])), 0.0)
        self.assertEqual(heuristic.evaluate(pred, Grid([[1, 0], [0, 0]])), 0.25)
        self.assertEqual(heuristic.evaluate(pred, Grid([[2, 2], [2, 2]])), 1.0)
        
    def test_shape_mismatch(self):
        # A shape mismatch should return a high penalty
        heuristic = PixelEditDistance()
        pred = Grid([[1, 1], [0, 0]])
        target = Grid([[1, 1, 1], [0, 0, 0]])
        self.assertEqual(heuristic.evaluate(pred, target), 1.5)

if __name__ == '__main__':
    unittest.main()
