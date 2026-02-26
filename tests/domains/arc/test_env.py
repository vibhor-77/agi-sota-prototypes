import unittest
import numpy as np
import os
import tempfile
import json
from domains.arc.env import Grid, ArcObject, BoundingBox, ARCEnvironment, load_official_arc_task, generate_2d_arc_task

class TestARCEnv(unittest.TestCase):
    def setUp(self):
        self.grid = Grid([
            [0, 1, 0],
            [2, 1, 0],
            [0, 0, 0]
        ])

    def test_grid_equality(self):
        grid2 = Grid([
            [0, 1, 0],
            [2, 1, 0],
            [0, 0, 0]
        ])
        grid3 = Grid([
            [0, 1, 0],
            [2, 2, 0],
            [0, 0, 0]
        ])
        self.assertEqual(self.grid, grid2)
        self.assertNotEqual(self.grid, grid3)

    def test_environment_initialization(self):
        env = ARCEnvironment(self.grid)
        self.assertEqual(env.get_observation(), self.grid)

    def test_load_official_task(self):
        # Create a mock json file
        mock_data = {
            "train": [
                {"input": [[0, 1]], "output": [[1, 0]]}
            ],
            "test": [
                {"input": [[0, 2]], "output": [[2, 0]]}
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_data, f)
            temp_path = f.name
            
        try:
            train_ex, test_ex = load_official_arc_task(temp_path)
            self.assertEqual(len(train_ex), 1)
            self.assertEqual(len(test_ex), 1)
            self.assertEqual(train_ex[0][0].arr.tolist(), [[0, 1]])
            self.assertEqual(train_ex[0][1].arr.tolist(), [[1, 0]])
            self.assertEqual(test_ex[0][0].arr.tolist(), [[0, 2]])
            self.assertEqual(test_ex[0][1].arr.tolist(), [[2, 0]])
        finally:
            os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
