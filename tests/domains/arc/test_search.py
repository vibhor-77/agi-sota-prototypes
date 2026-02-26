import unittest
import sys
import os
from domains.arc.env import Grid
from domains.arc.search import ARCBeamSearch

class TestARCSearch(unittest.TestCase):
    def test_beam_search_convergence(self):
        # Trivial task: Output is always the exact same as input (Identity)
        agent = ARCBeamSearch()
        
        train_examples = [
            (Grid([[1, 0], [0, 1]]), Grid([[1, 0], [0, 1]])),
            (Grid([[2, 2], [0, 1]]), Grid([[2, 2], [0, 1]]))
        ]
        
        # Mute prints for clean test output
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            best_prog = agent.search(train_examples, target=None, beam_width=10, max_generations=5)
            sys.stdout = sys.__stdout__
        
        self.assertIsNotNone(best_prog)
        # Ensure it found a program that scores 0.0 loss
        loss = agent._evaluate_program(best_prog, train_examples)
        self.assertEqual(loss, 0.0)

if __name__ == '__main__':
    unittest.main()
