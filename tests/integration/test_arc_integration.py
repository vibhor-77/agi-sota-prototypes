import unittest
import sys
import os
from domains.arc.env import ARCEnvironment, generate_2d_arc_task
from domains.arc.search import ARCBeamSearch

class TestARCEndToEnd(unittest.TestCase):
    def test_arc_sota_level_3(self):
        """
        Tests the entire pipeline: loading official JSON (Level 3: 74dd1130), 
        Beam Search synthesizing the correct AST geometry primitives, 
        and flawless execution on the hidden Grid.
        """
        train_ex, test_tests = generate_2d_arc_task(level=3)
        test_ex = test_tests[0]
        
        agent = ARCBeamSearch()
        
        # Mute prints for testing
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            best_program = agent.search(train_ex, target=None, beam_width=10, max_generations=5)
            sys.stdout = sys.__stdout__
            
        # The beam search may find multiple mathematically equivalent programs
        # So we only assert that it found *a* program that solves the validation output
        self.assertIsNotNone(best_program)
        
        # Execute on unseen test data
        env = ARCEnvironment(test_ex[0])
        predicted = env.execute_action(best_program)
        
        self.assertEqual(predicted, test_ex[1])

if __name__ == '__main__':
    unittest.main()
