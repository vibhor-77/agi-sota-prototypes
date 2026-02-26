import unittest
from core.feedback import Environment
from core.approximability import Heuristic
from core.abstraction import StateRepresentation, ActionGrammar
from core.exploration import SearchAlgorithm

class TestInterfaces(unittest.TestCase):
    def test_environment_abc(self):
        class IncompleteEnv(Environment):
            pass
        with self.assertRaises(TypeError):
            env = IncompleteEnv()
            
    def test_heuristic_abc(self):
        class IncompleteHeuristic(Heuristic):
            pass
        with self.assertRaises(TypeError):
            h = IncompleteHeuristic()
            
    def test_state_representation_abc(self):
        class IncompleteRepr(StateRepresentation):
            pass
        with self.assertRaises(TypeError):
            r = IncompleteRepr()
            
    def test_action_grammar_abc(self):
        class IncompleteGrammar(ActionGrammar):
            pass
        with self.assertRaises(TypeError):
            g = IncompleteGrammar()
            
    def test_search_algorithm_abc(self):
        class IncompleteSearch(SearchAlgorithm):
            pass
        with self.assertRaises(TypeError):
            s = IncompleteSearch()

if __name__ == '__main__':
    unittest.main()
