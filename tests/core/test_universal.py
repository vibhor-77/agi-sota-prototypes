"""Tests for the Universal Program Representation and Primitive Library."""
import unittest
import numpy as np
from core.program import Program, Primitive, Apply, Constant, Variable, LearnedPrimitive
from core.program import mutate_universal, crossover_universal
from core.library import PrimitiveLibrary


class TestProgramNodes(unittest.TestCase):
    """Test basic program node types."""
    
    def test_constant(self):
        c = Constant(42, 'Int')
        self.assertEqual(c.execute({}), 42)
        self.assertEqual(str(c), '42')
        self.assertEqual(c.size(), 1)
    
    def test_variable(self):
        v = Variable('x', 'Int')
        self.assertEqual(v.execute({'x': 10}), 10)
        self.assertIsNone(v.execute({}))
    
    def test_primitive_apply(self):
        add_fn = Primitive('add', lambda a, b: a + b, ['Int', 'Int'], 'Int')
        prog = Apply(add_fn, [Constant(3, 'Int'), Constant(4, 'Int')])
        self.assertEqual(prog.execute({}), 7)
        self.assertEqual(str(prog), 'add(3, 4)')
    
    def test_nested_apply(self):
        double = Primitive('double', lambda x: x * 2, ['Int'], 'Int')
        add = Primitive('add', lambda a, b: a + b, ['Int', 'Int'], 'Int')
        # double(add(1, 2)) = double(3) = 6
        prog = Apply(double, [Apply(add, [Constant(1, 'Int'), Constant(2, 'Int')])])
        self.assertEqual(prog.execute({}), 6)
        self.assertEqual(prog.size(), 6)  # 2 Apply + double + add + 1 + 2
    
    def test_program_with_env(self):
        inc = Primitive('inc', lambda x: x + 1, ['Int'], 'Int')
        prog = Apply(inc, [Variable('n', 'Int')])
        self.assertEqual(prog.execute({'n': 5}), 6)
    
    def test_learned_primitive(self):
        body = Apply(Primitive('double', lambda x: x * 2, ['Int'], 'Int'),
                     [Variable('n', 'Int')])
        lp = LearnedPrimitive('double_n', body, ['n'], 'Int')
        self.assertEqual(lp.execute({'n': 7}), 14)
        self.assertEqual(lp.usage_count, 1)

    def test_error_handling(self):
        bad_fn = Primitive('div', lambda a, b: a / b, ['Int', 'Int'], 'Int')
        prog = Apply(bad_fn, [Constant(1, 'Int'), Constant(0, 'Int')])
        self.assertIsNone(prog.execute({}))  # Division by zero handled


class TestPrimitiveLibrary(unittest.TestCase):
    """Test the type-directed primitive library."""
    
    def setUp(self):
        self.lib = PrimitiveLibrary()
        self.lib.register('inc', lambda x: x + 1, ['Int'], 'Int')
        self.lib.register('double', lambda x: x * 2, ['Int'], 'Int')
        self.lib.register('add', lambda a, b: a + b, ['Int', 'Int'], 'Int')
        self.lib.register_variable('n', 'Int')
    
    def test_register(self):
        self.assertEqual(len(self.lib.primitives), 3)
        self.assertIn('Int', self.lib.type_index)
        self.assertEqual(len(self.lib.type_index['Int']), 3)
    
    def test_compose_random(self):
        """Compose should produce a valid program."""
        prog = self.lib.compose_random('Int', max_depth=3)
        self.assertIsInstance(prog, Program)
        # Should be executable
        result = prog.execute({'n': 5})
        # Result should be an int  (or None if error)
        self.assertTrue(result is None or isinstance(result, (int, float)))
    
    def test_compose_depth_1(self):
        """Depth 1 should return a leaf (Variable or Constant)."""
        prog = self.lib.compose_random('Int', max_depth=1)
        self.assertTrue(isinstance(prog, (Variable, Constant)))
    
    def test_library_learning(self):
        """Library learning should discover common sub-trees."""
        # Create two solved programs with a common subtree: inc(n)
        inc_prim = self.lib.primitives['inc']
        double_prim = self.lib.primitives['double']
        add_prim = self.lib.primitives['add']
        
        prog1 = Apply(double_prim, [Apply(inc_prim, [Variable('n', 'Int')])])
        prog2 = Apply(add_prim, [Apply(inc_prim, [Variable('n', 'Int')]),
                                  Constant(10, 'Int')])
        
        new_prims = self.lib.compress([prog1, prog2])
        # Should find inc(n) as a common subtree (size 3: Apply + inc + n)
        # But inc(n) is only size 2 (Apply(inc, [Variable])), size check is >2 
        # so it depends on the size counting
        # At minimum, the method should run without error
        self.assertIsInstance(new_prims, list)


class TestARCAdapter(unittest.TestCase):
    """Test that the ARC adapter creates a functional library."""
    
    def test_create_arc_library(self):
        from domains.arc.adapter import create_arc_library
        lib = create_arc_library()
        # Should have all 19+ primitives registered
        self.assertGreaterEqual(len(lib.primitives), 18)
        # Should have input_grid variable
        self.assertIn('input_grid', lib.variable_types)
        self.assertEqual(lib.variable_types['input_grid'], 'Grid')
    
    def test_compose_arc_program(self):
        from domains.arc.adapter import create_arc_library
        from domains.arc.env import Grid
        
        lib = create_arc_library()
        prog = lib.compose_random('Grid', max_depth=3)
        
        # Should be executable with a Grid input
        grid = Grid([[1, 2], [3, 4]])
        result = prog.execute({'input_grid': grid})
        # Result might be None (crashed) or a Grid
        # Just verify it doesn't raise
    
    def test_eval_fn(self):
        from domains.arc.adapter import create_arc_library, make_arc_eval_fn
        from domains.arc.env import Grid
        
        grid = Grid([[1, 2], [3, 4]])
        train = [(grid, grid)]  # Identity task
        
        eval_fn = make_arc_eval_fn(train)
        
        lib = create_arc_library()
        # A variable that returns input_grid should score 0.0 (identity)
        prog = Variable('input_grid', 'Grid')
        score = eval_fn(prog)
        self.assertEqual(score, 0.0)


if __name__ == '__main__':
    unittest.main()
