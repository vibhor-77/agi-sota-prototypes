"""
Primitive Library with Library Learning

A growing library of composable primitives. Starts with domain-registered primitives,
grows via Library Learning: compressing frequent sub-programs from successful solutions
into reusable named primitives.

Inspired by DreamCoder's wake-sleep cycle for program abstraction.
"""
import random
import copy
from collections import Counter
from typing import Any, Dict, List, Optional, Callable
from core.program import Program, Primitive, Apply, Constant, Variable, LearnedPrimitive


class PrimitiveLibrary:
    """
    A growing library of composable primitives — Pillar 3 made general.
    
    Primitives are typed: each has input_types and an output_type.
    The library can compose random type-correct programs and learn
    new primitives from successful solutions.
    """
    
    def __init__(self):
        self.primitives: Dict[str, Primitive] = {}
        self.learned: Dict[str, LearnedPrimitive] = {}
        self.type_index: Dict[str, List[str]] = {}  # output_type -> [primitive_names]
        self.constant_generators: Dict[str, Callable] = {
            'Int': lambda: Constant(random.choice([1, 2, 3]), 'Int'),
            'Color': lambda: Constant(random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 'Color'),
        }
        self.variable_types: Dict[str, str] = {}  # var_name -> type
    
    def register(self, name: str, fn: Callable, input_types: List[str], output_type: str):
        """Register a primitive function with typed signature."""
        prim = Primitive(name, fn, input_types, output_type)
        self.primitives[name] = prim
        # Index by output type for type-directed synthesis
        if output_type not in self.type_index:
            self.type_index[output_type] = []
        self.type_index[output_type].append(name)
    
    def register_variable(self, name: str, type_name: str):
        """Register an environment variable (e.g., 'input_grid' of type 'Grid')."""
        self.variable_types[name] = type_name
    
    def register_constant_generator(self, type_name: str, gen_fn: Callable):
        """Register a generator for constants of a given type."""
        self.constant_generators[type_name] = gen_fn
    
    @property
    def all_primitives(self) -> List[Primitive]:
        """All registered primitives (built-in + learned)."""
        prims = list(self.primitives.values())
        # Include learned primitives wrapped as Primitives
        for lp in self.learned.values():
            prims.append(Primitive(lp.name, lambda env, _lp=lp: _lp.execute(env),
                                    [], lp.output_type))
        return prims
    
    def compose_random(self, output_type: str, max_depth: int = 3) -> Program:
        """
        Generate a random type-correct program that produces output_type.
        Uses type-directed synthesis: pick a primitive that returns output_type,
        then recursively generate its arguments.
        """
        # Base case: return a variable or constant
        if max_depth <= 1:
            return self._generate_leaf(output_type)
        
        # Find all primitives that produce this output type
        candidates = self.type_index.get(output_type, [])
        
        if not candidates:
            return self._generate_leaf(output_type)
        
        # Small chance of returning a leaf even at depth > 1
        if random.random() < 0.1:
            return self._generate_leaf(output_type)
        
        # Pick a random primitive and recursively generate its arguments
        prim_name = random.choice(candidates)
        prim = self.primitives[prim_name]
        
        args = []
        for in_type in prim.input_types:
            args.append(self.compose_random(in_type, max_depth - 1))
        
        program = Apply(prim, args)
        program._output_type = output_type
        return program
    
    def _generate_leaf(self, type_name: str) -> Program:
        """Generate a leaf node (variable or constant) of the given type."""
        # Check for matching variables
        matching_vars = [name for name, t in self.variable_types.items() if t == type_name]
        
        # Check for constant generators
        has_const_gen = type_name in self.constant_generators
        
        if matching_vars and (not has_const_gen or random.random() < 0.7):
            name = random.choice(matching_vars)
            return Variable(name, type_name)
        
        if has_const_gen:
            return self.constant_generators[type_name]()
        
        # Fallback: return a generic variable
        if matching_vars:
            name = random.choice(matching_vars)
            return Variable(name, type_name)
        
        return Constant(0, type_name)
    
    def compress(self, solved_programs: List[Program]) -> List[str]:
        """
        Library Learning: find common sub-trees in solved programs
        and abstract them into new named primitives.
        
        Returns names of newly learned primitives.
        """
        if len(solved_programs) < 2:
            return []
        
        # 1. Extract all sub-trees from solved programs (keyed by str representation)
        subtree_counts = Counter()
        subtree_examples = {}
        
        for prog in solved_programs:
            seen_in_prog = set()
            for node in prog.all_nodes():
                # Only consider non-trivial subtrees (size > 1, depth > 1)
                if node.size() > 1 and node.depth() > 1:
                    key = str(node)
                    if key not in seen_in_prog:
                        seen_in_prog.add(key)
                        subtree_counts[key] += 1
                        subtree_examples[key] = node
        
        # 2. Find subtrees that appear in >= 2 different solved programs
        new_primitives = []
        for key, count in subtree_counts.most_common(5):  # Top 5 most common
            if count < 2:
                break
            
            # Don't learn trivially small subtrees
            example = subtree_examples[key]
            if example.size() < 3:
                continue
            
            # Already learned?
            if key in self.learned:
                continue
            
            # Abstract into a learned primitive
            name = f"learned_{len(self.learned)}"
            learned = LearnedPrimitive(
                name=name,
                body=copy.deepcopy(example),
                input_vars=list(self.variable_types.keys()),
                output_type='Grid'  # Default; could be inferred
            )
            self.learned[name] = learned
            new_primitives.append(name)
            
            # Register so it can be used in future compositions
            if 'Grid' not in self.type_index:
                self.type_index['Grid'] = []
            self.type_index['Grid'].append(name)
            self.primitives[name] = Primitive(
                name, lambda env, _lp=learned: _lp.execute(env), [], 'Grid'
            )
        
        return new_primitives
    
    def summary(self) -> str:
        """Human-readable library summary."""
        lines = [f"PrimitiveLibrary: {len(self.primitives)} primitives, {len(self.learned)} learned"]
        for t, names in sorted(self.type_index.items()):
            lines.append(f"  {t}: {', '.join(names)}")
        if self.learned:
            lines.append("  Learned:")
            for name, lp in self.learned.items():
                lines.append(f"    {name} (used {lp.usage_count}x): {lp.body}")
        return '\n'.join(lines)
    
    def save(self, path: str):
        """
        Save learned primitives to disk (JSON).
        Only saves learned primitives — built-in primitives are re-registered by the adapter.
        """
        import json, os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        data = {
            'version': 1,
            'learned_count': len(self.learned),
            'primitives': []
        }
        for name, lp in self.learned.items():
            data['primitives'].append({
                'name': name,
                'body_str': str(lp.body),
                'body_program': _serialize_program(lp.body),
                'output_type': lp.output_type,
                'input_vars': lp.input_vars,
                'usage_count': lp.usage_count,
            })
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str) -> int:
        """
        Load learned primitives from disk and register them.
        Returns the number of primitives loaded.
        """
        import json, os
        if not os.path.exists(path):
            return 0
        
        with open(path) as f:
            data = json.load(f)
        
        loaded = 0
        for entry in data.get('primitives', []):
            name = entry['name']
            if name in self.learned:
                continue  # Already loaded
            
            # Reconstruct the program tree from serialized form
            body = _deserialize_program(entry['body_program'], self)
            if body is None:
                continue
            
            learned = LearnedPrimitive(
                name=name,
                body=body,
                input_vars=entry.get('input_vars', []),
                output_type=entry.get('output_type', 'Grid')
            )
            learned.usage_count = entry.get('usage_count', 0)
            self.learned[name] = learned
            
            # Register for future compositions
            out_type = entry.get('output_type', 'Grid')
            if out_type not in self.type_index:
                self.type_index[out_type] = []
            if name not in self.type_index[out_type]:
                self.type_index[out_type].append(name)
            self.primitives[name] = Primitive(
                name, lambda env, _lp=learned: _lp.execute(env), [], out_type
            )
            loaded += 1
        
        return loaded


def _serialize_program(prog: Program) -> dict:
    """Recursively serialize a Program tree to a JSON-safe dict."""
    if isinstance(prog, Variable):
        return {'type': 'Variable', 'name': prog.name, 'type_name': prog.type_name}
    elif isinstance(prog, Constant):
        return {'type': 'Constant', 'value': prog.value, 'type_name': prog.type_name}
    elif isinstance(prog, Primitive):
        return {'type': 'Primitive', 'name': prog.name}
    elif isinstance(prog, Apply):
        return {
            'type': 'Apply',
            'fn': _serialize_program(prog.fn_node),
            'args': [_serialize_program(a) for a in prog.arg_nodes]
        }
    elif isinstance(prog, LearnedPrimitive):
        return {'type': 'Learned', 'name': prog.name, 'body': _serialize_program(prog.body)}
    return {'type': 'Unknown'}


def _deserialize_program(data: dict, library: 'PrimitiveLibrary') -> Optional[Program]:
    """Reconstruct a Program tree from serialized dict."""
    if data is None:
        return None
    
    t = data.get('type')
    if t == 'Variable':
        return Variable(data['name'], data.get('type_name', 'Any'))
    elif t == 'Constant':
        return Constant(data['value'], data.get('type_name', 'Int'))
    elif t == 'Primitive':
        name = data['name']
        if name in library.primitives:
            return library.primitives[name]
        return None
    elif t == 'Apply':
        fn = _deserialize_program(data['fn'], library)
        if fn is None:
            return None
        args = [_deserialize_program(a, library) for a in data.get('args', [])]
        if None in args:
            return None
        return Apply(fn, args)
    elif t == 'Learned':
        body = _deserialize_program(data.get('body'), library)
        if body is None:
            return None
        return LearnedPrimitive(data['name'], body, [], 'Grid')
    return None
