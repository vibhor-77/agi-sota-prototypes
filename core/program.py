"""
Universal Program Representation

A typed program tree that replaces all domain-specific AST nodes.
Every program is a tree of composable nodes: Primitive, Apply, Constant, Variable.
This is the universal language in which the solver thinks.
"""
import copy
import random
from typing import Any, List, Optional, Callable


class Program:
    """Base class for all program nodes."""
    
    def execute(self, env: dict) -> Any:
        """Execute this program in the given environment."""
        raise NotImplementedError
    
    def children(self) -> List['Program']:
        """Return child program nodes for tree traversal."""
        return []
    
    def size(self) -> int:
        """AST node count (used by library learning for compression)."""
        return 1 + sum(c.size() for c in self.children())
    
    def all_nodes(self) -> List['Program']:
        """Flatten the tree into a list of all nodes (pre-order)."""
        result = [self]
        for c in self.children():
            result.extend(c.all_nodes())
        return result
    
    def depth(self) -> int:
        """Maximum depth of the program tree."""
        kids = self.children()
        if not kids:
            return 1
        return 1 + max(c.depth() for c in kids)
    
    def __repr__(self):
        return str(self)


class Primitive(Program):
    """
    A named function with a callable implementation.
    Example: Primitive('rotate90', rotate90_fn, ['Grid'], 'Grid')
    """
    def __init__(self, name: str, fn: Callable, input_types: List[str], output_type: str):
        self.name = name
        self.fn = fn
        self.input_types = input_types
        self.output_type = output_type
    
    def execute(self, env: dict) -> Any:
        # A bare primitive returns itself (needs Apply to be called)
        return self
    
    def __str__(self):
        return self.name


class Apply(Program):
    """
    Function application: Apply(f, [arg1, arg2, ...]).
    Applies a Primitive to its arguments.
    """
    def __init__(self, fn_node: Program, arg_nodes: List[Program]):
        self.fn_node = fn_node
        self.arg_nodes = arg_nodes
    
    def execute(self, env: dict) -> Any:
        fn = self.fn_node
        if isinstance(fn, Primitive):
            args = [a.execute(env) for a in self.arg_nodes]
            try:
                return fn.fn(*args)
            except Exception:
                return None
        # If fn is an Apply or other node, execute it first
        fn_val = fn.execute(env)
        if isinstance(fn_val, Primitive):
            args = [a.execute(env) for a in self.arg_nodes]
            try:
                return fn_val.fn(*args)
            except Exception:
                return None
        return None
    
    def children(self) -> List[Program]:
        return [self.fn_node] + self.arg_nodes
    
    def __str__(self):
        args = ', '.join(str(a) for a in self.arg_nodes)
        return f"{self.fn_node}({args})"


class Constant(Program):
    """A literal value (int, string, etc.)."""
    def __init__(self, value: Any, type_name: str = 'Int'):
        self.value = value
        self.type_name = type_name
    
    def execute(self, env: dict) -> Any:
        return self.value
    
    def __str__(self):
        return repr(self.value)


class Variable(Program):
    """Reference to an environment input."""
    def __init__(self, name: str, type_name: str = 'Any'):
        self.name = name
        self.type_name = type_name
    
    def execute(self, env: dict) -> Any:
        return env.get(self.name)
    
    def __str__(self):
        return self.name


class LearnedPrimitive(Program):
    """
    A primitive discovered by Library Learning.
    Wraps a sub-program that appeared frequently in successful solutions.
    """
    def __init__(self, name: str, body: Program, input_vars: List[str], output_type: str):
        self.name = name
        self.body = body
        self.input_vars = input_vars
        self.output_type = output_type
        self.usage_count = 0
    
    def execute(self, env: dict) -> Any:
        self.usage_count += 1
        return self.body.execute(env)
    
    def children(self) -> List[Program]:
        return [self.body]
    
    def __str__(self):
        return f"{self.name}({self.body})"


# --- Evolutionary Operators for Universal Programs ---

def mutate_universal(program: Program, library) -> Program:
    """
    Mutate a universal program by randomly altering one node.
    Works on any program tree regardless of domain.
    """
    prog = copy.deepcopy(program)
    nodes = prog.all_nodes()
    
    # Try to mutate a constant (targeted leaf mutation)
    constants = [n for n in nodes if isinstance(n, Constant)]
    if constants and random.random() < 0.5:
        target = random.choice(constants)
        if target.type_name == 'Int':
            target.value = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        elif target.type_name == 'Color':
            target.value = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        return prog
    
    # Replace a random subtree with a fresh random program
    applies = [n for n in nodes if isinstance(n, Apply) and n.arg_nodes]
    if applies:
        target = random.choice(applies)
        idx = random.randrange(len(target.arg_nodes))
        # Ensure we maintain type correctness
        old_arg = target.arg_nodes[idx]
        output_type = getattr(old_arg, '_output_type', 'Grid')
        if isinstance(old_arg, Constant):
            output_type = old_arg.type_name
        elif isinstance(old_arg, Variable):
            output_type = old_arg.type_name
        elif isinstance(old_arg, Apply) and hasattr(old_arg.fn_node, 'output_type'):
            output_type = old_arg.fn_node.output_type
            
        target.arg_nodes[idx] = library.compose_random(output_type, max_depth=3)
    return prog


def crossover_universal(parent_a: Program, parent_b: Program) -> Program:
    """
    Crossover: take the structure of parent_a and graft a subtree from parent_b.
    """
    a = copy.deepcopy(parent_a)
    b = copy.deepcopy(parent_b)
    
    a_applies = [n for n in a.all_nodes() if isinstance(n, Apply) and n.arg_nodes]
    b_nodes = b.all_nodes()
    
    if a_applies and b_nodes:
        target = random.choice(a_applies)
        donor = random.choice(b_nodes)
        idx = random.randrange(len(target.arg_nodes))
        
        # Enforce max depth to prevent infinitely growing trees and deepcopy stack overflows
        if a.depth() + donor.depth() < 12:
            target.arg_nodes[idx] = copy.deepcopy(donor)
    
    return a
