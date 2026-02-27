"""
Zork Domain Adapter

Registers Zork navigation and interaction actions as primitives
in the universal PrimitiveLibrary.
"""
from core.library import PrimitiveLibrary


# Standard Zork directions and interactions
ZORK_ACTIONS = [
    'north', 'south', 'east', 'west', 'up', 'down',
    'northeast', 'northwest', 'southeast', 'southwest',
    'take all', 'look', 'open all', 'inventory', 'drop all',
    'open mailbox', 'take leaflet', 'read leaflet',
    'open door', 'open trapdoor', 'take sword', 'take lamp',
    'turn lamp on', 'move rug', 'go down',
]


def create_zork_library() -> PrimitiveLibrary:
    """Create a PrimitiveLibrary populated with Zork action primitives."""
    lib = PrimitiveLibrary()
    
    # Register the environment variable
    lib.register_variable('env', 'ZorkEnv')
    lib.register_variable('observation', 'Text')
    
    # Register each action as a primitive (Env -> Text)
    for action in ZORK_ACTIONS:
        safe_name = action.replace(' ', '_')
        lib.register(safe_name, _make_action_fn(action), ['ZorkEnv'], 'Text')
    
    return lib


def _make_action_fn(action_str):
    """Create a function that executes a specific Zork action."""
    def action_fn(env):
        return env.step_raw(action_str)
    action_fn.__name__ = action_str.replace(' ', '_')
    return action_fn


def make_zork_eval_fn(env, target_score=350):
    """
    Create an evaluation function for Zork.
    Returns a callable: program -> float (0.0 = max score achieved).
    
    For Zork, a 'program' is a sequence of actions. The eval function
    executes them and returns 1.0 - (score / target_score).
    """
    def eval_fn(program):
        state = env.get_state()
        try:
            # Execute the program (which calls env.step_raw internally)
            program.execute({'env': env, 'observation': env.get_observation()})
            score = env.get_score()
            env.load_state(state)
            return 1.0 - (score / target_score)
        except Exception:
            env.load_state(state)
            return 1.0
    
    return eval_fn
