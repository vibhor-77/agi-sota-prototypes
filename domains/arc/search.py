import multiprocessing
import warnings
from functools import partial
from core.exploration import SearchAlgorithm
from domains.arc.dsl import ARCGrammar
from domains.arc.heuristics import PixelEditDistance

# Suppress pervasive ResourceWarnings from multiprocessing cleanup on some Python versions
warnings.filterwarnings("ignore", category=ResourceWarning)

# Top-level helper for multiprocessing map
def evaluate_single_program(program, examples, heuristic):
    total_dist = 0.0
    for inp, out in examples:
        try:
            pred = program.evaluate({'input_grid': inp})
            total_dist += heuristic.evaluate(pred, out)
        except Exception:
            total_dist += 10.0 # Penalty for runtime crashes (bounds, math errors)
    return total_dist / len(examples)

class ARCBeamSearch(SearchAlgorithm):
    """
    Pillar 4: Exploration.
    Systematically traverses the compositional program space using a Heuristic guide
    to converge on highly complex target tasks without brute-forcing exponential trees.
    Now utilizes Multi-Core Processing to aggressively evaluate deep DSL trees in parallel.
    """
    def __init__(self):
        self.grammar = ARCGrammar()
        self.heuristic = PixelEditDistance()
        self.cpu_count = max(1, multiprocessing.cpu_count() - 1)
        
    def _evaluate_program(self, program, examples):
        """Helper to evaluate a single program (used by tests)."""
        return evaluate_single_program(program, examples, self.heuristic)
        
    def _evaluate_batch(self, pool, programs, examples):
        """Evaluates a batch of programs using a provided pool."""
        # Bind the examples and heuristic to the function
        eval_func = partial(evaluate_single_program, examples=examples, heuristic=self.heuristic)
        
        # Parallel evaluate the beam array
        scores = pool.map(eval_func, programs)
        return list(zip(scores, programs))

    def search(self, start_state, target, beam_width=50, max_generations=20):
        print(f"[SYNTHESIS] Running Beam Search exploration over program space...")
        
        # Initialize a single pool for the entire search session to avoid repeated overhead and resource warnings
        pool = multiprocessing.Pool(processes=self.cpu_count)
        
        try:
            # 1. Random Start Generation
            candidates = []
            for _ in range(beam_width * 2):
                candidates.append(self.grammar.compose('Grid', max_depth=4))
                
            beam = self._evaluate_batch(pool, candidates, start_state)
                
            for gen in range(max_generations):
                beam.sort(key=lambda x: x[0])
                beam = beam[:beam_width]
                
                if beam[0][0] == 0.0:
                    print(f"[SYNTHESIS] Concept Converged automatically in generation {gen}")
                    return beam[0][1]
                    
                if gen % 5 == 0:
                    print(f"Gen {gen} | Best Heuristic Loss: {beam[0][0]:.3f} | Best Program: {beam[0][1]}")
                    
                # 2. Evolutionary Mutations
                new_candidates = []
                for score, prog in beam:
                    # Inject diversity
                    for _ in range(2):
                        new_candidates.append(self.grammar.compose('Grid', max_depth=5))
                        
                new_evals = self._evaluate_batch(pool, new_candidates, start_state)
                beam.extend(new_evals)
                
            beam.sort(key=lambda x: x[0])
            return beam[0][1]
        finally:
            pool.close()
            pool.join()
