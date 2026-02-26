import multiprocessing
import random
import warnings
from functools import partial
from core.exploration import SearchAlgorithm
from domains.arc.dsl import ARCGrammar, mutate_program, crossover
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

import concurrent.futures

class ARCBeamSearch(SearchAlgorithm):
    """
    Pillar 4: Exploration.
    Evolutionary program synthesis using mutation, crossover, and heuristic-guided beam search.
    Traverses the compositional program space to converge on target tasks.
    """
    def __init__(self):
        self.grammar = ARCGrammar()
        self.heuristic = PixelEditDistance()
        self.cpu_count = max(1, multiprocessing.cpu_count() - 1)
        
    def _evaluate_program(self, program, examples):
        """Helper to evaluate a single program (used by tests)."""
        return evaluate_single_program(program, examples, self.heuristic)

    def _evaluate_batch(self, executor, programs, examples):
        """Evaluates a batch of programs using the provided executor."""
        eval_func = partial(evaluate_single_program, examples=examples, heuristic=self.heuristic)
        scores = list(executor.map(eval_func, programs))
        return list(zip(scores, programs))

    def search(self, start_state, target, beam_width=100, max_generations=30, verbose=False):
        print(f"[SYNTHESIS] Evolutionary Beam Search (beam={beam_width}, gens={max_generations})...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.cpu_count) as executor:
            # 1. Seed population with random programs
            candidates = [self.grammar.compose('Grid', max_depth=4) for _ in range(beam_width * 2)]
            beam = self._evaluate_batch(executor, candidates, start_state)
            
            prev_best = float('inf')
                
            for gen in range(max_generations):
                beam.sort(key=lambda x: x[0])
                beam = beam[:beam_width]
                
                best_loss = beam[0][0]
                delta = prev_best - best_loss
                prev_best = best_loss
                
                if best_loss == 0.0:
                    print(f"[SYNTHESIS] ✅ Converged in generation {gen}! Program: {beam[0][1]}")
                    return beam[0][1]
                
                if verbose:
                    print(f"[VERBOSE] Gen {gen:3d} | Loss: {best_loss:.4f} (Δ={delta:+.4f}) | Top 3:")
                    for rank, (score, prog) in enumerate(beam[:3]):
                        print(f"  [{rank+1}] {score:.4f} → {prog}")
                elif gen % 5 == 0:
                    print(f"Gen {gen} | Best Loss: {best_loss:.3f} | {beam[0][1]}")
                    
                # 2. Evolutionary generation of new candidates
                new_candidates = []
                top_progs = [prog for _, prog in beam[:max(5, beam_width // 4)]]
                n_mutate = n_cross = n_random = 0
                
                for _ in range(beam_width):
                    r = random.random()
                    if r < 0.5:
                        parent = random.choice(top_progs)
                        new_candidates.append(mutate_program(parent, self.grammar))
                        n_mutate += 1
                    elif r < 0.8:
                        parent_a = random.choice(top_progs)
                        parent_b = random.choice(top_progs)
                        new_candidates.append(crossover(parent_a, parent_b))
                        n_cross += 1
                    else:
                        new_candidates.append(self.grammar.compose('Grid', max_depth=5))
                        n_random += 1
                
                if verbose:
                    print(f"[VERBOSE]   Generated: {n_mutate} mutate, {n_cross} crossover, {n_random} random")
                        
                new_evals = self._evaluate_batch(executor, new_candidates, start_state)
                beam.extend(new_evals)
                
            beam.sort(key=lambda x: x[0])
            if verbose:
                print(f"[VERBOSE] Final: loss={beam[0][0]:.4f} | {beam[0][1]}")
            return beam[0][1]
