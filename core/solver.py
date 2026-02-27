"""
Universal Solver — The 4-Pillar General Learning Loop

A single algorithm that solves ANY task by composing programs from a growing
primitive library, guided by a heuristic. Domain-agnostic: works for ARC,
Zork, or any environment that provides a feedback/heuristic interface.

The 4 Pillars:
  1. Feedback:        Environment provides observations and rewards
  2. Approximability: Heuristic evaluates candidate closeness to goal
  3. Abstraction:     PrimitiveLibrary composes programs from typed primitives
  4. Exploration:     Evolutionary search over the program space

After solving tasks, Library Learning compresses successful sub-programs
into new primitives, making the system smarter for all future tasks.
"""
import copy
import random
import time
import multiprocessing
import concurrent.futures
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

from core.program import Program, mutate_universal, crossover_universal
from core.library import PrimitiveLibrary


# Top-level function for multiprocessing (must be picklable)
def _evaluate_program(program, eval_fn):
    """Evaluate a single program using the provided evaluation function."""
    try:
        return eval_fn(program)
    except Exception:
        return float('inf')


class UniversalSolver:
    """
    The 4-Pillar General Learning Loop.
    
    Takes any evaluation function: program -> float (lower is better, 0.0 = solved).
    Uses evolutionary search with the primitive library to find a program
    that minimizes the evaluation score.
    """
    
    def __init__(self, library: PrimitiveLibrary):
        self.library = library
        self.solved_programs: List[Program] = []
        self.cpu_count = max(1, multiprocessing.cpu_count() - 1)
    
    def solve(self, eval_fn: Callable[[Program], float],
              output_type: str = 'Grid',
              beam_width: int = 200,
              max_generations: int = 50,
              verbose: bool = False) -> Tuple[Optional[Program], float]:
        """
        Unified solve loop for ANY domain.
        
        Args:
            eval_fn: Evaluates a program, returns float (0.0 = perfect)
            output_type: The type of program to synthesize
            beam_width: Number of candidates to maintain
            max_generations: Maximum evolutionary generations
            verbose: Log detailed progress
            
        Returns:
            (best_program, best_loss) tuple
        """
        print(f"[SOLVER] Universal Solver (beam={beam_width}, gens={max_generations}, "
              f"library={len(self.library.primitives)} primitives)")
        
        t0 = time.time()
        
        # 1. Seed population from library (3× beam for coverage)
        init_size = beam_width * 3
        candidates = [self.library.compose_random(output_type, max_depth=4)
                      for _ in range(init_size)]
        
        # Evaluate all candidates
        beam = [(eval_fn(p), p) for p in candidates]
        beam = [(s, p) for s, p in beam if s != float('inf')]
        if not beam:
            beam = [(float('inf'), candidates[0])]
        
        prev_best = float('inf')
        best_ever_loss = float('inf')
        best_ever_prog = beam[0][1]
        stale_gens = 0
        
        for gen in range(max_generations):
            beam.sort(key=lambda x: x[0])
            beam = beam[:beam_width]
            
            best_loss = beam[0][0]
            best_prog = beam[0][1]
            
            # Track best-ever (elitism)
            if best_loss < best_ever_loss:
                best_ever_loss = best_loss
                best_ever_prog = copy.deepcopy(best_prog)
            
            delta = prev_best - best_loss
            prev_best = best_loss
            
            # Perfect solution found
            if best_loss == 0.0:
                elapsed = time.time() - t0
                print(f"[SOLVER] ✅ Converged in gen {gen}! ({elapsed:.1f}s) | {best_prog}")
                self.solved_programs.append(copy.deepcopy(best_prog))
                return best_prog, 0.0
            
            # Track staleness
            if delta == 0.0:
                stale_gens += 1
            else:
                stale_gens = 0
            
            if verbose:
                print(f"[SOLVER] Gen {gen:3d} | Loss: {best_loss:.4f} "
                      f"(Δ={delta:+.4f}) stale={stale_gens} | {best_prog}")
            elif gen % 10 == 0:
                print(f"[SOLVER] Gen {gen} | Loss: {best_loss:.3f} | {best_prog}")
            
            # 2. Generate children
            n_children = int(beam_width * 1.5)
            new_candidates = []
            top_progs = [prog for _, prog in beam[:max(5, beam_width // 3)]]
            
            # Adaptive ratios
            random_ratio = 0.10 if stale_gens < 5 else 0.25
            
            for _ in range(n_children):
                r = random.random()
                if best_loss < 0.1 and r < 0.40:
                    # Targeted leaf mutation when close
                    parent = random.choice(top_progs)
                    new_candidates.append(mutate_universal(parent, self.library))
                elif r < 0.60:
                    # Structural mutation
                    parent = random.choice(top_progs)
                    new_candidates.append(mutate_universal(parent, self.library))
                elif r < (1.0 - random_ratio):
                    # Crossover
                    a = random.choice(top_progs)
                    b = random.choice(top_progs)
                    new_candidates.append(crossover_universal(a, b))
                else:
                    # Fresh random from library
                    new_candidates.append(
                        self.library.compose_random(output_type, max_depth=5))
            
            # Evaluate new candidates
            new_evals = [(eval_fn(p), p) for p in new_candidates]
            new_evals = [(s, p) for s, p in new_evals if s != float('inf')]
            beam.extend(new_evals)
        
        elapsed = time.time() - t0
        beam.sort(key=lambda x: x[0])
        final_loss = beam[0][0]
        final_prog = beam[0][1]
        
        # Return best-ever
        if best_ever_loss <= final_loss:
            final_loss = best_ever_loss
            final_prog = best_ever_prog
        
        if verbose:
            print(f"[SOLVER] Final: loss={final_loss:.4f} ({elapsed:.1f}s) | {final_prog}")
        
        return final_prog, final_loss
    
    def learn(self, verbose: bool = False) -> List[str]:
        """
        Library Learning: compress solved programs into new primitives.
        Call after solving a batch of tasks.
        
        Returns names of newly learned primitives.
        """
        if len(self.solved_programs) < 2:
            if verbose:
                print(f"[LEARN] Need ≥2 solved programs for learning (have {len(self.solved_programs)})")
            return []
        
        new_prims = self.library.compress(self.solved_programs)
        
        if verbose:
            if new_prims:
                print(f"[LEARN] Discovered {len(new_prims)} new primitives: {new_prims}")
                print(self.library.summary())
            else:
                print(f"[LEARN] No common sub-programs found across {len(self.solved_programs)} solutions")
        
        # Keep solved programs for future learning
        return new_prims
