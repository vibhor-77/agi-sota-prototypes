from core.exploration import SearchAlgorithm
from domains.arc.dsl import ARCGrammar
from domains.arc.heuristics import PixelEditDistance

class ARCBeamSearch(SearchAlgorithm):
    """
    Pillar 4: Exploration.
    Systematically traverses the compositional program space using a Heuristic guide
    to converge on highly complex target tasks without brute-forcing exponential trees.
    """
    def __init__(self):
        self.grammar = ARCGrammar()
        self.heuristic = PixelEditDistance()
        
    def _evaluate_program(self, program, examples):
        total_dist = 0.0
        for inp, out in examples:
            try:
                pred = program.evaluate({'input_grid': inp})
                total_dist += self.heuristic.evaluate(pred, out)
            except Exception:
                total_dist += 10.0 # Huge penalty for runtime crashes (bounds, math errors)
        return total_dist / len(examples)

    def search(self, start_state, target, beam_width=50, max_generations=20):
        print(f"[SYNTHESIS] Running Beam Search exploration over program space...")
        beam = []
        
        # 1. Random Start
        for _ in range(beam_width * 2):
            candidate = self.grammar.compose('Grid', max_depth=4)
            score = self._evaluate_program(candidate, start_state)
            beam.append((score, candidate))
            
        for gen in range(max_generations):
            beam.sort(key=lambda x: x[0])
            beam = beam[:beam_width]
            
            if beam[0][0] == 0.0:
                print(f"[SYNTHESIS] Concept Converged automatically in generation {gen}")
                return beam[0][1]
                
            if gen % 5 == 0:
                print(f"Gen {gen} | Best Heuristic Loss: {beam[0][0]:.3f} | Best Program: {beam[0][1]}")
                
            # 2. Evolutionary Mutations
            new_beam = []
            for score, prog in beam:
                new_beam.append((score, prog))
                # Inject diversity
                for _ in range(2):
                    new_candidate = self.grammar.compose('Grid', max_depth=5)
                    new_score = self._evaluate_program(new_candidate, start_state)
                    new_beam.append((new_score, new_candidate))
            beam = new_beam
            
        beam.sort(key=lambda x: x[0])
        return beam[0][1]
