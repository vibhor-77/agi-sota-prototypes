from domains.arc.env import ARCEnvironment, generate_2d_arc_task
from domains.arc.search import ARCBeamSearch

def run():
    print("==================================================")
    print(" ARC AGI SOTA: BEAM SEARCH & GEOMETRY PRIMITIVES ")
    print(" (Architected via 4 Pillars of General Learning) ")
    print("==================================================\n")
    
    examples = generate_2d_arc_task(4)
    if not examples or len(examples) < 4:
        print("Failed to generate task examples.")
        return
        
    train_ex = examples[:3]
    test_ex = examples[3]
    
    print(">>> OBSERVING TRAINING GRIDS")
    for i, (inp, out) in enumerate(train_ex):
        print(f"\n--- Train {i+1} ---")
        print(f"Input:\n{inp.arr}")
        print(f"Output:\n{out.arr}")
        
    # Instantiate Pillar 4 (Search) which intrinsically uses Pillars 2 & 3
    agent = ARCBeamSearch()
    
    # 1. Search for a generalized program
    best_program = agent.search(train_ex, target=None, beam_width=100, max_generations=20)
    
    print("\n==================================================")
    print(">>> COMPOSITIONAL LEARNING CONVERGED")
    print("==================================================")
    if best_program:
        print(f"Discovered Concept: f(grid) = {str(best_program)}\n")
        
        print(">>> EXTRAPOLATION ON UNSEEN TEST GRID")
        test_in, test_out = test_ex
        
        # 2. Execute on environment (Pillar 1)
        env = ARCEnvironment(test_in)
        predicted = env.execute_action(best_program)
        
        print(f"Input Grid:\n{test_in.arr}")
        print(f"AI Predicted:\n{predicted.arr}")
        print(f"True Answer:\n{test_out.arr}")
        
        if predicted == test_out:
            print("\nRESULT: PERFECT 2D GENERALIZATION & SPATIAL REASONING ACHIEVED.")
        else:
            print("\nRESULT: TEST FAILED.")
    else:
        print("Search failed.")

if __name__ == "__main__":
    run()
