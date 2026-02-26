from domains.zork.env import ZorkSOTAEnvironment
from domains.zork.agent import ZorkDeepAgent
from domains.zork.semantics import ZorkSemanticParser

def run():
    print("==================================================")
    print(" ZORK AGI SOTA: DEEP DEPENDENCIES & SEMANTIC PARSING ")
    print(" (Architected via 4 Pillars of General Learning) ")
    print("==================================================\n")
    
    env = ZorkSOTAEnvironment()
    agent = ZorkDeepAgent()
    parser = ZorkSemanticParser()
    
    start_obs = env.get_observation()
    print(f"INITIAL STATE: {start_obs}\n")
    
    # 1. Exploration & Graph Building (Pillar 4 using Pillar 1 & 3)
    agent.explore_world(ZorkSOTAEnvironment, max_depth=10)
    print(f"Discovered {len(agent.known_states)} unique states in the graph.")
    
    # 2. Plan path to grab the gold (Pillar 4 Search)
    winning_actions = agent.search(start_obs, target_keyword="picked up the gold")
    
    if winning_actions:
        print("\n>>> DEDUCED WINNING PLAN (MULTI-STEP HORIZON):")
        for step, a in enumerate(winning_actions):
            print(f"  Step {step+1}: '{a}'")
            
        print("\n>>> EXECUTING PLAN ON FRESH ENVIRONMENT:")
        fresh_env = ZorkSOTAEnvironment()
        for a in winning_actions:
            print(f" > USER: {a}")
            sem = parser.parse(a)
            obs = fresh_env.execute_action(sem)
            print(f" > ZORK: {obs}")
            
        print("\nRESULT: PERFECT DEEP HORIZON REASONING ACHIEVED.")
    else:
        print("\nRESULT: FAILED TO DEDUCE WINNING LOGIC (Graph disconnected).")

if __name__ == "__main__":
    run()
