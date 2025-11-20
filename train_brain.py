from hmsa.integration import IntegratedOptimizationSystem
import time
import random

def train_meta_learner():
    print("Initializing Artificial Training Loop...")
    ios = IntegratedOptimizationSystem()
    scenarios = ["medium", "hard", "maze", "extreme"] 
    try:
        for i in range(100): 
            scenario = scenarios[i % 4]
            print(f"\n[{i+1}/100] Generating Scenario: {scenario.upper()}")
            if scenario == "extreme":
                G, cfg = ios.create_test_problem("hard") 
                for u, v in G.edges():
                    if G[u][v]['weight'] > 5.0: G[u][v]['weight'] = 50.0
            else:
                G, cfg = ios.create_test_problem(scenario)

            start = time.time()
            # IOS handles the Fail -> Optimize -> Learn loop
            ios.solve_with_auto_tuning(G, cfg)
            print(f"Scenario resolved in {time.time() - start:.2f}s")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Knowledge bank saved.")

if __name__ == "__main__":
    train_meta_learner()
