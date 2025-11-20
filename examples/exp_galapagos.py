from hmsa import IntegratedOptimizationSystem, OrganicMyceliumSearchEngine, HMSAConfig
import networkx as nx
import random
import numpy as np

def get_swamp_map():
    G = nx.grid_2d_graph(30, 30)
    for u, v in G.edges(): G[u][v]['weight'] = random.uniform(1.0, 10.0)
    return G, [{'spores': [(0,0)], 'goals': [(29,29)]}]

def get_bunker_map():
    G = nx.grid_2d_graph(30, 30)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
        if random.random() < 0.1: G[u][v]['weight'] = 50.0
    return G, [{'spores': [(0,0)], 'goals': [(29,29)]}]

def run_galapagos():
    print("--- EXPERIMENT 1: GALAPAGOS ---")
    print("\n> Evolving Species A (Swamp)...")
    sys_A = IntegratedOptimizationSystem(base_dir="hmsa_galapagos_A")
    for _ in range(5): sys_A.solve_with_auto_tuning(*get_swamp_map())

    print("\n> Evolving Species B (Bunker)...")
    sys_B = IntegratedOptimizationSystem(base_dir="hmsa_galapagos_B")
    for _ in range(5): sys_B.solve_with_auto_tuning(*get_bunker_map())

    print("\n> CROSS-TESTING...")
    rec_A = sys_A.meta_learner.recommend_parameters(*get_bunker_map())
    print(f"Swamp Brain in Bunker: Energy {rec_A.get('MAX_ENERGY', 5.0):.1f}")
    rec_B = sys_B.meta_learner.recommend_parameters(*get_swamp_map())
    print(f"Bunker Brain in Swamp: Energy {rec_B.get('MAX_ENERGY', 5.0):.1f}")

if __name__ == "__main__":
    run_galapagos()
