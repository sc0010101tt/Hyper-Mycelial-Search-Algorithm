import networkx as nx
import matplotlib.pyplot as plt
from hmsa import OrganicMyceliumSearchEngine, HMSAConfig

def run_dormancy_test():
    print("--- EXPERIMENT 4: DORMANCY ---")
    G = nx.grid_2d_graph(20, 20)
    for u, v in G.edges(): G[u][v]['weight'] = 1.0
    nutrient_map = {n: 0.01 for n in G.nodes()} 
    config = [{'spores': [(10,10)], 'goals': []}]
    cfg = HMSAConfig(DORMANCY_THRESHOLD=0.5, MAX_ENERGY=20.0, DECAY_FACTOR=0.999, POOL_TAX_RATE=0.0, ENABLE_POOLING=False)

    print("Phase 1: Famine...")
    engine = OrganicMyceliumSearchEngine(G, config, nutrient_map=nutrient_map, cfg=cfg)
    engine.max_steps = 500
    engine.run_search()

    if len(engine.pq) > 0 and engine.get_statistics()['visited_nodes'] < 20:
        print(">> SUCCESS: Colony entered Hibernation.")
    else:
        print(f">> FAILURE: Nodes visited {engine.get_statistics()['visited_nodes']}")
        return

    print("Phase 2: The Rain...")
    engine.inject_environmental_change(new_nutrients={n: 1.0 for n in G.nodes()})
    engine.max_steps = 2000
    engine.status = "running" 
    engine.run_search()
    print(f"Final Nodes: {engine.get_statistics()['visited_nodes']}")

if __name__ == "__main__":
    run_dormancy_test()
