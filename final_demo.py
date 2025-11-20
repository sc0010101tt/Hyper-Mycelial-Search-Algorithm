import networkx as nx
import matplotlib.pyplot as plt
import random
import time
from hmsa import IntegratedOptimizationSystem, OrganicMyceliumSearchEngine, HMSAConfig

def run_inference_demo():
    print("Loading the Trained Brain...")
    ios = IntegratedOptimizationSystem()
    print("Generating new 'Hard' scenario (Walls weight=50.0)...")

    size = 40
    G = nx.grid_2d_graph(size, size)
    for u, v in G.edges(): G[u][v]['weight'] = random.uniform(1.0, 1.5)

    walls = []
    for _ in range(15):
        x, y = random.randint(5, 30), random.randint(5, 30)
        for i in range(8): 
            if G.has_edge((x, y+i), (x+1, y+i)):
                G[(x, y+i)][(x+1, y+i)]['weight'] = 50.0 
                walls.append(((x, y+i), (x+1, y+i)))

    config = [{'spores': [(0, 0)], 'goals': [(39, 39)]}]

    print("Analysing topology and retrieving parameters...")
    start_time = time.time()
    rec_params = ios.meta_learner.recommend_parameters(G, config)

    print(f"\nBRAIN RECOMMENDATION:")
    for k, v in rec_params.items(): print(f"  {k}: {v:.4f}")

    print(f"\n--- Attempt 1: Running with Brain Parameters ---")
    cfg = HMSAConfig(**rec_params)
    cfg.ENABLE_BIDIR = True
    cfg.ENABLE_BRIDGING = True
    cfg.ENABLE_ADRENALINE = True # Self-Correction Enabled

    engine = OrganicMyceliumSearchEngine(G, config, cfg=cfg)
    status = engine.run_search()

    # Fallback (Manual Override)
    if status != "success":
        print(f"Result: {status.upper()} - Activating Manual Survival Mode...")
        cfg.DECAY_FACTOR = 0.999  
        cfg.POOL_TAX_RATE = 0.05      
        cfg.BRIDGE_ENERGY_COST = 0.1  
        cfg.BRIDGING_COST_THRESHOLD = 10.0 
        cfg.MAX_ENERGY = 20.0         
        cfg.BRIDGE_DISCOUNT = 0.01
        engine = OrganicMyceliumSearchEngine(G, config, cfg=cfg)
        status = engine.run_search()

    print(f"Final Result: {status.upper()}")
    print(f"Total Execution Time: {time.time() - start_time:.4f}s")
    print(f"Steps Taken: {engine.step}")

    pos = dict((n, n) for n in G.nodes())
    plt.figure(figsize=(12, 12))
    colors = []
    for n in G.nodes():
        idx = engine.node_to_idx[n]
        st = engine.visited_state[idx]
        if st == 2: colors.append('#FF0000')
        elif st == 1: colors.append('#90EE90')
        elif st == -1: colors.append('#ADD8E6')
        else: colors.append('#f8f8f8')
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color=colors)
    nx.draw_networkx_edges(G, pos, edgelist=walls, width=4.0, edge_color='black', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    run_inference_demo()
