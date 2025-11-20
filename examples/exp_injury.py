import networkx as nx
import matplotlib.pyplot as plt
from hmsa import OrganicMyceliumSearchEngine, HMSAConfig

def run_injury_test():
    print("--- EXPERIMENT 3: INJURY ---")
    G = nx.grid_2d_graph(30, 30)
    for u, v in G.edges(): G[u][v]['weight'] = 1.0
    config = [{'spores': [(0,0)], 'goals': [(29,29)]}]
    cfg = HMSAConfig(ENABLE_ADRENALINE=True, STRESS_THRESHOLD=10, MAX_ENERGY=10.0, ENABLE_BIDIR=False, BRIDGE_DISCOUNT=0.1)

    engine = OrganicMyceliumSearchEngine(G, config, cfg=cfg, max_steps=5000)
    print("Phase 1: Initial Growth...")
    engine.max_steps = 300
    engine.run_search()

    print("!!! TRAUMA EVENT !!!")
    injury_updates = []
    for y in range(30):
        if G.has_edge((15, y), (16, y)): injury_updates.append(((15, y), (16, y), 1000.0))
    engine.inject_environmental_change(new_weights=injury_updates)

    print("Phase 2: Healing Response...")
    engine.max_steps = 2500
    engine.status = "running" 
    engine.run_search()

    if engine.status == "success": print(">> SUCCESS: Organism healed.")
    pos = dict((n, n) for n in G.nodes())
    plt.figure(figsize=(10, 10))
    colors = ['#90EE90' if engine.visited_state[i] != 0 else '#f0f0f0' for i in range(len(G.nodes()))]
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color=colors)
    nx.draw_networkx_edges(G, pos, edgelist=[(u,v) for u,v,w in injury_updates], width=4.0, edge_color='red')
    plt.show()

if __name__ == "__main__":
    run_injury_test()
