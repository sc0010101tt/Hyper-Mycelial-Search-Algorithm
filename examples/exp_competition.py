import networkx as nx
import matplotlib.pyplot as plt
from hmsa import OrganicMyceliumSearchEngine, HMSAConfig

def run_competition():
    print("--- EXPERIMENT 2: COMPETITION ---")
    size = 50
    G = nx.grid_2d_graph(size, size)
    for u, v in G.edges(): G[u][v]['weight'] = 1.0

    config = [
        {'spores': [(0,0)], 'goals': [(size-1, size-1)]},
        {'spores': [(size-1, size-1)], 'goals': [(0,0)]}
    ]
    cfg = HMSAConfig(MAX_ENERGY=10.0, COMPETITION_PENALTY=10.0, NUTRIENT_DEPLETION=0.0, ENABLE_BIDIR=False)

    engine = OrganicMyceliumSearchEngine(G, config, cfg=cfg, max_steps=3000)
    engine.run_search() 

    pos = dict((n, n) for n in G.nodes())
    plt.figure(figsize=(10, 10))
    colors = []
    for i in range(len(G.nodes())):
        c = engine.node_colony_map[i]
        if c == 0: colors.append('#ff9999')
        elif c == 1: colors.append('#9999ff')
        else: colors.append('#f8f8f8')
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color=colors)
    plt.show()

if __name__ == "__main__":
    run_competition()
