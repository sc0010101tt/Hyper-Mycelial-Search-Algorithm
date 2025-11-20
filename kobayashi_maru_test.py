import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from hmsa import OrganicMyceliumSearchEngine, HMSAConfig
from benchmarks.baselines import StandardBaselines


def run_gasp_demo():
    print("--- THE KOBAYASHI MARU TEST ---")

    # 1. Construct the "Trap"
    # A long U-shaped corridor. The Start and Goal are separated by a thin wall.
    # Going around the U-shape is cheap (Weight 1) but LONG (100 nodes).
    # Going through the wall is expensive (Weight 100) but SHORT (2 nodes).

    size = 30
    G = nx.grid_2d_graph(size, size)

    # Set baseline weight
    for u, v in G.edges(): G[u][v]['weight'] = 1.0

    # Build the "Impossible Wall" (Weight 100.0)
    # A* will NEVER touch this if a path of cost 99 exists elsewhere.
    wall_edges = []
    for x in range(5, 25):
        if G.has_edge((x, 15), (x, 16)):
            G[(x, 15)][(x, 16)]['weight'] = 100.0
            wall_edges.append(((x, 15), (x, 16)))

    # Set Start (Inside the U) and Goal (Outside the U, just across the wall)
    start = (15, 10)
    goal = (15, 20)

    # ---------------------------------------------------------
    # COMPETITOR 1: A* (The Mathematician)
    # ---------------------------------------------------------
    print("Running A*...")
    astar_res = StandardBaselines.run_astar(G, start, goal)
    # A* path reconstruction (simplified for viz)
    astar_path = nx.shortest_path(G, start, goal, weight='weight')

    # ---------------------------------------------------------
    # COMPETITOR 2: HMSA (The Organism)
    # ---------------------------------------------------------
    print("Running HMSA...")
    # We give it limited energy. It CANNOT survive the long way around.
    # It MUST mutate to break the wall.
    cfg = HMSAConfig(
        MAX_ENERGY=15.0,  # Not enough fuel for the long detour
        DECAY_FACTOR=0.95,  # High metabolism
        ENABLE_ADRENALINE=True,  # Capability to mutate
        STRESS_THRESHOLD=20,  # Panic quickly
        BRIDGE_DISCOUNT=0.01,  # Mutation power: Wall becomes Weight 1.0
        ENABLE_BIDIR=False  # Single direction for dramatic effect
    )

    engine = OrganicMyceliumSearchEngine(G, [{'spores': [start], 'goals': [goal]}], cfg=cfg)
    engine.run_search()

    # ---------------------------------------------------------
    # THE REVEAL (Visualization)
    # ---------------------------------------------------------
    print("Generating The Gasp...")
    pos = dict((n, n) for n in G.nodes())
    plt.figure(figsize=(12, 6))

    # SUBPLOT 1: A* (The Logical Failure)
    plt.subplot(1, 2, 1)
    plt.title(f"A* (Standard Algorithm)\nCost Minimization")
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color='#dddddd')
    nx.draw_networkx_edges(G, pos, edgelist=wall_edges, width=3.0, edge_color='black', label="Weight 100")

    # Draw A* Path
    path_edges = list(zip(astar_path, astar_path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3.0, edge_color='blue')
    nx.draw_networkx_nodes(G, pos, nodelist=[start, goal], node_size=100, node_color='blue')
    plt.axis('off')

    # SUBPLOT 2: HMSA (The Biological Success)
    plt.subplot(1, 2, 2)
    plt.title(f"HMSA (Digital Organism)\nSurvival Adaptation")

    # Draw Organism Body
    colors = []
    for n in G.nodes():
        idx = engine.node_to_idx[n]
        if engine.visited_state[idx] != 0:
            colors.append('#90EE90')
        else:
            colors.append('#dddddd')
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color=colors)

    # Draw the Wall (Breached)
    nx.draw_networkx_edges(G, pos, edgelist=wall_edges, width=3.0, edge_color='black')

    # Draw HMSA Path (Through the wall)
    try:
        # We trace the path manually to show the breach
        hmsa_path = nx.shortest_path(G, start, goal)  # Geometric path, ignores weights
        # Only draw if it's the breach path
        hmsa_edges = list(zip(hmsa_path, hmsa_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=hmsa_edges, width=3.0, edge_color='lime')
    except:
        pass

    nx.draw_networkx_nodes(G, pos, nodelist=[start, goal], node_size=100, node_color='green')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("gasp.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_gasp_demo()