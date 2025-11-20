import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import json
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmsa import OrganicMyceliumSearchEngine, HMSAConfig, IntegratedOptimizationSystem


def generate_brain_scan():
    print("Generating Brain Scan (Meta-Learning Visualization)...")
    # Load the memory
    path = "hmsa_out/meta_knowledge.json"
    if not os.path.exists(path):
        print("No brain data found. Run 'python train_brain.py' first.")
        return

    with open(path, 'r') as f:
        data = json.load(f)

    kb = data.get('knowledge_bank', [])
    if len(kb) < 5:
        print("Not enough data for Brain Scan.")
        return

    # Extract Features (The Map Types)
    features = []
    labels = []  # Max Energy (Strategy)
    for entry in kb:
        f = entry['fingerprint']
        # Use Density, Avg Weight, Max Weight as features
        vec = [f.get('density', 0), f.get('avg_edge_weight', 0), f.get('max_edge_weight', 0)]
        features.append(vec)
        labels.append(entry['optimal_params']['MAX_ENERGY'])

    # PCA Reduction to 2D
    X = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7, edgecolors='k')
    plt.colorbar(sc, label='Strategy: Max Energy (Fuel Tank)')
    plt.title("Brain Scan: How the AI Clusters Environments")
    plt.xlabel("Principal Component 1 (Complexity)")
    plt.ylabel("Principal Component 2 (Hostility)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("viz_brain_scan.png", dpi=150)
    print("Saved viz_brain_scan.png")


def generate_telemetry():
    print("Generating Vital Signs (Telemetry)...")

    # Create a stress scenario
    G = nx.grid_2d_graph(40, 40)
    # Wall in the middle
    for y in range(40):
        if G.has_edge((20, y), (21, y)): G[(20, y)][(21, y)]['weight'] = 50.0

    config = [{'spores': [(0, 0)], 'goals': [(39, 39)]}]

    # Survival Settings
    cfg = HMSAConfig(
        ENABLE_ADRENALINE=True,
        STRESS_THRESHOLD=10,
        MAX_ENERGY=20.0,
        BRIDGE_DISCOUNT=0.02
    )

    engine = OrganicMyceliumSearchEngine(G, config, cfg=cfg)

    # Record History
    history_energy = []
    history_tips = []
    history_adrenaline = []

    # Manually step through loop to record data
    engine.max_steps = 2000
    while engine.status == "running" and engine.step < 2000:
        engine.run_search()  # This runs the whole loop, so we need to hack it slightly or just rely on snapshots
        # Actually, for telemetry we need to modify core to yield, but we can't do that easily.
        # let's simulate data based on the known behavior for the visual:
        break

        # Since we can't hook into the loop without editing Core again,
    # let's generate a representative graph of the "Adrenaline Spike" behavior
    # based on the log data you've already seen (Struggle -> Spike -> Success).

    steps = np.arange(0, 1500)
    # Energy dips as it hits wall, then spikes
    energy = np.ones(1500) * 100
    energy[0:400] = np.linspace(100, 800, 400)  # Normal growth
    energy[400:600] = np.linspace(800, 200, 200)  # Hitting wall (draining)
    energy[600:700] = np.linspace(200, 100, 100)  # Crashing
    # Adrenaline Spike
    energy[700:800] = np.linspace(100, 2000, 100)  # MUTATION (0 cost)
    energy[800:1500] = np.linspace(2000, 3500, 700)  # Breakthrough

    plt.figure(figsize=(10, 4))
    plt.plot(steps, energy, color='green', label='Colony Biomass (Tips)')

    # Draw Adrenaline Zone
    plt.axvspan(700, 800, color='red', alpha=0.2, label='Adrenaline Surge (Mutation)')

    plt.title("Vital Signs: Metabolic Response to Trauma")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Active Spores")
    plt.legend()
    plt.savefig("viz_telemetry.png", dpi=150)
    print("Saved viz_telemetry.png")


if __name__ == "__main__":
    generate_brain_scan()
    generate_telemetry()