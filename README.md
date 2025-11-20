# HMSA: Hyper-Mycelial Search Algorithm

**HMSA** is a bio-mimetic artificial intelligence that solves graph traversal problems by simulating a living fungal colony. It features real-time physiological adaptation (Adrenaline), evolutionary reinforcement learning, and meta-cognitive planning.

## Capabilities
1.  **Metabolism:** Tips consume energy to move. High-resistance terrain drains energy faster.
2.  **Homeostasis:** If resources are scarce (Famine), the colony enters stasis to conserve energy.
3.  **Phenotypic Plasticity:** If progress stalls, the system triggers an **Adrenaline Surge**, temporarily mutating its physics to bridge obstacles.
4.  **Evolutionary RL:** Individual tips mutate their own decision-making weights (`Pheromone` vs `Heuristic`) during the search.
5.  **Meta-Learning:** A Bayesian "Brain" analyzes map topology before the run to predict the optimal physiology (Energy Capacity, Decay Rate).

## Quick Start
1.  **Install:** `pip install -e .`
2.  **Train:** `python train_brain.py` (Teaches the AI how to handle Mazes/Walls)
3.  **Run:** `python final_demo.py` (Watch it solve an Extreme scenario)

## Experiments
* `python examples/exp_galapagos.py`: Proves divergent evolution.
* `python examples/exp_competition.py`: Proves territorial resource wars.
* `python examples/exp_injury.py`: Proves resilience to trauma.
* `python examples/exp_dormancy.py`: Proves metabolic homeostasis.
