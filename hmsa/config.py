from dataclasses import dataclass

@dataclass
class HMSAConfig:
    # --- Core Physics ---
    DECAY_FACTOR: float = 0.995
    DORMANCY_THRESHOLD: float = 0.05
    MAX_ENERGY: float = 5.0
    MAX_TIPS: int = 20000

    # --- Graph & Weights ---
    USE_EDGE_WEIGHTS: bool = True
    DEFAULT_EDGE_WEIGHT: float = 1.0

    # --- Scents ---
    PHEROMONE_DECAY: float = 0.995
    PHEROMONE_WEIGHT: float = 0.2
    NEGSCENT_DECAY: float = 0.995
    NEGSCENT_WEIGHT: float = 1.0
    NUTRIENT_DEPLETION: float = 0.5

    # --- Heuristics ---
    GOAL_HEURISTIC_WEIGHT: float = 0.5

    # --- Colony Interaction ---
    ENABLE_POOLING: bool = True
    POOL_TAX_RATE: float = 0.05
    POOL_REVIVAL_COST: float = 2.0
    COMPETITION_PENALTY: float = 2.0

    # --- Advanced Growth ---
    ENABLE_BIDIR: bool = True
    ENABLE_BRIDGING: bool = True
    BRIDGING_COST_THRESHOLD: float = 3.0
    BRIDGE_ENERGY_COST: float = 1.5
    BRIDGE_DISCOUNT: float = 0.1 

    # --- Dynamic Morphology (Adrenaline) ---
    ENABLE_ADRENALINE: bool = False
    STRESS_THRESHOLD: int = 50      
    ADRENALINE_DURATION: int = 100  
    ADRENALINE_ENERGY_MULT: float = 2.0  

    # --- Evolutionary RL ---
    ENABLE_EVOLUTION: bool = False
    EVO_MUTATION_RATE: float = 0.1
    EVO_MUTATION_STRENGTH: float = 0.2

    # --- Misc ---
    SEED: int = 42
