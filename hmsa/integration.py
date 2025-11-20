import os
import json
import random
import logging
import networkx as nx
from .core import OrganicMyceliumSearchEngine
from .config import HMSAConfig
from .optimization import HyperparameterOptimizer
from .meta_learning import MyceliumMetaLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hmsa.integration")

class IntegratedOptimizationSystem:
    def __init__(self, base_dir="hmsa_out"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.meta_path = os.path.join(self.base_dir, 'meta_knowledge.json')
        self.meta_learner = MyceliumMetaLearner(knowledge_file=self.meta_path)

    def solve_with_auto_tuning(self, graph: nx.Graph, config: list):
        logger.info("Analyzing Graph Topology...")
        rec_params = self.meta_learner.recommend_parameters(graph, config)

        if not rec_params: logger.info("No prior knowledge found. Using Defaults.")
        else: logger.info(f"Applying learned parameters: {rec_params}")

        cfg = HMSAConfig(**rec_params) if rec_params else HMSAConfig()
        cfg.ENABLE_BIDIR = True
        cfg.ENABLE_BRIDGING = True
        cfg.ENABLE_ADRENALINE = True

        logger.info("Running Search with recommended parameters...")
        engine = OrganicMyceliumSearchEngine(graph, config, cfg=cfg)
        engine.run_search()

        if engine.status == "success":
            logger.info(f"Success! Steps: {engine.step}")
            return engine.get_statistics()

        logger.warning("Initial attempt failed. Triggering Hyper-Optimization...")
        opt = HyperparameterOptimizer([(graph, config)], n_trials=15, n_iterations_per_trial=1)
        best_params = opt.optimize()

        self.meta_learner.add_knowledge(graph, config, best_params, {'score': opt.best_score})
        self.meta_learner.save_knowledge(self.meta_path)
        logger.info("New knowledge assimilated into Meta-Learner.")
        return best_params

    def create_test_problem(self, difficulty="medium"):
        size = 40
        G = nx.grid_2d_graph(size, size)
        for u, v in G.edges(): G[u][v]['weight'] = random.uniform(1.0, 1.5)

        if difficulty == "hard":
            for _ in range(20):
                x, y = random.randint(5, 35), random.randint(5, 35)
                for i in range(10):
                    if G.has_edge((x, y+i), (x+1, y+i)): G[(x, y+i)][(x+1, y+i)]['weight'] = 20.0
        elif difficulty == "maze":
            edges = list(G.edges())
            random.shuffle(edges)
            G.remove_edges_from(edges[:int(len(edges)*0.3)])

        config = [{'spores': [(0, 0)], 'goals': [(size-1, size-1)]}]
        return G, config
