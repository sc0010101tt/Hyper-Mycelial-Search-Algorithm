import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("hmsa.meta_learning")

class MyceliumMetaLearner:
    def __init__(self, knowledge_file: Optional[str] = None):
        self.knowledge_bank = []
        self.model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
        self.scaler_X = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        self.target_names = []

        self.param_bounds = {
            'DECAY_FACTOR': (0.90, 0.999), 'PHEROMONE_WEIGHT': (0.1, 2.0),
            'NEGSCENT_WEIGHT': (0.5, 3.0), 'GOAL_HEURISTIC_WEIGHT': (0.0, 1.0),
            'POOL_TAX_RATE': (0.01, 0.10), 'BRIDGE_ENERGY_COST': (0.01, 0.5),
            'COMPETITION_PENALTY': (1.0, 5.0), 'BRIDGING_COST_THRESHOLD': (2.0, 15.0),
            'MAX_ENERGY': (5.0, 50.0), 'BRIDGE_DISCOUNT': (0.01, 0.5)
        }

        if knowledge_file and os.path.exists(knowledge_file):
            self.load_knowledge(knowledge_file)

    def extract_graph_fingerprint(self, graph: nx.Graph) -> Dict[str, float]:
        n_nodes = graph.number_of_nodes()
        if n_nodes == 0: return {}
        weights = [d.get('weight', 1.0) for u, v, d in graph.edges(data=True)]
        if not weights: weights = [1.0]
        return {
            'n_nodes': float(n_nodes), 'density': nx.density(graph),
            'avg_degree': sum(dict(graph.degree()).values()) / n_nodes,
            'avg_edge_weight': float(np.mean(weights)),
            'max_edge_weight': float(np.max(weights)), 'std_edge_weight': float(np.std(weights))
        }

    def extract_problem_fingerprint(self, graph: nx.Graph, config: List) -> Dict[str, float]:
        fp = self.extract_graph_fingerprint(graph)
        s, g = config[0].get('spores', [(0,0)]), config[0].get('goals', [(0,0)])
        fp['goal_distance'] = float(np.linalg.norm(np.array(s[0]) - np.array(g[0])))
        if not self.feature_names: self.feature_names = list(fp.keys())
        return fp

    def add_knowledge(self, graph, config, params, perf):
        fp = self.extract_problem_fingerprint(graph, config)
        self.knowledge_bank.append({'fingerprint': fp, 'optimal_params': params.copy(), 'performance': perf.copy()})
        self._update_model()
        return True

    def _update_model(self):
        if len(self.knowledge_bank) < 3: return
        try:
            X = np.array([[e['fingerprint'].get(k, 0.0) for k in self.feature_names] for e in self.knowledge_bank])
            if not self.target_names: self.target_names = list(self.knowledge_bank[0]['optimal_params'].keys())
            Y = np.array([[e['optimal_params'].get(k, 0.0) for k in self.target_names] for e in self.knowledge_bank])

            self.model.fit(self.scaler_X.fit_transform(X), Y)
            self.is_fitted = True
            logger.info(f"Meta-Learner Retrained on {len(X)} samples.")
        except Exception as e: logger.error(f"Model update failed: {e}")

    def recommend_parameters(self, graph, config, fallback_params=None):
        if not self.is_fitted: return fallback_params or {}
        try:
            fp = self.extract_problem_fingerprint(graph, config)
            vec = self.scaler_X.transform(np.array([[fp.get(k, 0.0) for k in self.feature_names]]))
            pred = self.model.predict(vec)[0]

            rec = {}
            for i, name in enumerate(self.target_names):
                val = float(pred[i])
                if name in self.param_bounds:
                    val = max(self.param_bounds[name][0], min(val, self.param_bounds[name][1]))
                rec[name] = val
            return rec
        except: return fallback_params or {}

    def save_knowledge(self, path):
        with open(path, 'w') as f: json.dump({'knowledge_bank': self.knowledge_bank, 'feature_names': self.feature_names, 'target_names': self.target_names}, f, indent=2)

    def load_knowledge(self, path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.knowledge_bank = data.get('knowledge_bank', [])
                self.feature_names = data.get('feature_names', [])
                self.target_names = data.get('target_names', [])
            if len(self.knowledge_bank) >= 3: self._update_model()
        except: pass
