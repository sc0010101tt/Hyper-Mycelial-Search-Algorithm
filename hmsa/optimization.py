import numpy as np
import logging
import random
import warnings
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import MinMaxScaler

from .core import OrganicMyceliumSearchEngine
from .config import HMSAConfig

logger = logging.getLogger("hmsa.hyperopt")

class HyperparameterOptimizer:
    def __init__(
            self,
            problem_instances: List[Tuple[nx.Graph, List[Dict]]],
            n_trials: int = 30,
            n_iterations_per_trial: int = 3,
            meta_learning: bool = True,
            max_steps_per_trial: int = 2000,
            scoring_weights: Optional[Dict[str, float]] = None
    ):
        self.problem_instances = problem_instances
        self.n_trials = n_trials
        self.n_iterations_per_trial = max(1, n_iterations_per_trial // 2)
        self.max_steps_per_trial = max_steps_per_trial
        self.scoring_weights = scoring_weights or {'success_rate': 0.6}

        self.param_bounds = {
            'DECAY_FACTOR': (0.90, 0.999),
            'PHEROMONE_WEIGHT': (0.1, 2.0),
            'NEGSCENT_WEIGHT': (0.5, 3.0),
            'GOAL_HEURISTIC_WEIGHT': (0.0, 1.0),
            'POOL_TAX_RATE': (0.01, 0.10),
            'BRIDGE_ENERGY_COST': (0.01, 0.5),
            'COMPETITION_PENALTY': (1.0, 5.0),
            'BRIDGING_COST_THRESHOLD': (2.0, 15.0),
            'MAX_ENERGY': (5.0, 50.0),
            'BRIDGE_DISCOUNT': (0.01, 0.5),
            'EVO_MUTATION_RATE': (0.01, 0.2), 
            'EVO_MUTATION_STRENGTH': (0.05, 0.5)
        }
        self.param_names = list(self.param_bounds.keys())
        self.trial_history = []
        self.best_params = None
        self.best_score = float('-inf')
        self.scaler = MinMaxScaler()
        self.gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True, n_restarts_optimizer=5)
        self.X_history = []
        self.y_history = []

    def _vector_to_dict(self, vector):
        return {k: float(v) for k, v in zip(self.param_names, vector)}

    def _suggest_params(self, trial_idx):
        if trial_idx < 5 or len(self.X_history) < 5:
            return {k: random.uniform(v[0], v[1]) for k, v in self.param_bounds.items()}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_train = self.scaler.fit_transform(np.array(self.X_history))
            y_train = np.array(self.y_history)
            self.gp.fit(X_train, y_train)

        n_samples = 2000
        X_random = np.random.uniform(0, 1, (n_samples, len(self.param_names)))
        mu, sigma = self.gp.predict(X_random, return_std=True)
        mu_sample_opt = np.max(y_train)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        best_idx = np.argmax(ei)
        return self._vector_to_dict(self.scaler.inverse_transform([X_random[best_idx]])[0])

    def _evaluate_params(self, params):
        scores = []
        clean = {k: max(v[0], min(params[k], v[1])) for k, v in self.param_bounds.items()}

        base_config = HMSAConfig(
            ENABLE_BRIDGING=True, ENABLE_BIDIR=True, 
            ENABLE_POOLING=True, ENABLE_EVOLUTION=True, 
            **clean
        )

        batch = random.sample(self.problem_instances, 3) if len(self.problem_instances) > 3 else self.problem_instances

        for graph, cfg_data in batch:
            inst_scores = []
            for _ in range(self.n_iterations_per_trial):
                try:
                    eng = OrganicMyceliumSearchEngine(graph, cfg_data, cfg=base_config, max_steps=self.max_steps_per_trial)
                    eng.run_search()
                    stats = eng.get_statistics()

                    found = sum(d['found_goals'] for d in stats['colonies'].values())
                    total = sum(len(c.get('goals', [])) for c in cfg_data)
                    sr = found/total if total > 0 else 0

                    visited = stats.get('visited_nodes', 0)
                    total_n = stats.get('total_nodes', 1)
                    eff = 1.0 - (visited / total_n)
                    if sr < 1.0: eff = 0.0

                    steps_norm = 1.0 - (stats['steps'] / self.max_steps_per_trial)
                    score = (sr * 100.0) + (steps_norm * 20.0) + (eff * 10.0)
                    inst_scores.append(score)
                except:
                    inst_scores.append(0.0)
            if inst_scores: scores.append(np.mean(inst_scores))

        if not scores: return 0.0
        # Robustness Metric: Mean - 0.5 * StdDev
        return float(np.mean(scores) - 0.5 * np.std(scores))

    def optimize(self):
        logger.info(f"Starting Robust Bayesian Optimization: {self.n_trials} trials")
        for t in range(self.n_trials):
            params = self._suggest_params(t)
            score = self._evaluate_params(params)

            self.X_history.append([params[k] for k in self.param_names])
            self.y_history.append(score)
            self.trial_history.append({'trial': t, 'params': params, 'score': score})

            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                logger.info(f"Trial {t} [New Best]: {score:.2f}")
            else:
                logger.info(f"Trial {t}: {score:.2f}")

        return self.best_params
