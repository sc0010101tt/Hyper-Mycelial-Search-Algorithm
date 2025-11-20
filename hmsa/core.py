import numpy as np
import networkx as nx
import heapq
import logging
import random
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict
from numba import njit
from .config import HMSAConfig

logger = logging.getLogger("hmsa_opt")

@njit(fastmath=True)
def _calc_score_jit(energy, nutrient, pheromone, neg_scent, heuristic_val, edge_weight, ph_w, ns_w, goal_w):
    # Score = Organic Potential * Structural Potential
    organic = energy * (1.0 + nutrient) * (1.0 + ph_w * pheromone) / (1.0 + ns_w * neg_scent)
    structural = (1.0 + goal_w / (1.0 + heuristic_val)) / edge_weight
    return organic * structural

@njit(fastmath=True)
def _calculate_decay_jit(current_energy, decay_factor, edge_weight, competition_penalty, nutrient_val):
    # Energy New = Energy Old * Decay * Nutrient / Resistance
    resistance = edge_weight * competition_penalty
    # Avoid division by zero if weight is 0 (e.g. during adrenaline surge)
    if resistance < 0.0001: resistance = 0.0001
    return current_energy * decay_factor * nutrient_val / resistance

class OrganicMyceliumSearchEngine:
    def __init__(self, graph: nx.Graph, spore_goals_config: List[Dict], nutrient_map: Dict = None, cfg: HMSAConfig = HMSAConfig(), **kwargs):
        self.cfg = cfg
        self.graph = graph
        self.initial_config = spore_goals_config

        # Determinism
        np.random.seed(self.cfg.SEED)
        random.seed(self.cfg.SEED)

        # Graph Mapping
        self.nodes = list(graph.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)

        # Fast Adjacency
        self.adj = []
        self.weights = []
        use_weights = self.cfg.USE_EDGE_WEIGHTS

        for n in self.nodes:
            nbrs_list = []
            w_list = []
            for nbr in graph.neighbors(n):
                nbrs_list.append(self.node_to_idx[nbr])
                if use_weights:
                    # Look for 'weight' or 'cost'
                    w = graph[n][nbr].get('weight', graph[n][nbr].get('cost', self.cfg.DEFAULT_EDGE_WEIGHT))
                    w_list.append(float(w))
                else:
                    w_list.append(1.0)
            self.adj.append(nbrs_list)
            self.weights.append(w_list)

        # Biological State Arrays
        self.pheromone = np.full(self.num_nodes, 0.01, dtype=np.float32)
        self.neg_scent = np.zeros(self.num_nodes, dtype=np.float32)
        self.nutrient = np.ones(self.num_nodes, dtype=np.float32)

        if nutrient_map:
            for n, val in nutrient_map.items():
                if n in self.node_to_idx:
                    self.nutrient[self.node_to_idx[n]] = float(val)

        self.visited_state = np.zeros(self.num_nodes, dtype=np.int8) 
        self.node_colony_map = np.full(self.num_nodes, -1, dtype=np.int32)
        self.heuristic_map = self._precalculate_heuristics(spore_goals_config)

        # Tip Buffers (Struct of Arrays)
        buffer_size = cfg.MAX_TIPS + 5000
        self.tip_nodes = np.zeros(buffer_size, dtype=np.int32)
        self.tip_parents = np.full(buffer_size, -1, dtype=np.int32)
        self.tip_energies = np.zeros(buffer_size, dtype=np.float32)
        self.tip_colonies = np.zeros(buffer_size, dtype=np.int32)
        self.tip_directions = np.zeros(buffer_size, dtype=np.int8)

        # Tip Genetics
        self.tip_w_ph = np.full(buffer_size, cfg.PHEROMONE_WEIGHT, dtype=np.float32)
        self.tip_w_ns = np.full(buffer_size, cfg.NEGSCENT_WEIGHT, dtype=np.float32)
        self.tip_w_goal = np.full(buffer_size, cfg.GOAL_HEURISTIC_WEIGHT, dtype=np.float32)

        self.free_indices = list(range(buffer_size - 1, -1, -1))
        self.pq = []

        self.colony_banks = defaultdict(float) 

        # Goal Management
        self.goals_flat = []
        self.found_goals_indices = set()
        self.colony_goals_indices = defaultdict(set)
        self.predecessors = {} 

        self._initialize_tips(spore_goals_config)

        self.step = 0
        self.status = "running"
        self.max_steps = kwargs.get('max_steps', 20000)

        # Dynamic Morphology State
        self.adrenaline_active = False
        self.adrenaline_timer = 0
        self.stress_counter = 0
        self.last_visit_count = 0

    def _precalculate_heuristics(self, config):
        h_map = np.full(self.num_nodes, 9999.0, dtype=np.float32)
        all_goals = []
        for c_data in config:
            for g in c_data.get('goals', []):
                if g in self.node_to_idx:
                    all_goals.append(g)

        if not all_goals: return h_map

        try:
            # Multi-source Dijkstra for heuristic map
            dists = nx.multi_source_dijkstra_path_length(self.graph, all_goals, weight='weight')
            for n, d in dists.items():
                if n in self.node_to_idx:
                    h_map[self.node_to_idx[n]] = d
        except:
            h_map.fill(0.0)
        return h_map

    def _initialize_tips(self, config):
        for cid, c_data in enumerate(config):
            for g in c_data.get('goals', []):
                if g in self.node_to_idx:
                    idx = self.node_to_idx[g]
                    self.colony_goals_indices[cid].add(idx)
                    self.goals_flat.append(idx)
                    # Bidirectional: Spawn Backward Tip at Goal
                    if self.cfg.ENABLE_BIDIR:
                        self._spawn_tip_internal(idx, -1, self.cfg.MAX_ENERGY, cid, -1)

            for spore in c_data.get('spores', []):
                if spore in self.node_to_idx:
                    self._spawn_tip_internal(self.node_to_idx[spore], -1, self.cfg.MAX_ENERGY, cid, 1)

    def _spawn_tip_internal(self, node_idx, parent_idx, energy, colony, direction):
        if not self.free_indices: return None
        idx = self.free_indices.pop()

        # Genetics: Inheritance and Mutation
        w_ph = self.cfg.PHEROMONE_WEIGHT
        w_ns = self.cfg.NEGSCENT_WEIGHT
        w_goal = self.cfg.GOAL_HEURISTIC_WEIGHT

        if parent_idx != -1:
            w_ph = self.tip_w_ph[parent_idx]
            w_ns = self.tip_w_ns[parent_idx]
            w_goal = self.tip_w_goal[parent_idx]

            if self.cfg.ENABLE_EVOLUTION and random.random() < self.cfg.EVO_MUTATION_RATE:
                gene = random.choice([0, 1, 2])
                change = random.uniform(-self.cfg.EVO_MUTATION_STRENGTH, self.cfg.EVO_MUTATION_STRENGTH)
                if gene == 0: w_ph = max(0.0, w_ph + change)
                elif gene == 1: w_ns = max(0.0, w_ns + change)
                elif gene == 2: w_goal = max(0.0, w_goal + change)

        self.tip_w_ph[idx] = w_ph
        self.tip_w_ns[idx] = w_ns
        self.tip_w_goal[idx] = w_goal

        sc = _calc_score_jit(
            energy, self.nutrient[node_idx], self.pheromone[node_idx],
            self.neg_scent[node_idx], self.heuristic_map[node_idx], 1.0,
            w_ph, w_ns, w_goal 
        )

        self.tip_nodes[idx] = node_idx
        self.tip_parents[idx] = parent_idx
        self.tip_energies[idx] = energy
        self.tip_colonies[idx] = colony
        self.tip_directions[idx] = direction

        if parent_idx != -1:
            self.predecessors[(node_idx, direction)] = self.tip_nodes[parent_idx]
        else:
            self.predecessors[(node_idx, direction)] = None

        heapq.heappush(self.pq, (-sc, idx))
        return idx

    def inject_environmental_change(self, new_weights=None, new_nutrients=None):
        if new_weights:
            for u, v, w in new_weights:
                if u in self.node_to_idx and v in self.node_to_idx:
                    u_idx = self.node_to_idx[u]
                    v_idx = self.node_to_idx[v]
                    try:
                        idx = self.adj[u_idx].index(v_idx)
                        self.weights[u_idx][idx] = float(w)
                    except ValueError: pass
                    try:
                        idx = self.adj[v_idx].index(u_idx)
                        self.weights[v_idx][idx] = float(w)
                    except ValueError: pass

        if new_nutrients:
            for node, val in new_nutrients.items():
                if node in self.node_to_idx:
                    idx = self.node_to_idx[node]
                    self.nutrient[idx] = float(val)

    def run_search(self) -> str:
        while self.pq and self.step < self.max_steps:

            # 1. Adrenaline Monitor
            if self.cfg.ENABLE_ADRENALINE and self.step % 10 == 0:
                current_visits = np.sum(self.visited_state != 0)
                progress = current_visits - self.last_visit_count
                self.last_visit_count = current_visits

                if progress == 0:
                    self.stress_counter += 10
                else:
                    self.stress_counter = max(0, self.stress_counter - 1)

                if self.stress_counter > self.cfg.STRESS_THRESHOLD and not self.adrenaline_active:
                    self.adrenaline_active = True
                    self.adrenaline_timer = self.cfg.ADRENALINE_DURATION
                    # logger.info(f"!!! ADRENALINE SURGE AT STEP {self.step} !!!")

            # 2. Set Dynamic Physiology
            curr_decay = self.cfg.DECAY_FACTOR
            curr_discount = self.cfg.BRIDGE_DISCOUNT
            curr_bridge_cost = self.cfg.BRIDGE_ENERGY_COST
            curr_max_energy = self.cfg.MAX_ENERGY

            if self.adrenaline_active:
                curr_decay = 1.0
                curr_discount = 0.01
                curr_bridge_cost = 0.0
                curr_max_energy *= self.cfg.ADRENALINE_ENERGY_MULT
                self.adrenaline_timer -= 1
                if self.adrenaline_timer <= 0:
                    self.adrenaline_active = False
                    self.stress_counter = 0

            # 3. Process Tip
            _, tip_idx = heapq.heappop(self.pq)

            if self.tip_energies[tip_idx] <= 0:
                self.free_indices.append(tip_idx)
                continue

            curr_node = self.tip_nodes[tip_idx]
            curr_colony = self.tip_colonies[tip_idx]
            curr_dir = self.tip_directions[tip_idx]
            curr_energy = self.tip_energies[tip_idx]

            # Pooling Tax
            if self.cfg.ENABLE_POOLING:
                tax = curr_energy * self.cfg.POOL_TAX_RATE
                self.colony_banks[curr_colony] += tax
                curr_energy -= tax
                self.tip_energies[tip_idx] = curr_energy

            # Visit & Collision
            existing_visit_state = self.visited_state[curr_node]
            collision = False
            if existing_visit_state != 0:
                prev_colony = self.node_colony_map[curr_node]
                if prev_colony == curr_colony:
                    # Intra-colony meeting (Bidirectional)
                    if existing_visit_state != curr_dir and existing_visit_state != 2:
                        self._register_meeting(curr_node, curr_colony)
                        self.visited_state[curr_node] = 2 
                        collision = True 
                else:
                    # Inter-colony competition
                    curr_energy /= self.cfg.COMPETITION_PENALTY

            if collision:
                self.free_indices.append(tip_idx)
                continue

            # Mark Visited
            if self.visited_state[curr_node] == 0:
                self.visited_state[curr_node] = curr_dir

            self.node_colony_map[curr_node] = curr_colony
            self.nutrient[curr_node] *= self.cfg.NUTRIENT_DEPLETION

            # Check Goals
            if curr_dir == 1 and curr_node in self.colony_goals_indices[curr_colony]:
                if curr_node not in self.found_goals_indices:
                    self.found_goals_indices.add(curr_node)

            # Stop Condition
            if self.goals_flat and len(self.found_goals_indices) == len(self.goals_flat):
                self.status = "success"
                break

            # Expand
            neighbors = self.adj[curr_node]
            weights = self.weights[curr_node]
            branches_spawned = 0
            any_child_spawned = False

            for i in range(len(neighbors)):
                nb_idx = neighbors[i]
                w = weights[i]

                if branches_spawned >= 2: break
                if self.visited_state[nb_idx] == curr_dir: continue

                # Bridging Logic
                cost_modifier = 1.0
                if self.cfg.ENABLE_BRIDGING and w > self.cfg.BRIDGING_COST_THRESHOLD:
                    if self.colony_banks[curr_colony] > curr_bridge_cost:
                        self.colony_banks[curr_colony] -= curr_bridge_cost
                        cost_modifier = curr_discount 

                eff_weight = w * cost_modifier
                comp_penalty = 1.0
                if self.node_colony_map[nb_idx] != -1 and self.node_colony_map[nb_idx] != curr_colony:
                    comp_penalty = self.cfg.COMPETITION_PENALTY

                # Calculate Decay (Physics)
                nut_val = self.nutrient[nb_idx]
                new_e = _calculate_decay_jit(curr_energy, curr_decay, eff_weight, comp_penalty, nut_val)
                new_e = min(new_e, curr_max_energy)

                if new_e > self.cfg.DORMANCY_THRESHOLD:
                    self.pheromone[nb_idx] += 0.5
                    self._spawn_tip_internal(nb_idx, tip_idx, new_e, curr_colony, curr_dir)
                    branches_spawned += 1
                    any_child_spawned = True

            # Dormancy Logic (Stasis)
            # If we didn't move because of resistance/famine, but have energy, hibernate.
            if not any_child_spawned:
                # Idle energy decay (only time based, no friction)
                idle_e = curr_energy * curr_decay
                if idle_e > 0.1: 
                    self._spawn_tip_internal(curr_node, tip_idx, idle_e, curr_colony, curr_dir)

            self.tip_energies[tip_idx] = 0
            self.free_indices.append(tip_idx)

            # Global Decay
            if self.step % 500 == 0:
                self.pheromone *= self.cfg.PHEROMONE_DECAY
            self.step += 1

        if self.status == "running":
            self.status = "exhausted" if not self.pq else "timeout"

        return self.status

    def _register_meeting(self, meeting_node_idx, colony_id):
        for g in self.colony_goals_indices[colony_id]:
            if g not in self.found_goals_indices:
                self.found_goals_indices.add(g)

    def get_path(self, start_node, end_node) -> List:
        try:
            return nx.shortest_path(self.graph, start_node, end_node, weight='weight')
        except:
            return []

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'steps': int(self.step),
            'status': str(self.status),
            'visited_nodes': int(np.sum(self.visited_state != 0)),
            'pool_balance': {c: float(v) for c, v in self.colony_banks.items()},
            'active_tips': int(len(self.pq)),
            'colonies': {int(c): {'found_goals': int(len(d['found_goals']))} for c, d in self.colonies.items()}
        }

    @property
    def visited(self) -> Set[Tuple[int, int]]:
        indices = np.where(self.visited_state != 0)[0]
        return {self.nodes[i] for i in indices}

    @property
    def colonies(self) -> Dict[int, Dict]:
        res = {}
        for cid, cfg in enumerate(self.initial_config):
            c_goals_indices = self.colony_goals_indices[cid]
            found_indices = c_goals_indices.intersection(self.found_goals_indices)
            found_nodes = {self.nodes[i] for i in found_indices}
            res[cid] = {
                'spores': cfg.get('spores', []),
                'goals': cfg.get('goals', []),
                'found_goals': found_nodes
            }
        return res
