import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

class LensPopulationEnv(gym.Env):
    """
    A Custom Gymnasium Environment that formulates population-based 
    optimization as a Markov Decision Process for Deep RL.
    """
    def __init__(self, problem_builder, pop_size=100, dim=24):
        super().__init__()
        self.problem_builder = problem_builder
        self.pop_size = pop_size
        self.dim = dim
        
        # We need to instantiate the problem inside the env to avoid multiprocessing pickle issues
        self.problem = self.problem_builder()
        
        # Attempt to extract objective components
        try:
            self.func, _, _, self.lb, self.ub, _, self.obj = self.problem._build_objective()
        except Exception as e:
            raise RuntimeError(f"Failed to extract objective: {e}")

        # Action Space: 5 Discrete heuristics (A1-A5)
        self.action_space = spaces.Discrete(5)
        
        # State Space: [x_i(24), f_i(1), best_neighbor(24), f_nb(1), centroid(24)] = 74
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, 
            shape=(self.dim * 3 + 2,), 
            dtype=np.float32
        )
        
        self.budget = self.problem.budget_factor
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = None

    def _evaluate_vector(self, x):
        x_norm = np.clip(x, -1, 1)
        xr = self.lb + (x_norm + 1.0) / 2.0 * (self.ub - self.lb)
        xc, xi = self.obj.split_theta(xr)
        xr_rounded = self.obj.pack_theta(xc, xi)
        return self.func(xr_rounded)

    def _update_population_stats(self):
        self.dist_matrix = cdist(self.pop, self.pop)
        clustering = DBSCAN(eps=0.3, min_samples=3).fit(self.pop)
        self.clusters = clustering.labels_

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize Population
        self.pop = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)
        
        for i in range(self.pop_size):
            self.fitness[i] = self._evaluate_vector(self.pop[i])
            self.evals += 1
            if self.fitness[i] < self.best_f:
                self.best_f = self.fitness[i]
                self.best_x = self.pop[i].copy()
                
        self.current_idx = 0
        self._update_population_stats()
        
        return self._get_obs(), {}

    def _get_obs(self):
        x_i = self.pop[self.current_idx]
        f_i = self.fitness[self.current_idx]
        
        # Best neighbor logic
        neighbors = np.argsort(self.dist_matrix[self.current_idx])[1:6]
        best_nb_idx = neighbors[np.argmin(self.fitness[neighbors])]
        x_nb = self.pop[best_nb_idx]
        f_nb = self.fitness[best_nb_idx]
        
        # Cluster logic
        my_cluster = self.clusters[self.current_idx]
        if my_cluster != -1:
            cluster_indices = np.where(self.clusters == my_cluster)[0]
            centroid = np.mean(self.pop[cluster_indices], axis=0)
        else:
            centroid = x_i
            
        # Normalize fitness for NN stability
        norm_f_i = np.clip(f_i / (abs(self.best_f) + 1e-9), -10, 10)
        norm_f_nb = np.clip(f_nb / (abs(self.best_f) + 1e-9), -10, 10)
        
        obs = np.concatenate([x_i, [norm_f_i], x_nb, [norm_f_nb], centroid])
        return obs.astype(np.float32)

    def step(self, action):
        x_i = self.pop[self.current_idx].copy()
        x_new = x_i.copy()
        
        # Execute Action (A1-A5 mapping)
        if action == 0:   # A1: Gaussian
            x_new = x_i + np.random.normal(0, 0.05, self.dim)
        elif action == 1: # A2: KNN Exploit
            neighbors = np.argsort(self.dist_matrix[self.current_idx])[1:6]
            best_nb = self.pop[neighbors[np.argmin(self.fitness[neighbors])]]
            x_new = x_i + 0.5 * np.random.random() * (best_nb - x_i)
        elif action == 2: # A3: KNN Explore
            neighbors = np.argsort(self.dist_matrix[self.current_idx])[1:6]
            rand_nb = self.pop[np.random.choice(neighbors)]
            x_new = x_i + np.random.uniform(-1, 1) * (rand_nb - x_i)
        elif action == 3: # A4: Global Jump
            unique_clusters = np.unique(self.clusters)
            if len(unique_clusters) > 1:
                target_cluster = np.random.choice(unique_clusters[unique_clusters != self.clusters[self.current_idx]])
                target_indices = np.where(self.clusters == target_cluster)[0]
                x_target = self.pop[np.random.choice(target_indices)]
                x_new = x_target + np.random.normal(0, 0.01, self.dim)
            else:
                x_new = np.random.uniform(-1, 1, self.dim)
        elif action == 4: # A5: DE/rand/1
            idxs = [idx for idx in range(self.pop_size) if idx != self.current_idx]
            r1, r2, r3 = self.pop[np.random.choice(idxs, 3, replace=False)]
            x_new = r1 + 0.8 * (r2 - r3)

        x_new = np.clip(x_new, -1, 1)
        
        # Evaluate
        f_new = self._evaluate_vector(x_new)
        self.evals += 1
        
        # Reward Calculation (Scale to help NN gradients)
        reward = 0.0
        if f_new < self.fitness[self.current_idx]:
            reward = (self.fitness[self.current_idx] - f_new) * 100.0
            self.pop[self.current_idx] = x_new
            self.fitness[self.current_idx] = f_new
            
            if f_new < self.best_f:
                self.best_f = f_new
                self.best_x = x_new.copy()
        else:
            reward = -0.01 # Slight penalty for wasted evaluation
            
        # Move to next explorer
        self.current_idx += 1
        if self.current_idx >= self.pop_size:
            self.current_idx = 0
            self._update_population_stats()
            
        terminated = self.evals >= self.budget
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {"best_f": self.best_f}