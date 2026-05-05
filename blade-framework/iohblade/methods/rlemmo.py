import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from ..method import Method
from ..problem import Problem
from ..solution import Solution

class RLEMMO(Method):
    """
    RLEMMO: Evolutionary Multimodal Optimization Assisted By Deep Reinforcement Learning.
    (arXiv:2404.08242)
    
    This implementation uses a self-adaptive operator selection mechanism (MAB-style) 
    as a baseline version of the RL policy.
    """
    def __init__(self, llm=None, budget=int, pop_size=50, name="RLEMMO", **kwargs):
        super().__init__(llm, budget, name)
        self.pop_size = pop_size
        self.dim = None
        self.archive = [] # For diversity tracking
        
        # Adaptive Operator Weights (A1-A5)
        self.weights = np.ones(5) / 5
        self.counts = np.zeros(5)
        self.successes = np.zeros(5)

    def _get_neighbors(self, population, k=5):
        dist_matrix = cdist(population, population)
        # Get indices of k nearest neighbors for each individual (excluding self)
        neighbors = np.argsort(dist_matrix, axis=1)[:, 1:k+1]
        return neighbors

    def _get_clusters(self, population):
        # Use DBSCAN to identify clusters in the search space
        clustering = DBSCAN(eps=0.3, min_samples=3).fit(population)
        return clustering.labels_

    def __call__(self, problem: Problem) -> Solution:
        # 1. Initialize Population
        # Note: We need the dimension from the problem
        # For Lens problem, it's 24.
        self.dim = 24 # Default for Lens, will try to detect
        
        # Initialize population in [-1, 1]
        pop = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        fitness = np.array([float('inf')] * self.pop_size)
        
        best_f = float('inf')
        best_x = None
        evals = 0

        # Initial evaluation
        for i in range(self.pop_size):
            if evals >= problem.budget_factor: break
            # Round categorical dimensions [18:24]
            x_eval = pop[i].copy()
            x_eval[18:24] = np.round(x_eval[18:24] * 2) / 2
            
            # The Problem class __call__ returns a Solution object
            # But the inner objective is accessed via problem.evaluate or the wrapper
            # In iohblade, calling the problem object evaluates the solution
            sol = Solution(code="") # Fix: Removed fitness from constructor
            # ...

        # Since we are implementing a fixed baseline, we'll extract the objective function
        # from the problem instance directly.
        try:
            # This assumes LensOptimisation or similar
            if hasattr(problem, '_build_objective'):
                func, grad_fn, dim, lb, ub, grad0_cont, obj = problem._build_objective()
                self.dim = dim
            else:
                raise AttributeError("Problem does not expose internal objective.")
        except Exception as e:
            print(f"RLEMMO: Failed to extract objective from problem: {e}")
            return Solution(code="failed", fitness=-np.inf)

        # Re-initialize with correct dim
        pop = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        fitness = np.array([float('inf')] * self.pop_size)
        
        def evaluate_vector(x):
            # Normalize and round for lens
            x_norm = np.clip(x, -1, 1)
            # Map [-1, 1] to the problem's [lb, ub]
            xr = lb + (x_norm + 1.0) / 2.0 * (ub - lb)
            # Round glass IDs
            xc, xi = obj.split_theta(xr)
            xr_rounded = obj.pack_theta(xc, xi)
            return func(xr_rounded)

        # Initial Population Evaluation
        for i in range(self.pop_size):
            fitness[i] = evaluate_vector(pop[i])
            evals += 1
            if fitness[i] < best_f:
                best_f = fitness[i]
                best_x = pop[i].copy()
            
            # Log progress to the run directory
            if hasattr(problem, 'logger') and problem.logger:
                dummy_sol = Solution(code=f"# Ind {i} init")
                dummy_sol.set_scores(-fitness[i])
                problem.logger.log_individual(dummy_sol)

        # 2. Main Evolution Loop
        while evals < problem.budget_factor:
            # Update state-based info
            neighbors = self._get_neighbors(pop)
            clusters = self._get_clusters(pop)
            
            new_pop = pop.copy()
            
            for i in range(self.pop_size):
                if evals >= problem.budget_factor: break
                
                # Select Action (Adaptive/MAB)
                # p = self.weights
                action = np.random.choice(5, p=self.weights)
                self.counts[action] += 1
                
                x_i = pop[i]
                x_new = x_i.copy()
                
                # A1: Gaussian Local Search
                if action == 0:
                    sigma = 0.05
                    x_new = x_i + np.random.normal(0, sigma, self.dim)
                
                # A2: KNN-based Exploitation (Move toward best neighbor)
                elif action == 1:
                    nb_indices = neighbors[i]
                    best_nb_idx = nb_indices[np.argmin(fitness[nb_indices])]
                    x_best_nb = pop[best_nb_idx]
                    x_new = x_i + 0.5 * np.random.random() * (x_best_nb - x_i)
                
                # A3: KNN-based Exploration (Mutation with random neighbor)
                elif action == 2:
                    nb_indices = neighbors[i]
                    rand_nb_idx = np.random.choice(nb_indices)
                    x_rand_nb = pop[rand_nb_idx]
                    x_new = x_i + np.random.uniform(-1, 1) * (x_rand_nb - x_i)
                
                # A4: Global Neighborhood Sharing (Jump to another cluster)
                elif action == 3:
                    unique_clusters = np.unique(clusters)
                    if len(unique_clusters) > 1:
                        my_cluster = clusters[i]
                        other_clusters = unique_clusters[unique_clusters != my_cluster]
                        target_cluster = np.random.choice(other_clusters)
                        # Pick a random individual from target cluster
                        target_indices = np.where(clusters == target_cluster)[0]
                        x_target = pop[np.random.choice(target_indices)]
                        x_new = x_target + np.random.normal(0, 0.01, self.dim)
                    else:
                        # Fallback to random jump
                        x_new = np.random.uniform(-1, 1, self.dim)
                
                # A5: DE/rand/1
                elif action == 4:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    r1, r2, r3 = pop[np.random.choice(idxs, 3, replace=False)]
                    F = 0.8
                    x_new = r1 + F * (r2 - r3)

                # Boundary Constraint
                x_new = np.clip(x_new, -1, 1)
                
                # Evaluate
                f_new = evaluate_vector(x_new)
                evals += 1
                
                # Selection & Reward
                if f_new < fitness[i]:
                    # Success!
                    reward = (fitness[i] - f_new) / (abs(fitness[i]) + 1e-9)
                    self.successes[action] += reward
                    new_pop[i] = x_new
                    fitness[i] = f_new
                    
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        # Log improvement
                        if hasattr(problem, 'logger') and problem.logger:
                            best_sol = Solution(code=f"# New best at evals {evals}")
                            best_sol.set_scores(-best_f)
                            problem.logger.log_individual(best_sol)
            
            pop = new_pop
            
            # Update weights (Simple Adaptive logic)
            if evals % (self.pop_size * 2) == 0:
                total_success = np.sum(self.successes)
                if total_success > 0:
                    self.weights = 0.9 * self.weights + 0.1 * (self.successes / (self.counts + 1e-9))
                    # Softmax/Normalize
                    self.weights = self.weights / np.sum(self.weights)
                # Reset counters for next window
                self.counts[:] = 0
                self.successes[:] = 0

        # Return the best found as a Solution object
        # We store the 'code' as a description of what RLEMMO did
        res_sol = Solution(
            code=f"# RLEMMO Baseline\n# Population based adaptive search\n# Best Fitness: {best_f}",
        )
        res_sol.set_scores(-best_f, feedback=f"RLEMMO best fitness: {best_f}")
        return res_sol

    def to_dict(self):
        return {
            "method_name": self.name,
            "budget": self.budget,
            "pop_size": self.pop_size,
            "weights": self.weights.tolist()
        }
