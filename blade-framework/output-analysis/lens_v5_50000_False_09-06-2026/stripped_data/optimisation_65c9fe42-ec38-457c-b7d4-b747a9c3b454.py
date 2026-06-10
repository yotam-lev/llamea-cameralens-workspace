import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self._hess_func = None
        
        # Core parameters
        self.n_pop = 20
        self.CR = 0.85
        self.F_base = 0.6
        
        # History for stagnation detection
        self.improv_window = 10
        self.history = []
        
        # Population state
        self.pop = None
        self.fitness = None

    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        x = np.clip(x.copy(), -1.0, 1.0)
        # Clip categorical dimensions strictly
        cat_x = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _get_hessian_info(self, x):
        """Compute Hessian and eigendecomposition for escape/curvature info."""
        H = self._hess_func(x)
        eigvals, eigvecs = np.linalg.eigh(H)
        # Regularize for stability: ensure positive definite for metric
        # but keep original eigvals for saddle detection
        shift = max(0, -eigvals.min() + 1e-6)
        H_reg = H + shift * np.eye(18)
        return H_reg, eigvals, eigvecs

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self._hess_func = hess_func
        self.pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        self.fitness = np.full(self.n_pop, np.inf)

        # Initial evaluation
        for i in range(len(self.pop)):
            if self.evals >= self.budget: break
            self.fitness[i] = self._evaluate(self.pop[i], func)

        self.history = [self.best_f]
        H_reg = None
        eigvals = None
        eigvecs = None
        it = 0

        while self.evals < self.budget:
            it += 1
            
            # Update Hessian on best individual periodically
            if (it % 5 == 0 or H_reg is None) and self._hess_func:
                idx = np.argmin(self.fitness)
                if self.evals < self.budget:
                    H_reg, eigvals, eigvecs = self._get_hessian_info(self.pop[idx])
                best_idx = idx
            else:
                best_idx = np.argmin(self.fitness)
            
            best_x = self.pop[best_idx]
            best_f = self.fitness[best_idx]

            # Detect saddle for escape
            has_saddle = False
            saddle_dir = None
            if eigvals is not None:
                min_idx = np.argmin(eigvals)
                if eigvals[min_idx] < 0:
                    has_saddle = True
                    saddle_dir = eigvecs[:, min_idx]
                    # Adaptive step size for escape
                    escape_step = np.abs(eigvals[min_idx]) * 0.5
                    escape_step = max(0.1, min(escape_step, 1.0))

            # Differential Evolution with Hessian-Aware Mutation
            for i in range(len(self.fitness)):
                if self.evals >= self.budget: break
                
                if has_saddle:
                    # Saddle Escape Strategy: Jump along repulsive direction
                    # Mix with random to maintain diversity
                    alpha = np.random.uniform(0.2, 0.8)
                    diff_dir = saddle_dir
                    # Rotate escape direction slightly for exploration
                    noise = np.random.randn(18)
                    diff_dir = diff_dir * (1 - alpha) + noise * alpha
                    diff_dir = diff_dir / np.linalg.norm(diff_dir)
                    
                    target = best_x[:18] + escape_step * diff_dir
                    mutant = np.full(self.dim, np.inf)
                    mutant[:18] = target
                    # Categorical perturbation
                    mutant[18:24] = best_x[18:24].copy()
                    if np.random.random() < 0.1:
                        mutant[18:24] = np.clip(np.round(np.random.uniform(0, 6, 6)), 0, 5).astype(int)
                else:
                    # Curvature-Preconditioned DE Strategy
                    a, b, c = np.random.choice(len(self.fitness), 3, replace=False)
                    diff = self.pop[a] - self.pop[b]
                    
                    # Precondition by inverse sqrt of eigenvalues
                    # This scales mutations to be larger in flat directions
                    # and smaller in stiff directions
                    inv_sqrt_eigs = 1.0 / np.sqrt(np.clip(eigvals, 1e-6, None))
                    diff[:18] *= inv_sqrt_eigs
                    
                    # Adaptive F based on curvature
                    # Higher condition number -> larger F to explore valleys
                    cond = np.max(eigvals) / np.min(eigvals)
                    F = self.F_base * np.log1p(cond) / np.log1p(50)
                    F = np.clip(F, 0.2, 1.0)
                    
                    mutant = best_x + F * diff
                    # Categorical handling for DE
                    mutant[18:24] = np.clip(np.round(mutant[18:24]), 0, 5).astype(int)

                # Boundary and Categorical enforcement
                mutant[:18] = np.clip(mutant[:18], -1.0, 1.0)
                mutant[18:24] = np.clip(np.round(mutant[18:24]), 0, 5).astype(int)
                
                # Trial vector selection
                j_rand = np.random.randint(0, self.dim)
                for j in range(self.dim):
                    if np.random.random() < self.CR or j == j_rand:
                        pass # Keep mutant
                    else:
                        mutant[j] = self.pop[i][j]
                
                # Re-apply constraints after crossover
                mutant[:18] = np.clip(mutant[:18], -1.0, 1.0)
                mutant[18:24] = np.clip(np.round(mutant[18:24]), 0, 5).astype(int)

                f_m = self._evaluate(mutant, func)
                
                if f_m < self.fitness[i]:
                    self.pop[i] = mutant
                    self.fitness[i] = f_m

            # Local Search on Best for Convergence
            if self.evals < self.budget and self._hess_func and it % 8 == 0:
                best_idx = np.argmin(self.fitness)
                c = self.pop[best_idx][:18].copy()
                cat = self.pop[best_idx][18:24].copy()
                
                if H_reg is not None:
                    try:
                        res = minimize(
                            lambda xc: func(np.concatenate([xc, cat])),
                            c, method='trust-constr', 
                            hess=lambda xc: H_reg,
                            bounds=[(-1.0, 1.0)] * 18,
                            options={'maxiter': 20, 'verbose': 0}
                        )
                        if res.success and self.evals < self.budget:
                            cand = np.concatenate([res.x, cat])
                            f_c = self._evaluate(cand, func)
                            if f_c < self.fitness[best_idx]:
                                self.pop[best_idx] = cand
                                self.fitness[best_idx] = f_c
                    except Exception:
                        pass

            # Stagnation check
            self.history.append(self.best_f)
            if len(self.history) > self.improv_window:
                self.history.pop(0)
            
        return self.best_f, self.best_x