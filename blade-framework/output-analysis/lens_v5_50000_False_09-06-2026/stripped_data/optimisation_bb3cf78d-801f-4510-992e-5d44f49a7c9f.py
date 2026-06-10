import numpy as np
import cma
from scipy.linalg import inv

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        # Adaptive state
        self.smoothed_improve = 0.0
        self.prev_best = float('inf')
        self.sigma_mult = 1.0
        self.cat_mut_rate = 0.2
        self.cat_probs = np.ones(6) / 6.0
        self.hess_update_counter = 0
        self.sigma_base = 0.5
        
    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        x = np.clip(x.copy(), -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # Initialize CMA-ES on continuous subspace
        x0 = np.random.uniform(-1, 1, self.dim - 6)
        self.es = cma.CMAEvolutionStrategy(x0, self.sigma_base, {'popsize': 30, 'CMA_esigma': 0.5})
        
        while self.evals < self.budget:
            # 1. Sampling
            pop_c = self.es.ask()
            pop_size = pop_c.shape[0]
            
            # Generate categorical parts with adaptive probabilities
            pop_cat = np.zeros((pop_size, 6), dtype=int)
            for i in range(pop_size):
                probs = self.cat_probs.copy()
                if np.random.rand() < self.cat_mut_rate:
                    flip = np.random.randint(0, 6)
                    probs[flip] = 0.0
                    probs /= probs.sum() + 1e-12
                for j in range(6):
                    pop_cat[i, j] = np.random.choice(np.arange(6), p=probs)
                    
            # Combine, enforce bounds, and evaluate
            pop_full = np.hstack([pop_c, pop_cat])
            pop_full = np.clip(pop_full, -1.0, 1.0)
            pop_full[18:24] = np.clip(np.round(pop_full[18:24]), 0, 5).astype(int)
            
            pop_f = np.array([self._evaluate(x, func) for x in pop_full])
            
            # Update CMA-ES
            self.es.tell(pop_c, pop_f)
            
            # Reinforce categorical probabilities based on top performers
            top_idx = np.argsort(pop_f)[:max(1, pop_size // 5)]
            counts = np.bincount(pop_cat[top_idx].flatten(), minlength=6)
            self.cat_probs = 0.85 * self.cat_probs + 0.15 * (counts + 1) / (counts.sum() + 6)
            
            # 2. Runtime-Adaptive Controller
            # Compute smoothed improvement rate
            delta = max(0, self.prev_best - self.best_f)
            self.smoothed_improve = 0.9 * self.smoothed_improve + 0.1 * delta
            
            # Adaptive Step Size (Sigma)
            if self.smoothed_improve > 1e-5:
                self.sigma_mult = max(0.5, self.sigma_mult * 0.85)
            elif self.smoothed_improve < 1e-8:
                self.sigma_mult = min(2.0, self.sigma_mult * 1.15)
            else:
                self.sigma_mult = 1.0
                
            self.es.sigma *= self.sigma_mult
            self.sigma_mult = 1.0
            
            # Adaptive Categorical Mutation
            if self.smoothed_improve < 1e-8:
                self.cat_mut_rate = min(0.5, self.cat_mut_rate + 0.015)
            else:
                self.cat_mut_rate = max(0.1, self.cat_mut_rate - 0.015)
                
            # Hessian Preconditioning & Update Frequency
            self.hess_update_counter += 1
            hess_freq = 10 if self.smoothed_improve > 1e-6 else 5
            
            if hess_func is not None and self.hess_update_counter >= hess_freq:
                # Probe at current mean + best categorical configuration
                mean_c = self.es.mean
                full_x = np.zeros(self.dim)
                full_x[:18] = mean_c
                best_cat_idx = np.argmin(pop_f)
                full_x[18:24] = pop_cat[best_cat_idx]
                full_x = np.clip(full_x, -1.0, 1.0)
                
                if self.evals >= self.budget: break
                
                H_raw = hess_func(full_x)[:18, :18]
                eigs, Q = np.linalg.eigh(H_raw)
                
                # Regularize for positive definiteness
                eigs_reg = np.abs(eigs) + 1e-6
                P = Q @ np.diag(1.0 / eigs_reg) @ Q.T
                
                # Update CMA covariance with preconditioner
                # Scale to maintain stability
                trace_P = np.trace(P)
                if trace_P > 0 and np.trace(self.es.C) > 0:
                    P = P / trace_P * np.trace(self.es.C)
                self.es.C = P
                
                # Dynamic frequency based on conditioning
                cond = eigs_reg[-1] / eigs_reg[0]
                if cond > 500:
                    self.hess_update_counter = 0
                else:
                    self.hess_update_counter = hess_freq
                    
            if self.evals >= self.budget: break
            
        return self.best_f, self.best_x