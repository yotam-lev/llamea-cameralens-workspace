import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        self.pop_size = 32
        self.history_len = 10
        self.f_history = []
        self.cont_history = []
        
        # Hessian state
        self.H_inv = None
        self.H_reg = None
        self.H_valid = False
        self.hess_func = None
        self.last_H_x = None
        self.damping = 1e-7
        
        # Control
        self.cat_perturb_prob = 0.15
        self.cont_refine_rate = 0.8
        self.trust_limit = 0.3

    def _clip_and_map(self, x):
        x = np.asarray(x, dtype=float)
        x[:18] = np.clip(x[:18], -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        return x

    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        xc = self._clip_and_map(x)
        f = func(xc)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = xc.copy()
        return f

    def _update_hessian(self, x):
        try:
            H = self.hess_func(x)
            vals, vecs = np.linalg.eigh(H)
            min_val = np.min(vals)
            damp = max(self.damping, abs(min_val) * 0.1) if min_val < 0 else self.damping
            vals_reg = np.abs(vals) + damp
            self.H_reg = vecs @ np.diag(vals_reg) @ vecs.T
            self.H_inv = np.linalg.inv(self.H_reg)
            self.max_eig = np.max(vals_reg)
            self.min_eig = np.min(vals_reg)
            self.cond = self.max_eig / max(self.min_eig, 1e-9)
            self.H_valid = True
            self.last_H_x = x.copy()
        except Exception:
            self.H_valid = False

    def _adapt(self):
        self.f_history.append(self.best_f)
        if len(self.f_history) > self.history_len:
            self.f_history.pop(0)
        
        imp = 0.0
        if len(self.f_history) >= self.history_len:
            imp = (self.f_history[-1] - self.f_history[-self.history_len]) / max(abs(self.f_history[-self.history_len]), 1e-9)
            
        progress = self.evals / self.budget
        
        # Bidirectional coupling logic
        # If continuous improvement stalls, increase categorical mutation
        if imp > -1e-6:
            self.cat_perturb_prob = np.clip(0.1 + 0.2 * (1 - progress), 0.05, 0.3)
            self.cont_refine_rate = 0.6 + 0.4 * progress
        else:
            # Continuous improving: lock categories, refine geometry aggressively
            self.cat_perturb_prob = np.clip(0.05 + 0.05 * progress, 0.05, 0.1)
            self.cont_refine_rate = 0.9
            
        # Hessian condition adaptation
        if self.H_valid and self.cond > 1e3:
            self.cat_perturb_prob *= 1.2  # Stiffness suggests geometry traps, need category change
            self.damping = min(self.damping * 2, 1e-4)
        else:
            self.damping = max(self.damping * 0.9, 1e-8)

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        
        # Init population
        pop_cont = np.random.uniform(-1, 1, size=(self.pop_size, 18))
        pop_cat = np.zeros((self.pop_size, 6), dtype=int)
        
        # Initial eval
        for i in range(self.pop_size):
            if self.evals >= self.budget: break
            x = self._combine(pop_cont[i], pop_cat[i])
            self._evaluate(x, func)
            
        if self.evals >= self.budget:
            return self.best_f, self.best_x

        it = 0
        while self.evals < self.budget and it < self.budget // self.pop_size:
            it += 1
            self._adapt()
            
            # Hessian update on best
            if self.hess_func and (not self.H_valid or it % 24 == 0):
                if self.evals < self.budget - 15:
                    self._update_hessian(self.best_x)

            # Mixed-variable loop
            for i in range(self.pop_size):
                if self.evals >= self.budget: break
                
                # 1. Categorical perturbation (Bidirectional trigger)
                if np.random.rand() < self.cat_perturb_prob:
                    if np.random.rand() < 0.5:
                        # Exploit successful material
                        pop_cat[i] = self.best_x[18:24].copy()
                        pop_cat[i][np.random.randint(0, 6, 2)] = np.random.randint(0, 6, 2)
                    else:
                        # Explore
                        pop_cat[i] = np.random.randint(0, 6, 6)
                
                # 2. Continuous optimization (Curvature guided)
                # Newton step with Hessian
                if self.H_valid and np.random.rand() < self.cont_refine_rate:
                    try:
                        g = grad_func(self.best_x)
                        step = -self.H_inv @ g
                        # Adaptive damping based on condition
                        norm_step = np.linalg.norm(step)
                        if norm_step > 0 and self.cond > 500:
                            scale = min(1.0, 0.5 / norm_step)
                            step *= scale
                        pop_cont[i] += step
                    except Exception:
                        pass
                
                # 3. Evaluation
                trial = self._combine(pop_cont[i], pop_cat[i])
                f_trial = self._evaluate(trial, func)
                
                # Selection
                if f_trial <= self.best_f:
                    # Local acceptance
                    self.best_f = f_trial
                    self.best_x = trial.copy()
                    
            # Global selection pressure via population refresh hint
            if it % 10 == 0 and self.evals < self.budget - 20:
                # Re-seed worst performers based on current best geometry manifold
                if self.H_valid:
                    mean_cont = np.mean(pop_cont, axis=0)
                    for j in range(self.pop_size):
                        if np.random.rand() < 0.3:
                            pop_cont[j] = mean_cont + np.random.randn(18) * 0.1
                            pop_cont[j] = np.clip(pop_cont[j], -1, 1)
                            
        return self.best_f, self.best_x