import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)

    def _evaluate(self, x, func):
        if self.evals >= self.budget: return float('inf')
        eval_x = np.clip(x.copy(), -1.0, 1.0)
        eval_x[18:24] = np.clip(np.round(eval_x[18:24]), 0, 5).astype(int)
        
        f = func(eval_x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = eval_x.copy()
        return f

    def _regularize_hessian(self, H):
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals = np.abs(eigvals) + 1e-6
        H_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return np.linalg.inv(H_reg)

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # Initial random sample to start
        x_current = np.random.uniform(-1, 1, self.dim)
        self._evaluate(x_current, func)
        
        T = 1.0
        alpha = 1.0
        min_T = 1e-6
        cat_pert_prob = 0.1
        
        while T > min_T and self.evals < self.budget:
            # Update temperature
            T *= 0.98
            
            # Obtain Hessian for current position to guide step
            if hess_func is not None and self.evals < self.budget:
                H_inv = self._regularize_hessian(hess_func(x_current))
            else:
                # Fallback to identity if no Hessian
                H_inv = np.eye(18)
            
            # Propose move in continuous subspace scaled by inverse Hessian
            noise_cont = np.random.randn(18)
            step_cont = alpha * T * (H_inv @ noise_cont)
            
            x_trial = x_current.copy()
            x_trial[:18] += step_cont
            
            # Perturb categorical variables stochastically
            if np.random.rand() < cat_pert_prob:
                idx = np.random.randint(0, 6)
                delta = np.random.choice([-1, 0, 1])
                x_trial[18:24] = np.clip(x_trial[18:24] + delta, 0, 5)
            
            # Clip and repair
            x_trial = np.clip(x_trial, -1.0, 1.0)
            x_trial[18:24] = np.clip(np.round(x_trial[18:24]), 0, 5).astype(int)
            
            f_trial = self._evaluate(x_trial, func)
            f_current = self.best_f if np.allclose(x_current, self.best_x) else self._evaluate(x_current, func) # Conservative current val
            
            # Standard SA acceptance
            df = f_trial - self._evaluate(x_current, func)
            if df < 0 or np.random.rand() < np.exp(-df / max(T, 1e-10)):
                x_current = x_trial.copy()
                # Adaptive step scaling based on acceptance
                alpha *= (1.0 + 0.1 * np.exp(-df))
            else:
                alpha *= 0.95
                
        return self.best_f, self.best_x