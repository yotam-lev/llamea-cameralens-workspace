import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.hess_freq = 10
        self.reg = 1e-4
        self.H = None
        self.H_valid = False
        self.pop_size = 24
        
    def _clip_and_map(self, x):
        xc = np.clip(x, -1.0, 1.0)
        xc[18:24] = np.clip(np.round(xc[18:24]), 0, 5).astype(int)
        return xc

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
            H_raw = self.hess_func(x)
            eigs, vecs = np.linalg.eigh(H_raw)
            eigs_reg = np.abs(eigs)
            shift = max(0, self.reg - eigs_reg.min())
            eigs_reg += shift
            self.H = vecs @ np.diag(eigs_reg) @ vecs.T
            self.H_valid = True
        except:
            pass

    def _conditional_geometric_opt(self, x_full, func, hess_func):
        """Performs Hessian-accelerated refinement of continuous variables for fixed discrete state."""
        xc = x_full[:18].copy()
        xd = x_full[18:24].copy()
        
        try:
            res = minimize(
                lambda xc_c: func(np.concatenate([xc_c, xd])),
                xc, method='trust-constr',
                hess=lambda xc_c: self.H if self.H_valid else None,
                bounds=[(-1.0, 1.0)] * 18,
                options={'maxiter': 30, 'verbose': 0}
            )
            if res.success and self.evals < self.budget:
                xc_opt = np.clip(res.x, -1.0, 1.0)
                x_new = np.concatenate([xc_opt, xd])
                f_new = self._evaluate(x_new, func)
                return x_new, f_new
        except:
            pass
        return x_full, float('inf')

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        pop = np.random.uniform(-1, 1, size=(self.pop_size, self.dim))
        last_cat = np.zeros((self.pop_size, 6), dtype=int)
        gbest_idx = 0
        gbest_f = float('inf')
        
        # Initial evaluation
        for i in range(self.pop_size):
            f = self._evaluate(pop[i], func)
            if f < gbest_f:
                gbest_f = f
                gbest_idx = i
            last_cat[i] = np.clip(np.round(pop[i][18:24]), 0, 5).astype(int)
                
        it = 0
        while self.evals < self.budget:
            it += 1
            
            if self.hess_func and (it % self.hess_freq == 0 or not self.H_valid):
                if self.evals < self.budget:
                    self._update_hessian(pop[gbest_idx])
                    
            for i in range(self.pop_size):
                if self.evals >= self.budget:
                    break
                    
                # Standard update step
                r1, r2 = np.random.rand(), np.random.rand()
                w = 0.5 + 0.5 * np.random.rand()
                vel = w * (pop[gbest_idx] - pop[i]) + r1 * np.random.randn(self.dim)
                pop[i] += vel
                
                # Discrete-Guided Conditional Trigger
                curr_cat = np.clip(np.round(pop[i][18:24]), 0, 5).astype(int)
                if not np.array_equal(curr_cat, last_cat[i]):
                    # Categorical change detected; trigger geometric refinement
                    pop[i], f_refined = self._conditional_geometric_opt(pop[i], func, self.hess_func)
                    if f_refined < self.best_f:
                        gbest_f = f_refined
                        gbest_idx = i
                    last_cat[i] = curr_cat.copy()
                else:
                    # No change, evaluate normally
                    f = self._evaluate(pop[i], func)
                    if f < gbest_f:
                        gbest_f = f
                        gbest_idx = i
                        
        return self.best_f, self.best_x