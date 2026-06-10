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
        n_pop = 25
        pop = np.random.uniform(-1, 1, size=(n_pop, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])
        
        inertia = np.zeros((n_pop, self.dim))
        mu = 0.65
        stagnation = 0
        last_best_f = self.best_f
        
        while self.evals < self.budget:
            best_idx = np.argmin(pop_f)
            
            H_raw = None
            eigs = None
            Q = None
            H_reg = None
            if hess_func is not None and self.evals < self.budget:
                x_best = pop[best_idx].copy()
                x_best[18:24] = np.clip(np.round(x_best[18:24]), 0, 5).astype(int)
                if self.evals >= self.budget: break
                H_raw = hess_func(x_best)
                eigs, Q = np.linalg.eigh(H_raw)
                H_reg = Q @ np.diag(np.abs(eigs) + 1e-5) @ Q.T
            
            if self.best_f == last_best_f:
                stagnation += 1
            else:
                stagnation = 0
                last_best_f = self.best_f
                
            if self.evals < self.budget:
                if H_raw is not None:
                    # Resonance scaling: amplify steep walls, expand shallow valleys
                    scales = np.where(np.abs(eigs) > 1.0, np.sqrt(np.abs(eigs)), 
                                      1.0 / np.sqrt(np.abs(eigs) + 1e-6))
                    # Boost steep directions to force wall crossing
                    steep_mask = np.abs(eigs) > np.median(np.abs(eigs)) * 2.0
                    scales[steep_mask] *= 1.8
                else:
                    scales = np.ones(18)
                    
                noise = np.random.randn(n_pop, self.dim)
                noise[:, :18] *= scales
                
                inertia = mu * inertia + noise
                new_pop = pop + inertia
                pop = np.clip(new_pop, -1.0, 1.0)
                
                cat_base = np.clip(np.round(pop[best_idx][18:24]), 0, 5).astype(int)
                for i in range(n_pop):
                    flip_prob = 0.25 + 0.5 * min(stagnation / 8.0, 1.0)
                    if np.random.rand() < flip_prob:
                        mut = np.random.choice([-1, 0, 1], size=6)
                        pop[i, 18:24] = np.clip(cat_base + mut, 0, 5).astype(int)
                        
                for i in range(n_pop):
                    if self.evals >= self.budget: break
                    f = self._evaluate(pop[i], func)
                    if f < pop_f[i]:
                        pop_f[i] = f
                        
            if H_reg is not None and self.evals < self.budget and stagnation == 0:
                best_idx = np.argmin(pop_f)
                xc = pop[best_idx][:18]
                cat = np.clip(np.round(pop[best_idx][18:24]), 0, 5).astype(int)
                
                def obj(xs): return func(np.concatenate([xs, cat]))
                def jac(xs): return grad_func(np.concatenate([xs, cat]))[:18] if grad_func else np.zeros(18)
                def hess(xs): return H_reg
                    
                res = minimize(obj, xc, jac=jac, hess=hess, method='trust-constr',
                               bounds=[(-1.0, 1.0)]*18, options={'maxiter': 15, 'verbose': 0})
                
                x_ref = np.concatenate([res.x, cat])
                if self.evals < self.budget:
                    f_ref = self._evaluate(x_ref, func)
                    if f_ref < pop_f[best_idx]:
                        pop[best_idx] = x_ref
                        pop_f[best_idx] = f_ref
                        
        return self.best_f, self.best_x