import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        self.H_reg = None
        self.Q = None
        self.eigs_vals = None
        self.cond_num = 1.0
        self.flat_dir = None
        
        self.prev_best_f = float('inf')
        self.last_improve_eval = 0
        self.improve_rate = 1.0
        
        self.discrete_pop = np.zeros((50, 6), dtype=int)
        self.continuous_pop = np.random.uniform(-1, 1, size=(50, 18))
        self.discrete_fitness = np.full(50, float('inf'))
        self.continuous_fitness = np.full(50, float('inf'))

    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        x_safe = x.copy()
        x_safe[0:18] = np.clip(x_safe[0:18], -1.0, 1.0)
        x_safe[18:24] = np.clip(np.round(x_safe[18:24]), 0, 5).astype(int)
        f = func(x_safe)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x_safe.copy()
        return f

    def _regularize_hessian(self, H_raw):
        eigs, Q = np.linalg.eigh(H_raw)
        min_abs = np.min(np.abs(eigs))
        self.cond_num = np.max(np.abs(eigs)) / (min_abs + 1e-8)
        self.Q = Q
        self.eigs_vals = eigs
        self.H_reg = Q @ np.diag(np.abs(eigs) + 1e-6) @ Q.T
        self.flat_dir = Q[:, np.argmin(np.abs(eigs))]

    def _conditional_refine(self, cat_block, func, hess_func):
        if self.evals >= self.budget:
            return None
        best_c = np.random.uniform(-1, 1, 18)
        best_f = float('inf')
        
        for _ in range(5):
            x0 = best_c.copy()
            def obj(xc): return func(np.concatenate([xc, cat_block]))
            def jac(xc): return grad_func(np.concatenate([xc, cat_block]))[:18] if grad_func else np.zeros(18)
            def hess(xc): return self.H_reg if self.H_reg is not None else np.eye(18)
            
            res = minimize(obj, x0, jac=jac, hess=hess, method='trust-constr',
                           bounds=[(-1.0, 1.0)]*18, options={'maxiter': 30, 'verbose': 0})
            
            if self.evals >= self.budget: break
            c_sol = np.clip(res.x, -1.0, 1.0)
            x_full = np.concatenate([c_sol, cat_block])
            f_val = self._evaluate(x_full, func)
            if self.evals < self.budget and f_val < best_f:
                best_f = f_val
                best_c = c_sol.copy()
                
        return best_c, best_f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.discrete_pop = np.random.randint(0, 6, size=(50, 6))
        
        for i in range(50):
            c = np.random.uniform(-1, 1, 18)
            x = np.concatenate([c, self.discrete_pop[i]])
            self.continuous_fitness[i] = self._evaluate(x, func)
            if self.evals >= self.budget: break

        exploit_freq = 4
        hess_freq = 10
        temp = 1.0

        while self.evals < self.budget:
            update = self.evals % 5 == 0
            if update:
                delta = max(1e-8, self.prev_best_f - self.best_f)
                self.improve_rate = 0.85 * self.improve_rate + 0.15 * delta
                self.prev_best_f = self.best_f
                self.last_improve_eval = self.evals

            if update and self.evals % hess_freq == 0 and hess_func is not None:
                best_idx = np.argmin(self.continuous_fitness)
                x_probe = np.concatenate([self.continuous_pop[best_idx], self.discrete_pop[best_idx]])
                if self.evals < self.budget:
                    H_raw = hess_func(x_probe)[:18, :18]
                    self._regularize_hessian(H_raw)

            if update and self.evals % exploit_freq == 0 and hess_func is not None:
                indices = np.argsort(self.continuous_fitness)[:15]
                for idx in indices:
                    cat = self.discrete_pop[idx]
                    res = self._conditional_refine(cat, func, hess_func)
                    if res is not None:
                        new_c, new_f = res
                        self.continuous_pop[idx] = new_c
                        self.continuous_fitness[idx] = new_f
                        self.discrete_fitness[idx] = new_f

            if self.evals >= self.budget: break

            remaining = 1.0 - self.evals / self.budget
            temp *= np.exp(-0.02 * remaining)
            
            cat_mut_prob = np.clip(0.4 * (1.0 + self.cond_num / 2000 - self.improve_rate * 2), 0.1, 0.9)
            
            for i in range(50):
                if np.random.rand() < cat_mut_prob:
                    base_cat = self.discrete_pop[i].copy()
                    mut = np.random.choice([-1, 0, 1], size=6)
                    new_cat = np.clip(base_cat + mut, 0, 5)
                    
                    if self.continuous_fitness[i] < self.discrete_fitness[i]:
                        self.discrete_pop[i] = new_cat
                        self.discrete_fitness[i] = self.continuous_fitness[i]
                        self.continuous_fitness[i] = float('inf')
                        
                        new_c, new_f = self._conditional_refine(new_cat, func, hess_func)
                        if new_c is not None and self.evals < self.budget:
                            self.continuous_pop[i] = new_c
                            self.continuous_fitness[i] = new_f
                            self.discrete_fitness[i] = new_f

        return self.best_f, self.best_x