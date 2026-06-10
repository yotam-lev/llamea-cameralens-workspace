import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        # DE Parameters
        self.pop_size = 45
        self.F = 0.85
        self.CR = 0.85
        self.memetic_freq = 12
        self.top_k = 4
        
        self.pop = np.zeros((self.pop_size, self.dim))
        self.pop_f = np.full(self.pop_size, float('inf'))
        self.H_precond = None
        self.gbest_idx = -1
        self.gbest_f = float('inf')
        
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

    def _regularize(self, H):
        eigs, Q = np.linalg.eigh(H)
        return Q @ np.diag(np.abs(eigs) + 1e-6) @ Q.T

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # Initialize population
        self.pop = np.random.uniform(-1, 1, size=(self.pop_size, self.dim))
        
        # Initial evaluation
        for i in range(self.pop_size):
            if self.evals >= self.budget: break
            self.pop_f[i] = self._evaluate(self.pop[i], func)
            if self.pop_f[i] < self.gbest_f:
                self.gbest_f = self.pop_f[i]
                self.gbest_idx = i

        gen = 0
        while self.evals < self.budget:
            gen += 1
            
            # Hessian Preconditioner Update
            if hess_func is not None and self.evals < self.budget:
                if self.H_precond is None or gen % 5 == 0:
                    x_c = self.pop[self.gbest_idx, :18]
                    cat = np.clip(np.round(self.pop[self.gbest_idx, 18:24]), 0, 5).astype(int)
                    H_raw = hess_func(np.concatenate([x_c, cat]))
                    self.H_precond = self._regularize(H_raw)

            new_pop = np.copy(self.pop)
            trial_f = np.full(self.pop_size, float('inf'))
            
            for i in range(self.pop_size):
                if self.evals >= self.budget: break
                
                # Select distinct parents
                idx = np.random.choice(self.pop_size, size=3, replace=False)
                r1, r2, r3 = idx
                
                # Curvature-scaled mutation
                diff = self.pop[r2] - self.pop[r3]
                if self.H_precond is not None:
                    diff = self.H_precond @ diff
                trial = self.pop[r1] + self.F * diff
                
                # Crossover: Continuous
                cr_mask = np.random.rand(18) < self.CR
                trial[:18] = np.where(cr_mask, trial[:18], self.pop[i, :18])
                trial[:18] = np.clip(trial[:18], -1.0, 1.0)
                
                # Crossover: Categorical (valid ID sampling)
                cat_mask = np.random.rand(6) < self.CR
                trial[18:24] = np.random.randint(0, 6, size=6) if np.any(cat_mask) else self.pop[i, 18:24].copy()
                
                # Evaluate
                trial_f[i] = self._evaluate(trial, func)
                if trial_f[i] < self.gbest_f:
                    self.gbest_f = trial_f[i]
                    self.gbest_idx = i
                    
                # Greedy Selection
                if trial_f[i] < self.pop_f[i]:
                    new_pop[i] = trial
                    self.pop_f[i] = trial_f[i]
                else:
                    new_pop[i] = self.pop[i]
                    
            self.pop = new_pop
            
            # Memetic Repair
            if hess_func is not None and self.evals < self.budget and gen % self.memetic_freq == 0:
                sorted_idx = np.argsort(self.pop_f)[:self.top_k]
                for idx in sorted_idx:
                    if self.evals >= self.budget: break
                    
                    x_c = self.pop[idx, :18].copy()
                    cat = self.pop[idx, 18:24].copy()
                    full_x = np.concatenate([x_c, cat])
                    H_reg = self._regularize(hess_func(full_x))
                    
                    def obj(xc): return func(np.concatenate([xc, cat]))
                    def jac(xc):
                        if grad_func is None: return np.zeros(18)
                        g = grad_func(np.concatenate([xc, cat]))
                        return g[:18]
                    def hess(xc): return H_reg
                    
                    if self.evals < self.budget:
                        res = minimize(
                            obj, x_c, jac=jac, hess=hess,
                            method='trust-constr',
                            bounds=[(-1.0, 1.0)] * 18,
                            options={'maxiter': 20}
                        )
                        if self.evals < self.budget:
                            cand = np.concatenate([res.x, cat])
                            cand_f = self._evaluate(cand, func)
                            if cand_f < self.pop_f[idx]:
                                self.pop[idx] = cand
                                self.pop_f[idx] = cand_f
                                if cand_f < self.gbest_f:
                                    self.gbest_f = cand_f
                                    self.gbest_idx = idx
                                if cand_f < self.best_f:
                                    self.best_f = cand_f
                                    self.best_x = cand

        return self.best_f, self.best_x