import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.n_pop = 45
        self.F = 0.8
        self.CR = 0.9
        self.H_inv = None
        self.H_pd = None
        self.H_valid = False
        self.cat_ids = np.arange(6)

    def _evaluate(self, x):
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = self.func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.func = func
        self.hess_func = hess_func
        
        pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        f_pop = np.array([self._evaluate(pop[i]) for i in range(self.n_pop)])
        
        while self.evals < self.budget:
            # 1. Hessian Update & Regularization
            if not self.H_valid or self.evals % 20 == 0:
                try:
                    H = self.hess_func(self.best_x)
                    eigs, V = np.linalg.eigh(H)
                    # PD enforcement via absolute eigenvalues
                    self.H_pd = V @ np.diag(np.abs(eigs) + 1e-5) @ V.T
                    self.H_inv = V @ np.diag(1.0 / (np.abs(eigs) + 1e-5)) @ V.T
                    self.H_valid = True
                except Exception:
                    self.H_valid = False

            # Adaptive parameters based on stagnation
            stagnation = max(1e-9, self.best_f - np.min(f_pop))
            self.F = 0.6 + 0.4 * np.exp(-stagnation * 50)
            self.CR = 0.75 + 0.15 * np.cos(np.pi * self.evals / self.budget)
            
            # 2. Curvature-Adaptive Differential Evolution
            for i in range(self.n_pop):
                idx = np.random.choice(self.n_pop, 3, replace=False)
                while np.any(idx == i): idx = np.random.choice(self.n_pop, 3, replace=False)
                
                x_r1, x_r2, x_r3 = pop[idx[0]], pop[idx[1]], pop[idx[2]]
                diff = x_r2 - x_r3
                
                # Hessian Preconditioning: Scale difference by inverse curvature
                if self.H_valid:
                    diff[:18] = self.H_inv @ diff[:18]
                else:
                    diff[:18] *= 0.5
                    
                v_mut = pop[i] + self.F * diff
                
                # Crossover
                cr_mask = np.random.rand(self.dim) < self.CR
                x_trial = np.where(cr_mask, v_mut, pop[i])
                
                # Simulated Annealing-style Categorical Mutation
                for d in range(18, 24):
                    if np.random.rand() < 0.12:
                        curr = int(np.clip(np.round(x_trial[d]), 0, 5))
                        cand = self.cat_ids[self.cat_ids != curr]
                        x_trial[d] = np.random.choice(cand)
                        
                f_trial = self._evaluate(x_trial)
                
                # Selection
                if f_trial < f_pop[i]:
                    pop[i] = x_trial
                    f_pop[i] = f_trial
                    if f_trial < self.best_f:
                        self.best_f = f_trial
                        self.best_x = x_trial

            # 3. Hybridization: Trust-Region Geometry Optimization
            if self.evals < self.budget and self.H_valid:
                top_idx = np.argsort(f_pop)[:5]
                for idx in top_idx:
                    if self.evals >= self.budget: break
                    
                    fixed_cats = pop[idx][18:24]
                    def local_obj(xc):
                        xc = np.clip(xc, -1.0, 1.0)
                        return self._evaluate(np.concatenate([xc, fixed_cats]))
                    
                    res = minimize(local_obj, pop[idx][:18], method='trust-constr',
                                 hess=lambda x: self.H_pd, bounds=[(-1, 1)]*18,
                                 options={'maxiter': 30, 'verbose': 0})
                    
                    if res.success and res.fun < f_pop[idx]:
                        pop[idx] = np.concatenate([res.x, fixed_cats])
                        f_pop[idx] = res.fun
                        if res.fun < self.best_f:
                            self.best_f = res.fun
                            self.best_x = pop[idx]
                            
            # 4. Periodic Greedy Categorical Refinement on Global Best
            if self.evals < self.budget and self.evals % 25 == 0:
                g_best = self.best_x.copy()
                for d in range(18, 24):
                    best_d_f = self.best_f
                    for val in self.cat_ids:
                        cat_test = g_best.copy()
                        cat_test[d] = val
                        if self.evals < self.budget:
                            f_test = self._evaluate(cat_test)
                            if f_test < best_d_f:
                                best_d_f = f_test
                                g_best[d] = val
                if best_d_f < self.best_f:
                    self.best_f = best_d_f
                    self.best_x = g_best

        return self.best_f, self.best_x