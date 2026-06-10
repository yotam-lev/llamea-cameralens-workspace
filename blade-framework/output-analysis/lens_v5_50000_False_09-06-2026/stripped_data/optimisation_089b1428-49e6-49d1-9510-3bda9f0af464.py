import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.hess_func = None
        self.H = None
        self.H_inv = None
        self.eigs = None
        self.vecs = None
        self.H_valid = False
        self.reg = 1e-3
        self.cat_basins = {}
        
        # Adaptive controls
        self.ls_interval = 12
        self.mutation_scale = 1.0
        self.stall_count = 0
        self.prev_best_f = float('inf')
        
    def _clip_and_map(self, x):
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        return x

    def _evaluate(self, x_raw, func):
        if self.evals >= self.budget:
            return float('inf')
        x = self._clip_and_map(x_raw)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _update_hessian(self, x):
        if self.evals >= self.budget: return
        try:
            H_raw = self.hess_func(x)
            eigs, vecs = np.linalg.eigh(H_raw)
            eigs_reg = np.abs(eigs) + self.reg
            self.H = vecs @ np.diag(eigs_reg) @ vecs.T
            self.H_inv = vecs @ np.diag(1.0 / eigs_reg) @ vecs.T
            self.eigs = eigs_reg
            self.vecs = vecs
            self.H_valid = True
            # Adaptive mutation scale based on stiffness
            cond = self.eigs[-1] / self.eigs[0] if self.eigs[0] > 1e-12 else 1e12
            self.mutation_scale = max(0.05, min(1.0, 1.0 / np.sqrt(cond)))
        except Exception:
            self.H_valid = False
            self.mutation_scale = 1.0

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        n_pop = 30
        pop = np.random.uniform(-1, 1, size=(n_pop, self.dim))
        vel = np.zeros_like(pop)
        pbest_x = pop.copy()
        pbest_f = np.full(n_pop, np.inf)
        gbest_idx = 0
        gbest_f = float('inf')
        
        for i in range(n_pop):
            f = self._evaluate(pop[i], func)
            pbest_f[i] = f
            if f < gbest_f:
                gbest_f = f
                gbest_idx = i
                
        it = 0
        while self.evals < self.budget:
            it += 1
            
            # Adaptive Stall Detection & Local Search Frequency
            if self.best_f < self.prev_best_f:
                self.stall_count = 0
                self.ls_interval = min(20, self.ls_interval + 1)
            else:
                self.stall_count += 1
                if self.stall_count >= 6:
                    self.ls_interval = max(4, self.ls_interval - 2)
            self.prev_best_f = self.best_f
            
            if self.evals < self.budget and (not self.H_valid or it % 8 == 0):
                if self.best_f < float('inf'):
                    self._update_hessian(self.best_x)
                    
            for i in range(n_pop):
                if self.evals >= self.budget: break
                
                r1, r2 = np.random.rand(), np.random.rand()
                wc, c1, c2 = 0.4, 1.4, 1.4
                
                if self.H_valid and grad_func is not None:
                    if self.evals < self.budget:
                        try:
                            g = grad_func(self.best_x)
                            n_step = -self.H_inv @ g
                            # Adaptive step scaling based on stiffness
                            n_step *= min(1.0, self.mutation_scale / (np.linalg.norm(n_step) + 1e-12))
                            pop[i][:18] += n_step * r1
                        except Exception:
                            pass
                            
                vel[i][:18] = wc * vel[i][:18] + c1 * r1 * (pbest_x[i][:18] - pop[i][:18]) + c2 * r2 * (pbest_x[gbest_idx][:18] - pop[i][:18])
                pop[i][:18] += vel[i][:18]
                
                curr_cat_float = pop[i][18:24].copy()
                cat_key = tuple(np.round(curr_cat_float).astype(int))
                
                if cat_key not in self.cat_basins:
                    self.cat_basins[cat_key] = pop[i][:18].copy()
                else:
                    if self.H_valid:
                        low_idx = np.argsort(self.eigs)[:2]
                        transition = np.zeros(18)
                        for idx in low_idx:
                            transition += self.vecs[:, idx]
                        self.cat_basins[cat_key] = 0.9 * self.cat_basins[cat_key] + 0.1 * pop[i][:18]
                        self.cat_basins[cat_key] += transition * 0.04
                        pop[i][:18] = 0.75 * pop[i][:18] + 0.25 * self.cat_basins[cat_key]
                        
                pop[i] = np.clip(pop[i], -1.0, 1.0)
                pop[i][18:24] = np.clip(np.round(pop[i][18:24]), 0, 5)
                
                # Adaptive discrete mutation probability
                if self.H_valid and np.random.rand() < 0.25 * self.mutation_scale:
                    min_eig_idx = np.argmin(self.eigs)
                    dir_d = self.vecs[:, min_eig_idx]
                    cat_pert = np.clip(curr_cat_float + dir_d[18:24] * np.sqrt(2.0 / (self.eigs[min_eig_idx] + 1e-8)), 0, 5)
                    test_x = pop[i].copy()
                    test_x[18:24] = np.round(cat_pert)
                    
                    if self.evals < self.budget:
                        f_test = self._evaluate(test_x, func)
                        f_curr = self._evaluate(pop[i], func)
                        if f_test < f_curr:
                            pop[i] = test_x
                
                f = self._evaluate(pop[i], func)
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_x[i] = pop[i].copy()
                if f < gbest_f:
                    gbest_f = f
                    gbest_idx = i
                    
            if self.evals < self.budget and it % self.ls_interval == 0:
                xb = pop[gbest_idx][:18].copy()
                cat_int = np.clip(np.round(pop[gbest_idx][18:24]), 0, 5).astype(int)
                if self.H_valid:
                    try:
                        def obj(xc): return func(np.concatenate([xc, cat_int]))
                        res = minimize(obj, xb, method='trust-constr', hess=lambda xc: self.H,
                                       bounds=[(-1.0, 1.0)]*18, options={'maxiter': 20})
                        if res.success:
                            cand = np.concatenate([res.x, cat_int])
                            f_c = self._evaluate(cand, func)
                            if f_c < gbest_f:
                                gbest_f = f_c
                                pop[gbest_idx] = cand
                                pbest_x[gbest_idx] = cand.copy()
                                pbest_f[gbest_idx] = f_c
                    except Exception:
                        pass
                        
        return self.best_f, self.best_x