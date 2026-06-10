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
        self.reg = 1e-4
        
        # Adaptive controls
        self.ls_interval = 15
        self.mutation_scale = 1.0
        self.stall_count = 0
        self.prev_best_f = float('inf')
        self.ls_counter = 0
        
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
            # Dynamic regularization based on condition number
            cond = eigs[-1] / eigs[0] if eigs[0] > 1e-12 else 1e12
            self.reg = self.reg * (1 + 0.1 * np.log1p(cond))
            eigs_reg = np.abs(eigs) + self.reg
            self.H = vecs @ np.diag(eigs_reg) @ vecs.T
            self.H_inv = vecs @ np.diag(1.0 / eigs_reg) @ vecs.T
            self.eigs = eigs_reg
            self.vecs = vecs
            self.H_valid = True
            # Mutation scale depends on stiffness and budget progress
            progress = self.evals / self.budget
            stiffness_factor = 1.0 / np.sqrt(cond) if cond > 1.0 else 1.0
            budget_factor = 1.0 - 0.8 * progress
            self.mutation_scale = max(0.01, min(1.0, stiffness_factor * budget_factor))
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
            progress = self.evals / self.budget
            
            # Adaptive Stall Detection & Local Search Frequency
            if self.best_f < self.prev_best_f:
                self.stall_count = 0
                self.ls_interval = min(25, self.ls_interval + 2)
            else:
                self.stall_count += 1
                self.ls_interval = max(3, self.ls_interval - 1)
            self.prev_best_f = self.best_f
            
            # Budget-aware frequency acceleration
            base_interval = 20 * (1 - progress * 0.8)
            self.ls_interval = max(3, int(min(self.ls_interval, base_interval)))
            
            # Adaptive Hessian update frequency
            if self.evals < self.budget and (not self.H_valid or it % self.ls_interval == 0):
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
                            n_step *= min(1.0, self.mutation_scale / (np.linalg.norm(n_step) + 1e-12))
                            pop[i][:18] += n_step * r1
                        except Exception:
                            pass
                            
                vel[i][:18] = wc * vel[i][:18] + c1 * r1 * (pbest_x[i][:18] - pop[i][:18]) + c2 * r2 * (pbest_x[gbest_idx][:18] - pop[i][:18])
                pop[i][:18] += vel[i][:18]
                
                # Adaptive categorical perturbation
                if self.H_valid and np.random.rand() < 0.2 * (1 - progress):
                    min_eig_idx = np.argmin(self.eigs)
                    dir_d = self.vecs[:, min_eig_idx]
                    cat_pert = np.clip(pop[i][18:24] + dir_d[18:24] * np.sqrt(2.0 / (self.eigs[min_eig_idx] + 1e-8)), 0, 5)
                    test_x = pop[i].copy()
                    test_x[18:24] = np.round(cat_pert)
                    
                    if self.evals < self.budget:
                        f_test = self._evaluate(test_x, func)
                        f_curr = self._evaluate(pop[i], func)
                        if f_test < f_curr:
                            pop[i] = test_x
                
                pop[i] = np.clip(pop[i], -1.0, 1.0)
                pop[i][18:24] = np.clip(np.round(pop[i][18:24]), 0, 5)
                
                f = self._evaluate(pop[i], func)
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_x[i] = pop[i].copy()
                if f < gbest_f:
                    gbest_f = f
                    gbest_idx = i
                    
            if self.evals < self.budget and (it % self.ls_interval == 0):
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