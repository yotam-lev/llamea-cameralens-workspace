import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.f = 0.8
        self.cr = 0.9
        self.n_pop = 40
        self.hess_freq = 15
        self.ls_freq = 20
        self.reg = 1e-2
        self.H = None
        self.H_inv = None
        self.H_valid = False
        self.hess_func = None

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
            H = self.hess_func(x)
            eigs = np.linalg.eigvalsh(H)
            shift = max(0, self.reg - eigs.min())
            H_reg = H + shift * np.eye(18)
            self.H = H_reg
            self.H_inv = np.linalg.inv(H_reg)
            self.H_valid = True
        except Exception:
            pass

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        gbest_idx = 0
        gbest_f = float('inf')
        
        for i in range(self.n_pop):
            f = self._evaluate(pop[i], func)
            if f < gbest_f:
                gbest_f = f
                gbest_idx = i

        it = 0
        while self.evals < self.budget:
            it += 1
            if self.hess_func and (it % self.hess_freq == 0 or not self.H_valid):
                if self.evals < self.budget:
                    self._update_hessian(pop[gbest_idx])

            new_pop = np.zeros_like(pop)
            for i in range(self.n_pop):
                # Hessian-Adaptive Mutation
                r1, r2, r3 = np.random.choice(self.n_pop, 3, replace=False)
                diff_cont = pop[r1, :18] - pop[r2, :18]
                
                if self.H_valid and self.H_inv is not None:
                    d_cont = np.dot(self.H_inv, diff_cont)
                else:
                    d_cont = diff_cont

                v_cont = pop[r3, :18] + self.f * d_cont
                v_cat = np.clip(np.round(pop[r3, 18:24]), 0, 5).astype(int)
                
                # Crossover (Integer-aware for categorical dims)
                trial_cont = np.copy(v_cont)
                trial_cat = np.copy(v_cat)
                cross_idx = np.random.rand(self.dim) < self.cr
                cross_idx[18:24] = False 
                trial_cont[cross_idx[:18]] = pop[i, :18][cross_idx[:18]]
                trial_cat[cross_idx[18:24]] = pop[i, 18:24][cross_idx[18:24]].astype(int)

                # Boundary & Mapping
                trial_cont = np.clip(trial_cont, -1.0, 1.0)
                trial_cat = np.clip(np.round(trial_cat), 0, 5).astype(int)
                
                trial = np.concatenate([trial_cont, trial_cat])
                
                f = self._evaluate(trial, func)
                if f <= self._evaluate(pop[i], func):
                    new_pop[i] = trial
                    if f < gbest_f:
                        gbest_f = f
                        gbest_idx = i
                else:
                    new_pop[i] = pop[i]

            pop = new_pop

            # Trust-Region Refinement
            if self.evals < self.budget and it % self.ls_freq == 0:
                xb = pop[gbest_idx][:18].copy()
                cat_int = pop[gbest_idx][18:24].copy()
                if self.H_valid and self.H_inv is not None:
                    res = minimize(
                        lambda xc: func(np.concatenate([xc, cat_int])),
                        xb, method='trust-constr',
                        hess=lambda xc: self.H,
                        bounds=[(-1.0, 1.0)]*18, 
                        options={'maxiter': 30, 'verbose': 0}
                    )
                    if res.success and self.evals < self.budget:
                        cand = np.concatenate([res.x, cat_int])
                        f_c = self._evaluate(cand, func)
                        if f_c < gbest_f:
                            gbest_f = f_c
                            gbest_idx = gbest_idx # Update index if needed
                            pop[gbest_idx] = cand
        return self.best_f, self.best_x