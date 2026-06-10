import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.pop_size = 40
        self.F = 0.6
        self.CR = 0.8
        self.hess_interval = 15
        self.ls_interval = 30
        self.reg = 1e-2
        self.H = None
        self.H_inv = None
        self.H_valid = False

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

    def _ensure_hessian(self, x):
        if self.H_valid: return
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

    def _apply_hessian_scaling(self, diff):
        if self.H_valid and self.H_inv is not None:
            try:
                return np.linalg.solve(self.H_inv, diff[:18])
            except:
                pass
        return diff[:18]

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        pop = np.random.uniform(-1, 1, size=(self.pop_size, self.dim))
        fvals = np.array([self._evaluate(ind, func) for ind in pop])
        
        best_idx = np.argmin(fvals)
        gbest = pop[best_idx].copy()
        gbest_f = fvals[best_idx]
        
        iteration = 0
        while self.evals < self.budget:
            iteration += 1
            
            if self.hess_func and (iteration % self.hess_interval == 0 or not self.H_valid):
                if self.evals < self.budget:
                    self._ensure_hessian(gbest)
                    
            if self.evals < self.budget and iteration % self.ls_interval == 0:
                cat_int = np.clip(np.round(gbest[18:24]), 0, 5).astype(int)
                xb = gbest[:18].copy()
                if self.H_valid:
                    H_ls = self.H.copy()
                    res = minimize(
                        lambda xc: func(np.concatenate([xc, cat_int])),
                        xb, method='trust-constr', hess=lambda xc: H_ls,
                        bounds=[(-1.0, 1.0)]*18, options={'maxiter': 50, 'verbose': 0}
                    )
                    if res.success:
                        cand = np.concatenate([res.x, cat_int])
                        f_c = self._evaluate(cand, func)
                        if f_c < gbest_f:
                            gbest_f = f_c
                            gbest = cand
                            pop[best_idx] = cand
                            fvals[best_idx] = f_c

            for _ in range(self.pop_size // 2):
                idx = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = pop[idx]
                
                diff = b - c
                diff_scaled = self._apply_hessian_scaling(diff)
                
                v_mut = a[:18] + self.F * diff_scaled
                v_mut = np.concatenate([v_mut, a[18:24] + self.F * diff[18:24]])
                
                cr_mask = np.random.rand(self.dim) < self.CR
                v_recomb = np.where(cr_mask, v_mut, a)
                
                v_recomb[18:24] = np.clip(np.round(v_recomb[18:24]), 0, 5)
                
                f_v = self._evaluate(v_recomb, func)
                
                target_idx = np.random.randint(self.pop_size)
                if f_v < fvals[target_idx]:
                    pop[target_idx] = v_recomb
                    fvals[target_idx] = f_v
                    if f_v < gbest_f:
                        gbest_f = f_v
                        gbest = v_recomb.copy()
                        best_idx = target_idx

        return self.best_f, self.best_x