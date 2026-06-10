import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.hess_freq = 12
        self.ls_freq = 25
        self.reg = 1e-3
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

    def _update_hessian(self, x):
        try:
            H = self.hess_func(x)
            eigs = np.linalg.eigvalsh(H)
            shift = max(0, self.reg - eigs.min())
            H_reg = H + shift * np.eye(18)
            self.H = H_reg
            self.H_inv = np.linalg.inv(H_reg)
            self.H_valid = True
        except:
            pass

    def _get_H_regularized(self, x):
        try:
            H = self.hess_func(x)
            eigs = np.linalg.eigvalsh(H)
            shift = max(0, self.reg - eigs.min())
            return H + shift * np.eye(18)
        except:
            return None

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
            
            if self.hess_func and (it % self.hess_freq == 0 or not self.H_valid):
                if self.evals < self.budget:
                    self._update_hessian(pop[gbest_idx])
                    
            w = 0.5 + 0.2 * np.random.rand()
            c1 = 1.4
            c2 = 1.4
            
            for i in range(n_pop):
                r1, r2 = np.random.rand(), np.random.rand()
                
                diff_g = pbest_x[gbest_idx] - pop[i]
                diff_p = pbest_x[i] - pop[i]
                
                # Hessian preconditioning for social term
                if self.H_valid and self.H_inv is not None:
                    diff_g = np.dot(self.H_inv, diff_g)
                    
                v = w * vel[i] + c1 * r1 * diff_p + c2 * r2 * diff_g
                vel[i] = v
                pop[i] += v
                
                f = self._evaluate(pop[i], func)
                
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_x[i] = pop[i].copy()
                
                if f < gbest_f:
                    gbest_f = f
                    gbest_idx = i
                    
            if self.evals < self.budget and it % self.ls_freq == 0:
                xb = pop[gbest_idx][:18].copy()
                cat_int = np.clip(np.round(pop[gbest_idx][18:24]), 0, 5).astype(int)
                H_ls = self._get_H_regularized(pop[gbest_idx])
                
                if H_ls is not None:
                    res = minimize(
                        lambda xc: func(np.concatenate([xc, cat_int])),
                        xb, method='trust-constr', hess=lambda xc: H_ls,
                        bounds=[(-1.0, 1.0)]*18, options={'maxiter': 40, 'verbose': 0}
                    )
                    if res.success and self.evals < self.budget:
                        cand = np.concatenate([res.x, cat_int])
                        f_c = self._evaluate(cand, func)
                        if f_c < gbest_f:
                            gbest_f = f_c
                            pop[gbest_idx] = cand
                            pbest_x[gbest_idx] = cand.copy()
                            pbest_f[gbest_idx] = f_c
                            
        return self.best_f, self.best_x