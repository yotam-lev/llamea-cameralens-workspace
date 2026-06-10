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
        self.ls_freq = 25
        self.reg = 1e-4
        self.H = None
        self.H_inv = None
        self.eigs = None
        self.vecs = None
        self.H_valid = False
        self.last_cat = None
        
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
            self.H_inv = vecs @ np.diag(1.0 / eigs_reg) @ vecs.T
            self.eigs = eigs_reg
            self.vecs = vecs
            self.H_valid = True
        except:
            pass

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        n_pop = 24
        pop = np.random.uniform(-1, 1, size=(n_pop, self.dim))
        vel = np.zeros_like(pop)
        self.last_cat = pop[0][18:24].copy()
        
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
                    
            w = 0.5
            c1 = 1.2
            c2 = 1.2
            
            for i in range(n_pop):
                r1, r2 = np.random.rand(), np.random.rand()
                
                diff_g = pbest_x[gbest_idx] - pop[i]
                diff_p = pbest_x[i] - pop[i]
                
                if self.H_valid:
                    diff_g = self.H_inv @ diff_g
                    
                vel[i] = w * vel[i] + c1 * r1 * diff_p + c2 * r2 * diff_g
                pop[i] += vel[i]
                
                # Plateau escape via low-curvature injection
                if self.H_valid and grad_func:
                    try:
                        g = grad_func(pop[i])
                        gn = np.linalg.norm(g)
                        hn = np.linalg.norm(self.H)
                        if hn > 0 and gn < 1e-3 * hn:
                            min_idx = np.argmin(self.eigs)
                            if self.eigs[min_idx] > 1e-12:
                                jump = np.sqrt(2.0 / self.eigs[min_idx]) * self.vecs[:, min_idx]
                                pop[i] += jump
                    except:
                        pass
                        
                # Material context reset
                curr_cat = pop[i][18:24]
                if self.last_cat is not None and not np.array_equal(curr_cat, self.last_cat):
                    vel[i] = np.zeros(self.dim)
                self.last_cat = curr_cat.copy()
                
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
                H_ls = self.H if self.H_valid else None
                
                if H_ls is not None:
                    try:
                        res = minimize(
                            lambda xc: func(np.concatenate([xc, cat_int])),
                            xb, method='trust-constr', hess=lambda xc: H_ls,
                            bounds=[(-1.0, 1.0)]*18, options={'maxiter': 25}
                        )
                        if res.success and self.evals < self.budget:
                            cand = np.concatenate([res.x, cat_int])
                            f_c = self._evaluate(cand, func)
                            if f_c < gbest_f:
                                gbest_f = f_c
                                pop[gbest_idx] = cand
                                pbest_x[gbest_idx] = cand.copy()
                                pbest_f[gbest_idx] = f_c
                    except:
                        pass
                            
        return self.best_f, self.best_x