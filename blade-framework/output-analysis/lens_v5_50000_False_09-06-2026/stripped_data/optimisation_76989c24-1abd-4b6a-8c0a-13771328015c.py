import numpy as np
from scipy.optimize import minimize
import cma

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
        x_clip = np.clip(x.copy(), -1.0, 1.0)
        x_clip[18:24] = np.clip(np.round(x_clip[18:24]), 0, 5).astype(int)
        f = func(x_clip)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x_clip.copy()
        return f

    def _regularize_hessian(self, H):
        eigvals = np.linalg.eigvalsh(H)
        shift = max(0.0, -eigvals.min() + 1e-4)
        return H + (shift + 1.0) * np.eye(H.shape[0])

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        dim_cont = 18
        sigma0 = 0.5
        x0 = np.zeros(dim_cont)
        es = cma.CMAEvolutionStrategy(x0, sigma0)
        
        cat_theta = np.ones(6) / 6.0
        mu = max(4, 4 + int(3 + np.log(dim_cont)))
        
        local_search_count = 0
        
        while self.evals < self.budget:
            candidates = es.ask()
            fit_vals = np.empty(len(candidates))
            
            for i, c in enumerate(candidates):
                if self.evals >= self.budget:
                    fit_vals[i] = float('inf')
                    break
                full_x = np.zeros(self.dim)
                full_x[:18] = c
                
                cat_samples = np.random.choice(6, size=6, p=cat_theta)
                full_x[18:24] = cat_samples
                
                f = self._evaluate(full_x, func)
                fit_vals[i] = f
                
                if np.isfinite(f):
                    elite_mask = fit_vals < np.mean(fit_vals)
                    cat_theta = np.zeros(6)
                    if elite_mask.any():
                        elite_cats = full_x[elite_mask][:, 18:24]
                        counts = np.bincount(elite_cats.flatten(), minlength=6)
                        cat_theta = np.exp(counts / (np.max(counts) + 1e-3))
                        cat_theta /= cat_theta.sum()
            
            if np.all(fit_vals == float('inf')):
                break
                
            es.tell([c for i, c in enumerate(candidates) if np.isfinite(fit_vals[i])], 
                    fit_vals[np.isfinite(fit_vals)])
            
            es.disp()
            
            if es.sigma < 1e-4 and local_search_count < 5:
                best_idx = np.argmin(fit_vals)
                if np.isfinite(fit_vals[best_idx]):
                    best_c = candidates[best_idx]
                    best_cat = full_x[best_idx, 18:24]
                    
                    if hess_func is not None:
                        full_b = np.empty(self.dim)
                        full_b[:18] = best_c
                        full_b[18:24] = best_cat
                        H = hess_func(full_b)
                        H_reg = self._regularize_hessian(H)
                        
                        def sub_hess(x): return H_reg
                        def sub_grad(x):
                            if grad_func:
                                fb = np.empty(self.dim)
                                fb[:18] = x
                                fb[18:24] = best_cat
                                return grad_func(fb)[:18]
                            return np.zeros(18)
                            
                        bounds = [(-1.0, 1.0)] * 18
                        res = minimize(lambda x: self._evaluate(np.concatenate([x, best_cat]), func), 
                                       best_c, jac=sub_grad, hess=sub_hess, 
                                       method='trust-constr', bounds=bounds, options={'maxiter': 30})
                        
                        if res.fun < fit_vals[best_idx]:
                            final = np.empty(self.dim)
                            final[:18] = res.x
                            final[18:24] = best_cat
                            f_ref = self._evaluate(final, func)
                            if f_ref < self.best_f:
                                self.best_f = f_ref
                                self.best_x = final
                            fit_vals[best_idx] = f_ref
                            es.tell([res.x], [f_ref])
                            local_search_count += 1
            
        return self.best_f, self.best_x