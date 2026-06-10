import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        # Reduced population for resource safety
        self.pop_size = 25
        self.cont_pop = np.random.uniform(-1, 1, size=(self.pop_size, 18))
        self.disc_pop = np.random.randint(0, 6, size=(self.pop_size, 6))
        self.fitness = np.full(self.pop_size, float('inf'))
        
        # Adaptive state
        self.prev_best_f = float('inf')
        self.smoothed_improvement = 1e-8
        self.alpha_improve = 0.1
        self.hessian_cond = 1.0
        self.H_reg = np.eye(18)
        
        # Adaptive parameters
        self.ls_freq = 5
        self.mut_scale = 0.1
        self.ls_budget = 10

    def _clip_and_eval(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        x_safe = x.copy()
        x_safe[:18] = np.clip(x_safe[:18], -1.0, 1.0)
        x_safe[18:24] = np.clip(np.round(x_safe[18:24]), 0, 5).astype(int)
        f = func(x_safe)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x_safe.copy()
        return f

    def _adaptive_controller(self, improve_rate, cond_num, budget_frac):
        if improve_rate < 1e-6:
            self.mut_scale = np.clip(self.mut_scale * 1.5, 0.1, 1.0)
            self.ls_freq = max(2, self.ls_freq // 2)
        else:
            self.mut_scale = np.clip(self.mut_scale * 0.8, 0.01, 0.5)
            self.ls_freq = min(20, self.ls_freq + 1)
            
        if cond_num > 1e4:
            self.ls_freq = max(3, self.ls_freq // 2)
            
        if budget_frac < 0.2:
            self.ls_freq = max(2, self.ls_freq // 2)
            self.mut_scale *= 0.9

    def _local_search(self, xc, cat, func, H_reg):
        if self.evals >= self.budget:
            return xc, float('inf')
            
        def obj(c):
            return func(np.concatenate([c, cat]))
            
        hess_mat = H_reg if H_reg is not None else np.eye(18)
        bounds = [(-1.0, 1.0)] * 18
        res = minimize(obj, xc, method='trust-constr', bounds=bounds, 
                       hess=lambda x: hess_mat, options={'maxiter': self.ls_budget, 'verbose': 0})
        
        new_c = np.clip(res.x, -1.0, 1.0)
        f_val = self._clip_and_eval(np.concatenate([new_c, cat]), func)
        return new_c, f_val

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        for i in range(self.pop_size):
            x = np.concatenate([self.cont_pop[i], self.disc_pop[i]])
            self.fitness[i] = self._clip_and_eval(x, func)
            if self.evals >= self.budget: return self.best_f, self.best_x
            
        self.prev_best_f = self.best_f

        while self.evals < self.budget:
            improve = max(1e-12, self.prev_best_f - self.best_f)
            self.smoothed_improvement = (1-self.alpha_improve)*self.smoothed_improvement + self.alpha_improve*improve
            frac = 1.0 - self.evals / self.budget
            self._adaptive_controller(self.smoothed_improvement, self.hessian_cond, frac)
            self.prev_best_f = self.best_f
            
            if self.evals % 20 == 0 and hess_func is not None:
                best_idx = np.argmin(self.fitness)
                x_probe = np.concatenate([self.cont_pop[best_idx], self.disc_pop[best_idx]])
                if self.evals < self.budget:
                    H_raw = hess_func(x_probe)
                    eigs, Q = np.linalg.eigh(H_raw[:18, :18])
                    min_abs = np.min(np.abs(eigs))
                    self.hessian_cond = np.max(np.abs(eigs)) / (min_abs + 1e-8)
                    self.H_reg = Q @ np.diag(np.abs(eigs) + 1e-6) @ Q.T

            if self.evals < self.budget and self.evals % self.ls_freq == 0:
                best_idx = np.argmin(self.fitness)
                new_c, new_f = self._local_search(self.cont_pop[best_idx], self.disc_pop[best_idx], func, self.H_reg)
                if self.evals < self.budget and new_c is not None:
                    self.cont_pop[best_idx] = new_c
                    self.fitness[best_idx] = new_f

            for i in range(self.pop_size):
                if self.evals >= self.budget: break
                if np.random.rand() < 0.3:
                    self.disc_pop[i] = np.clip(self.disc_pop[i] + np.random.randint(-1, 2, size=6), 0, 5)
                    self.fitness[i] = float('inf')
                self.cont_pop[i] += np.random.normal(0, self.mut_scale, size=18)
                if np.isinf(self.fitness[i]):
                    x = np.concatenate([self.cont_pop[i], self.disc_pop[i]])
                    self.fitness[i] = self._clip_and_eval(x, func)
                    if self.evals >= self.budget: break

        return self.best_f, self.best_x