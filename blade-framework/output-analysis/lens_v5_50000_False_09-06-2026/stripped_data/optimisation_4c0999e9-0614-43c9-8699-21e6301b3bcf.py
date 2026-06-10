import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.cat_mean = np.zeros(6)
        self.cat_cov = np.eye(6) * 0.5
        self.H_pd = None
        self.H_valid = False

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

        if self.evals >= self.budget:
            return self.best_f, self.best_x

        n_samples = 30
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        for i in range(n_samples):
            if self.evals >= self.budget: break
            self._evaluate(pop[i])

        x_curr = self.best_x.copy()
        f_curr = self.best_f

        while self.evals < self.budget:
            # Hessian update & PD enforcement
            if not self.H_valid or self.evals % 20 == 0:
                if self.evals >= self.budget: break
                try:
                    H = self.hess_func(self.best_x)
                    eigs, V = np.linalg.eigh(H)
                    self.H_pd = V @ np.diag(np.abs(eigs) + 1e-6) @ V.T
                    self.H_valid = True
                except Exception:
                    pass

            # 1. Categorical Optimization (Curvature-Adaptive Covariance)
            if self.evals < self.budget:
                grad_cat = np.zeros(6)
                for d in range(6):
                    x_up = x_curr.copy(); x_up[18+d] += 1.0
                    if self.evals >= self.budget: break
                    f_up = self._evaluate(x_up)
                    x_down = x_curr.copy(); x_down[18+d] -= 1.0
                    if self.evals >= self.budget: break
                    f_down = self._evaluate(x_down)
                    grad_cat[d] = (f_up - f_down) / 2.0

                # High continuous condition number -> conservative categorical steps
                cond_num = np.linalg.cond(self.H_pd) if self.H_valid else 1.0
                step_scale = min(1.0, 10.0 / max(cond_num, 1.0))

                self.cat_cov = 0.9 * self.cat_cov + 0.1 * np.eye(6)
                cat_prop = self.cat_mean + np.sqrt(self.cat_cov * step_scale) @ np.random.randn(6)
                cat_prop = np.clip(np.round(cat_prop), 0, 5).astype(int)

                x_cat_new = x_curr.copy()
                x_cat_new[18:24] = cat_prop
                if self.evals >= self.budget: break
                f_cat_new = self._evaluate(x_cat_new)

                if f_cat_new < f_curr:
                    f_curr = f_cat_new
                    x_curr[:18] = self.best_x[:18]
                    x_curr = np.clip(x_curr, -1, 1)
                    x_curr[18:24] = cat_prop
                    step = cat_prop - self.cat_mean
                    self.cat_mean += 0.5 * step
                    self.cat_cov = (1 - 0.5) * self.cat_cov + 0.5 * np.outer(step, step)

            # 2. Continuous Optimization (Conditioned by Categorical Uncertainty)
            fixed_cat = x_curr[18:24].copy()
            def cont_obj(xc):
                if self.evals >= self.budget: return float('inf')
                return self._evaluate(np.concatenate([xc, fixed_cat]))

            def cont_hess(xc):
                if not self.H_valid: return np.eye(18) * 1e3
                det_cov = np.linalg.det(np.maximum(self.cat_cov, 1e-3))
                scale = max(0.1, min(10.0, 1.0 / det_cov))
                return self.H_pd * scale

            if self.H_valid and self.evals < self.budget:
                try:
                    res = minimize(cont_obj, x_curr[:18], method='trust-constr',
                                   hess=cont_hess, bounds=[(-1.0, 1.0)]*18,
                                   options={'maxiter': 20, 'verbose': 0})
                    if self.evals < self.budget and res.fun < f_curr:
                        f_curr = res.fun
                        x_curr = np.concatenate([res.x, fixed_cat])
                        x_curr = np.clip(x_curr, -1, 1)
                        x_curr[18:24] = np.clip(np.round(x_curr[18:24]), 0, 5).astype(int)
                        self.cat_mean = 0.8 * self.cat_mean + 0.2 * x_curr[18:24].astype(float)
                except Exception:
                    pass

        return self.best_f, self.best_x