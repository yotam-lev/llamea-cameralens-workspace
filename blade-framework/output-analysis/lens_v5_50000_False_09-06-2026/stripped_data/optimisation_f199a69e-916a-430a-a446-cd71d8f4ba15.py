import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)

    def _evaluate(self, x):
        if self.evals >= self.budget:
            return float('inf')
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

        x = np.random.uniform(-1, 1, size=self.dim)
        g_best = x.copy()
        g_f = self._evaluate(g_best)

        while self.evals < self.budget:
            # 1. Curvature Analysis & PD Enforcement
            if self.evals >= self.budget: break
            H = self.hess_func(x)
            eig_vals, eig_vecs = np.linalg.eigh(H)
            # Strict PD regularization via eigenvalue absolute mapping
            H_reg = eig_vecs @ np.diag(np.abs(eig_vals) + 1e-5) @ eig_vecs.T
            cond = eig_vals.max() / max(np.abs(eig_vals.min()), 1e-9)

            # 2. Dynamic Coupling: Landscape stiffness gates discrete exploration
            is_flexible = cond < 25.0
            mut_prob = 0.35 if is_flexible else 0.02

            # 3. Continuous Optimization conditioned on current discrete state
            def cont_obj(xc):
                xc_full = np.concatenate([xc, x[18:24]])
                return self._evaluate(xc_full)

            if self.evals < self.budget and self.evals % 10 == 0:
                res = minimize(cont_obj, x[:18], method='trust-constr',
                               hess=lambda xc: H_reg, bounds=[(-1.0, 1.0)]*18,
                               options={'maxiter': 12, 'verbose': 0})
                if res.success:
                    cand = np.concatenate([res.x, x[18:24]])
                    f_c = self._evaluate(cand)
                    if f_c < g_f:
                        g_f = f_c
                        g_best = cand.copy()
                        x = cand.copy()
                    else:
                        x[:18] = res.x
            else:
                # Curvature-guided geometric step
                dx = -np.linalg.solve(H_reg, x[:18] - g_best[:18]) * 0.35
                x[:18] += dx

            # 4. Discrete Update driven by Hessian-aligned sensitivity
            if self.evals < self.budget and np.random.rand() < mut_prob:
                for k in range(6):
                    if is_flexible or np.random.rand() < 0.1:
                        x_probe = x.copy()
                        x_probe[18+k] = np.clip(x_probe[18+k] + 1, 0, 5)
                        f_probe = self._evaluate(x_probe)
                        if f_probe < g_f:
                            g_f = f_probe
                            g_best = x_probe.copy()
                            x = x_probe.copy()
                        elif is_flexible and np.random.rand() < 0.4:
                            # Accept marginal worsening only in geometrically stable regions
                            x[18+k] = x_probe[18+k]

            x = np.clip(x, -1.0, 1.0)
            x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
            f = self._evaluate(x)
            if f < g_f:
                g_f = f
                g_best = x.copy()

        return self.best_f, self.best_x