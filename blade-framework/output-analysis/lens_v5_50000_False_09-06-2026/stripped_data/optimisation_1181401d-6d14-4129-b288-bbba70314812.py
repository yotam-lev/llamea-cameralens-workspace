import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self._hess_func = None
        self.temp = 5.0
        self.H_reg = None
        self.H_inv_sqrt = None
        self.ls_trigger = False
        self.reservoir = []
        self.reservoir_f = []
        self.arch_cap = 12
        self.expl_steps = 0

    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _update_curvature(self, x):
        H = self._hess_func(x)
        eigs, V = np.linalg.eigh(H)
        shift = max(0, 1e-4 - eigs.min())
        self.H_reg = H + shift * np.eye(18)
        D_inv_sqrt = 1.0 / np.sqrt(np.clip(np.abs(eigs), 1e-4, None))
        self.H_inv_sqrt = V @ np.diag(D_inv_sqrt) @ V.T

    def _reservoir_update(self, x, f):
        if len(self.reservoir) < self.arch_cap:
            self.reservoir.append(x.copy())
            self.reservoir_f.append(f)
        elif f < self.reservoir_f[-1]:
            idx = np.argmin(self.reservoir_f)
            self.reservoir[idx] = x.copy()
            self.reservoir_f[idx] = f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self._hess_func = hess_func
        n_samples = 12
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        pop[18:24] = np.clip(np.round(pop[18:24]), 0, 5).astype(int)
        for i in range(n_samples):
            if self.evals >= self.budget: break
            f = self._evaluate(pop[i], func)
            self._reservoir_update(pop[i], f)

        current_idx = np.argmin(self.reservoir_f)
        current = self.reservoir[current_idx].copy()
        current_f = self.reservoir_f[current_idx]

        self.expl_steps = 0
        self.temp = 8.0

        while self.evals < self.budget:
            self.expl_steps += 1
            self.temp *= 0.995

            if self.ls_trigger or self.temp < 1.0:
                self.ls_trigger = False
                if self.H_reg is None and self._hess_func and self.evals < self.budget:
                    self._update_curvature(current)
                if self.H_reg is not None:
                    cat = np.clip(np.round(current[18:24]), 0, 5).astype(int)
                    res = minimize(
                        lambda xc: func(np.concatenate([xc, cat])),
                        current[:18], method='trust-constr', hess=lambda xc: self.H_reg,
                        bounds=[(-1.0, 1.0)]*18, options={'maxiter': 60, 'verbose': 0}
                    )
                    if res.success and self.evals < self.budget:
                        cand = np.concatenate([res.x, cat])
                        f_c = self._evaluate(cand, func)
                        if f_c < current_f:
                            current = cand
                            current_f = f_c
                            self.temp = min(10.0, self.temp * 2.0)
                self.expl_steps = 0
                continue

            step = np.random.standard_normal(self.dim)
            u = np.random.uniform(0, 1, self.dim)
            levy = step / (np.abs(u) + 1e-12) ** (1.0/1.5)
            levy[:18] = self.H_inv_sqrt @ levy[:18] if self.H_inv_sqrt is not None else levy[:18]
            levy[:18] *= self.temp

            prop = current + levy
            prop[:18] = np.clip(prop[:18], -1.0, 1.0)
            prop[18:24] = np.clip(np.round(prop[18:24]), 0, 5).astype(int)

            f_prop = self._evaluate(prop, func)
            self._reservoir_update(prop, f_prop)

            if f_prop < current_f or np.random.random() < np.exp(-(f_prop - current_f) / max(self.temp, 1e-8)):
                current_idx = np.argmin(self.reservoir_f)
                current = self.reservoir[current_idx].copy()
                current_f = self.reservoir_f[current_idx]
                self.ls_trigger = True

        return self.best_f, self.best_x