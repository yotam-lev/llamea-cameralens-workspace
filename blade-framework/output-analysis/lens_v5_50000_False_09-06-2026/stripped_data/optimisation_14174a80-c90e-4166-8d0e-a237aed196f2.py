import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.N = 64
        self.T = 1.0
        self.T_min = 1e-5
        self.c1, self.c2 = 1.5, 1.8
        self.H = np.eye(18)
        self.D_inv = np.eye(18)
        self.H_valid = False
        self.hess_func = None

    def _clip_map(self, x):
        xc = np.clip(x, -1.0, 1.0)
        xc[18:24] = np.clip(np.round(xc[18:24]), 0, 5).astype(int)
        return xc

    def _eval(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        xc = self._clip_map(x)
        f = func(xc)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = xc.copy()
        return f

    def _update_spectral(self, x):
        try:
            if self.hess_func is None: return
            H = self.hess_func(x)
            vals, vecs = np.linalg.eigh(H)
            vals_reg = np.abs(vals) + 1e-6
            self.D_inv = vecs @ np.diag(1.0 / vals_reg) @ vecs.T
            self.H = vecs @ np.diag(vals_reg) @ vecs.T
            self.H_valid = True
        except Exception:
            self.H_valid = False

    def _cooling(self):
        progress = self.evals / self.budget
        self.T = self.T_min + (1.0 - self.T_min) * np.exp(-4 * progress)

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func

        pop = np.random.uniform(-1, 1, size=(self.N, self.dim))
        pbest_x = np.zeros((self.N, self.dim))
        pbest_f = np.full(self.N, float('inf'))
        v = np.random.uniform(-0.5, 0.5, (self.N, self.dim))

        for i in range(self.N):
            if self.evals >= self.budget: break
            f = self._eval(pop[i], func)
            if f < pbest_f[i]:
                pbest_f[i] = f
                pbest_x[i] = pop[i].copy()

        if self.evals >= self.budget:
            return self.best_f, self.best_x

        self._update_spectral(self.best_x)
        self._cooling()

        while self.evals < self.budget:
            w = self.T * 0.7 + 0.15
            for i in range(self.N):
                if self.evals >= self.budget: break

                diff_c = self.best_x[:18] - pop[i, :18]
                dir_c = self.D_inv @ diff_c if self.H_valid else diff_c

                v[i, :18] = w * v[i, :18] + self.c1 * np.random.rand(18) * (pbest_x[i, :18] - pop[i, :18]) + self.c2 * np.random.rand(18) * dir_c
                v[i, 18:24] *= w

                pop[i, :18] += v[i, :18]

                if np.random.rand() < self.T:
                    pop[i, 18:24] = np.random.randint(0, 6, 6)
                elif np.random.rand() < 0.1 * self.T:
                    pop[i, 18:24] = pbest_x[i, 18:24].copy()

                f = self._eval(pop[i], func)
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_x[i] = pop[i].copy()

            if self.evals % 20 == 0 and self.evals < self.budget:
                self._update_spectral(self.best_x)
                self._cooling()

            if self.H_valid and self.evals % 40 == 0 and self.evals < self.budget:
                cat_fixed = self.best_x[18:24].copy()
                try:
                    def ref_func(xc):
                        if self.evals >= self.budget: return float('inf')
                        full = np.concatenate([xc, cat_fixed])
                        return self._eval(full, func)
                    res = minimize(ref_func, self.best_x[:18], method='trust-constr',
                                   bounds=[(-1.0, 1.0)] * 18, hess=lambda xc: self.H,
                                   options={'maxiter': 25, 'verbose': 0})
                    if res.success:
                        cand = np.concatenate([res.x, cat_fixed])
                        self._eval(cand, func)
                except Exception:
                    pass

        return self.best_f, self.best_x