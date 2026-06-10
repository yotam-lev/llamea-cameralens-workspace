import numpy as np

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
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        x_curr = np.random.uniform(-1, 1, self.dim)
        f_curr = self._evaluate(x_curr, func)

        T = 1.0
        T_min = 1e-4
        cool_rate = 0.994
        max_iter = self.budget // 4

        for _ in range(max_iter):
            if self.evals >= self.budget: break
            if T < T_min:
                T = 1.0
                continue

            if hess_func is not None:
                full_x = np.concatenate([x_curr[:18], np.clip(np.round(x_curr[18:24]), 0, 5).astype(int)])
                H_raw = hess_func(full_x)
                eigs, Q = np.linalg.eigh(H_raw)
                H_reg = np.abs(eigs) + 1e-6
                inv_sqrt = Q @ np.diag(1.0 / np.sqrt(H_reg)) @ Q.T
                noise_c = np.random.randn(18)
                step_c = inv_sqrt @ noise_c * (0.5 / (np.trace(H_reg) + 1e-8))
                step_cat = np.random.randn(6) * 0.6 / (np.sqrt(H_reg[0, 0]) + 1.0)
            else:
                step_c = np.random.randn(18) * 0.3
                step_cat = np.random.randn(6) * 0.5

            x_cand = x_curr + np.concatenate([step_c, step_cat])
            f_cand = self._evaluate(x_cand, func)

            if f_cand < f_curr:
                x_curr = x_cand
                f_curr = f_cand
            elif np.random.rand() < np.exp(-(f_cand - f_curr) / T):
                x_curr = x_cand
                f_curr = f_cand

            T *= cool_rate

        return self.best_f, self.best_x