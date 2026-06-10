import numpy as np
from scipy.optimize import minimize

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
        T_min = 1e-5
        cool_rate = 0.995
        sigma_c = 0.3
        sigma_cat = 0.5
        win_size = 30
        acc_history = []
        ls_counter = 0
        ls_interval = 10

        for _ in range(self.budget // 2):
            if self.evals >= self.budget: break

            # Self-tuning: Hessian-conditioned adaptation
            if hess_func is not None and _ % 4 == 0:
                fixed_cat = np.clip(np.round(x_curr[18:24]), 0, 5).astype(int)
                full_x = np.concatenate([x_curr[:18], fixed_cat])
                H_raw = hess_func(full_x)
                eigs, Q = np.linalg.eigh(H_raw)
                H_reg = np.abs(eigs) + 1e-6
                cond = np.max(H_reg) / np.min(H_reg)
                # Inverse curvature scaling for adaptive step sizes
                sigma_c = max(1e-3, 0.4 / (np.sqrt(cond) + 1.0))
                sigma_cat = max(1e-3, 0.6 / (np.sqrt(np.mean(H_reg)) + 1.0))

            step_c = np.random.randn(18) * sigma_c
            step_cat = np.random.randn(6) * sigma_cat

            # Anisotropic projection along regularized eigenmodes
            if hess_func is not None and _ % 4 == 0:
                step_c = Q @ (step_c / np.sqrt(np.abs(eigs) + 1e-6))

            x_cand = x_curr + np.concatenate([step_c, step_cat])
            f_cand = self._evaluate(x_cand, func)

            improved = f_cand < f_curr
            if improved or np.random.rand() < np.exp(-(f_cand - f_curr) / max(T, 1e-10)):
                x_curr = x_cand
                f_curr = f_cand
            acc_history.append(improved)
            if len(acc_history) > win_size: 
                acc_history.pop(0)
            acc_rate = np.mean(acc_history)

            # Adaptive cooling based on runtime feedback (improvement rate)
            if acc_rate > 0.6: T *= 0.98
            elif acc_rate < 0.15: T *= 1.05
            if T < T_min: T = 1.0

            ls_counter += 1
            # Dynamic local search frequency: triggers on stagnation or periodic reset
            if (ls_counter >= ls_interval) or (acc_rate < 0.1 and _ > win_size):
                ls_counter = 0
                ls_interval = int(np.random.uniform(8, 15))
                fixed_cat = np.clip(np.round(x_curr[18:24]), 0, 5).astype(int)

                def hess_wrapper(x_c):
                    return hess_func(np.concatenate([x_c, fixed_cat]))
                def fun_wrapper(x_c):
                    return func(np.concatenate([x_c, fixed_cat]))

                res = minimize(fun_wrapper, x_curr[:18], method='trust-constr',
                               bounds=[(-1.0, 1.0)]*18, hess=hess_wrapper,
                               options={'maxiter': 15})
                if self.evals < self.budget:
                    x_curr[:18] = res.x
                    f_curr = self._evaluate(x_curr, func)

            T *= cool_rate

        return self.best_f, self.best_x