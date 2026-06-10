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
        x_eval = np.clip(x.copy(), -1.0, 1.0)
        x_eval[18:24] = np.clip(np.round(x_eval[18:24]), 0, 5).astype(int)
        f = func(x_eval)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x_eval.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        x = np.random.uniform(-1, 1, self.dim)
        self._evaluate(x, func)
        x_c = x[:18].copy()
        x_d = x[18:24].copy()
        T = 1.0
        min_T = 1e-9
        local_iter = 3

        while T > min_T and self.evals < self.budget:
            # 1. Hessian Acquisition & Regularization
            if hess_func is not None and self.evals < self.budget:
                H = hess_func(np.concatenate([x_c, x_d]))
                eigs, vecs = np.linalg.eigh(H)
                eigs = np.abs(eigs) + 1e-6
                H_reg = vecs @ np.diag(eigs) @ vecs.T
            else:
                H_reg = np.eye(18)

            # Curvature metrics for dynamic mixed-variable coupling
            cond = H_reg.diagonal().max() / (H_reg.diagonal().min() + 1e-12)

            # Gradient preparation
            if grad_func is not None and self.evals < self.budget:
                jac_c = grad_func(np.concatenate([x_c, x_d]))[:18]
            else:
                jac_c = None

            # 2. Continuous Subspace Refinement (Trust-Region)
            bounds = [(-1.0, 1.0)] * 18
            max_it = min(local_iter, max(1, (self.budget - self.evals) // 5))

            def local_obj(x_sub):
                return self._evaluate(np.concatenate([x_sub, x_d]), func)

            def local_hess(x_sub):
                return H_reg

            res = minimize(
                fun=local_obj,
                x0=x_c,
                method='trust-constr',
                bounds=bounds,
                jac=jac_c if jac_c is not None else False,
                hess=local_hess,
                options={'maxiter': max_it, 'verbose': 0}
            )
            x_c = res.x
            local_iter = max(1, local_iter - 1)

            # 3. Dynamic Categorical Exploration (Curvature-Gated)
            # High condition number -> rugged landscape -> increase discrete switching
            cat_prob = np.clip(0.05 * np.log(cond + 1.0), 0.02, 0.6)

            x_d_trial = x_d.copy()
            if np.random.rand() < cat_prob:
                idx = np.random.randint(0, 6)
                delta = np.random.choice([-1, 1])
                x_d_trial[idx] = np.clip(x_d[idx] + delta, 0, 5)

            # 4. Curvature-Scaled Acceptance (Overflow-Safe)
            x_trial = np.concatenate([x_c, x_d_trial])
            f_trial = self._evaluate(x_trial, func)
            f_curr = self._evaluate(np.concatenate([x_c, x_d]), func)

            df = f_trial - f_curr
            if df < 0:
                accept = True
            else:
                # Scale temperature by inverse curvature to maintain proper SA dynamics
                T_adj = T * (1.0 / np.log(cond + 1.0))
                log_ratio = -df / max(T_adj, 1e-10)
                log_ratio = np.clip(log_ratio, -50.0, 0.0)
                accept = np.random.rand() < np.exp(log_ratio)

            if accept:
                x_c = x_trial[:18]
                x_d = x_trial[18:]
                T *= 0.95
            else:
                # Reject: damp geometry to prevent divergence in tight curvature
                x_c = x_c - 0.5 * res.x
                x_c = np.clip(x_c, -1.0, 1.0)
                T *= 0.95

        return self.best_f, self.best_x