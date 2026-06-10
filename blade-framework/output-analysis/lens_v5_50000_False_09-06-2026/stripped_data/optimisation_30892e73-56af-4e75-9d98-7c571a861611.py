import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.prop_L = np.eye(18)
        self.H_reg = np.eye(18)

    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        x = np.clip(x.copy(), -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_samples = 40
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])

        temp = 0.5
        beta = 0.1
        stagnation_steps = 0
        last_best_idx = np.argmin(pop_f)

        while self.evals < self.budget:
            curr_best_idx = np.argmin(pop_f)
            if self.best_f <= pop_f[curr_best_idx]:
                stagnation_steps += 1
            else:
                stagnation_steps = 0
                last_best_idx = curr_best_idx

            # Adaptive temperature & exploration scale
            temp = np.clip(temp * (1.0 + 0.15 * np.random.exponential()), 0.05, 3.0) if stagnation_steps > 2 else temp * 0.95
            beta = np.clip(beta + 0.005 * (stagnation_steps > 0) - 0.002 * (stagnation_steps == 0), 0.05, 1.5)

            # Hessian Analysis & PD Regularization
            if hess_func is not None and self.evals % 8 == 0:
                x_probe = pop[last_best_idx].copy()
                if self.evals >= self.budget: break
                H_raw = hess_func(x_probe)[:18, :18]
                eigs, Q = np.linalg.eigh(H_raw)
                # Enforce positive definiteness via absolute eigenvalues
                eigs_pd = np.abs(eigs) + 1e-6
                self.H_reg = Q @ np.diag(eigs_pd) @ Q.T
                # Curvature-aware proposal covariance (whitened + beta expansion)
                prop_cov = Q @ np.diag(1.0 / eigs_pd + beta) @ Q.T
                self.prop_L = np.linalg.cholesky(prop_cov)

            # Curvature-Resonant Spectral Exploration
            if self.evals < self.budget:
                z = np.random.randn(n_samples, 18)
                proposals_c = pop[last_best_idx][:18] + z @ self.prop_L.T
                cat_shifts = np.random.choice([-1, 0, 1], size=(n_samples, 6))
                proposals_cat = np.clip(np.round(pop[last_best_idx][18:24][None, :] + cat_shifts), 0, 5).astype(int)

                new_x = np.hstack([proposals_c, proposals_cat])
                new_f = np.array([self._evaluate(new_x[i], func) for i in range(n_samples)])

                # Adaptive Metropolis Acceptance
                delta = new_f - pop_f
                accept = delta < 0
                if np.any(~accept):
                    u = np.random.rand(~accept.sum())
                    accept[~accept] = u < np.exp(-delta[~accept] / temp)

                pop[accept] = new_x[accept]
                pop_f[accept] = new_f[accept]

            # Local Exploitation (Trust-Region)
            if stagnation_steps == 0 or self.evals % 12 == 0:
                best_idx = np.argmin(pop_f)
                xc = pop[best_idx][:18]
                cat = np.clip(np.round(pop[best_idx][18:24]), 0, 5).astype(int)

                def obj(xs): return func(np.concatenate([xs, cat]))
                def jac(xs): return grad_func(np.concatenate([xs, cat]))[:18] if grad_func else np.zeros(18)
                def hess(xs): return self.H_reg

                res = minimize(obj, xc, jac=jac, hess=hess, method='trust-constr',
                               bounds=[(-1.0, 1.0)]*18, options={'maxiter': 25, 'verbose': 0})
                if self.evals < self.budget:
                    x_ref = np.concatenate([res.x, cat])
                    f_ref = self._evaluate(x_ref, func)
                    if f_ref < pop_f[best_idx]:
                        pop[best_idx] = x_ref
                        pop_f[best_idx] = f_ref

        return self.best_f, self.best_x