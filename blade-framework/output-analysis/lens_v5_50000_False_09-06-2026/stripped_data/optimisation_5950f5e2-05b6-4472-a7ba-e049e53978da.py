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
        if self.evals >= self.budget: return float('inf')
        x = np.clip(x.copy(), -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        pop = np.random.uniform(-1, 1, size=(25, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])

        cond_escape = 1e4
        stagnation = 0

        while self.evals < self.budget:
            if self.evals >= self.budget: break

            # Spectral Manifold Diffusion with Curvature Invariance
            for i in range(25):
                if self.evals >= self.budget: break
                x_c = pop[i, :18].copy()
                x_cat = pop[i, 18:24].copy()

                if self.evals >= self.budget: break
                H_raw = hess_func(np.concatenate([x_c, x_cat]))
                if H_raw is None or np.any(np.isnan(H_raw)): continue

                eigs, Q = np.linalg.eigh(H_raw)
                eigs_reg = np.abs(eigs) + 1e-6
                cond = eigs_reg[-1] / (eigs_reg[0] + 1e-8)

                # Novel: Curvature-weighted tangent diffusion for anisotropic exploration
                tangent_scale = 1.0 / np.sqrt(eigs_reg)
                diffusion = Q @ (tangent_scale * np.random.randn(18))

                # Spectral Reflection Mechanism: Leaps across narrow curvature valleys
                if cond > cond_escape:
                    w_min = Q[:, 0]
                    proj = np.dot(w_min, x_c - self.best_x[:18])
                    x_c = x_c - 2 * proj * w_min

                # Adaptive step sizing based on local conditioning
                alpha = 0.4 / (np.sqrt(cond) + 1.0)
                x_c_new = x_c + alpha * diffusion

                # Hessian-scaled trust-region exploitation for well-conditioned basins
                if cond < cond_escape * 0.05:
                    H_pd = Q @ np.diag(np.maximum(eigs, 1e-4)) @ Q.T
                    def obj(z): return func(np.concatenate([z, x_cat]))
                    def jac(z): return grad_func(np.concatenate([z, x_cat]))
                    def hes(z): return H_pd
                    
                    if self.evals >= self.budget: break
                    res = minimize(obj, x_c_new, jac=jac, hess=hes,
                                   method='trust-constr', bounds=[(-1, 1)]*18,
                                   options={'maxiter': 10, 'verbose': 0})
                    if self.evals >= self.budget: break
                    x_c_new = res.x

                pop[i, :18] = np.clip(x_c_new, -1.0, 1.0)
                f_new = self._evaluate(pop[i], func)
                if f_new < pop_f[i]: pop_f[i] = f_new

            # Diversity Injection via Curvature-Aware DE
            if self.evals < self.budget:
                idx = np.argsort(pop_f)[:12]
                mu = pop[idx[0], :18] + 0.7 * (pop[idx[1], :18] - pop[idx[2], :18])
                mu += np.random.randn(18) * 0.02
                x_diff = np.concatenate([np.clip(mu, -1, 1), np.clip(np.round(np.random.uniform(0, 5.99, 6)), 0, 5).astype(int)])
                f_diff = self._evaluate(x_diff, func)
                if f_diff < pop_f[-1]:
                    pop[-1] = x_diff
                    pop_f[-1] = f_diff

            # Adaptive Reset
            current_best = np.min(pop_f)
            if current_best == np.min(pop_f): stagnation += 1
            else: stagnation = 0

            if stagnation > 20:
                pop = np.random.uniform(-1, 1, size=(25, self.dim))
                pop_f = np.array([self._evaluate(x, func) for x in pop])
                stagnation = 0

        return self.best_f, self.best_x