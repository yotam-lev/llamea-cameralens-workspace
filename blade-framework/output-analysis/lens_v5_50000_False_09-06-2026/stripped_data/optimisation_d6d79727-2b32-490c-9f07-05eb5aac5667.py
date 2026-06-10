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
        x = np.clip(x.copy(), -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _spectral_escape(self, x, func, grad_func, hess_func):
        xc = x[:18]
        cat = x[18:24]
        x_full = np.concatenate([xc, cat])

        if self.evals >= self.budget: return np.concatenate([xc, cat])
        H_raw = hess_func(x_full)
        
        eigs, Q = np.linalg.eigh(H_raw)
        eps = 1e-6
        eigs_pd = np.abs(eigs) + 1e-4
        cond = eigs_pd[-1] / (eigs_pd[0] + eps)

        # 1. Narrow Valley / Deep Minimum Escape: Anisotropic Levy-Walk
        if cond > 50.0 or np.min(eigs_pd) > 1.0:
            levy = np.random.standard_cauchy(18)
            # Scale noise by inverse square root of curvature to hop across ridges
            step = Q @ (levy / np.sqrt(eigs_pd + eps))
            alpha = np.clip(2.0 / (np.max(eigs_pd) + 1e-4), 0.01, 5.0)
            xc_new = xc + alpha * step

        # 2. Saddle Traversal: Steepest descent along negative eigenvector
        elif np.min(eigs) < -eps:
            v_neg = Q[:, np.argmin(eigs)]
            xc_new = xc - 0.5 * v_neg

        # 3. Promising Basin Exploitation: Hessian-Conditioned Trust Region
        else:
            H_pd = Q @ np.diag(eigs_pd) @ Q.T
            def obj(xc):
                if self.evals >= self.budget: return float('inf')
                return func(np.concatenate([xc, cat]))
            def jac(xc):
                if self.evals >= self.budget: return np.zeros(18)
                return grad_func(np.concatenate([xc, cat]))
            def hess(xc): return H_pd

            res = minimize(obj, xc, jac=jac, hess=hess, method='trust-constr',
                           bounds=[(-1.0, 1.0)] * 18, options={'maxiter': 3, 'verbose': 0})
            xc_new = res.x

        return np.concatenate([xc_new, cat])

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        pop = np.random.uniform(-1, 1, size=(16, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])

        stagnation = 0
        prev_best = self.best_f

        while self.evals < self.budget:
            if self.best_f == prev_best:
                stagnation += 1
            else:
                stagnation = 0
                prev_best = self.best_f

            # Global reset upon stagnation to prevent basin-lock
            if stagnation > 20:
                pop = np.random.uniform(-1, 1, size=(16, self.dim))
                pop_f = np.array([self._evaluate(x, func) for x in pop])
                stagnation = 0
                continue

            worst_idx = np.argmax(pop_f)
            x_worst = pop[worst_idx]

            if hess_func is not None and grad_func is not None:
                if self.evals < self.budget:
                    x_prop = self._spectral_escape(x_worst, func, grad_func, hess_func)
                    f_prop = self._evaluate(x_prop, func)
                    if f_prop < pop_f[worst_idx]:
                        pop[worst_idx] = x_prop
                        pop_f[worst_idx] = f_prop
                        worst_idx = np.argmin(pop_f)

            # Curvature-Guided Differential Evolution for diversity
            p_idx = np.argsort(pop_f)[:8]
            for _ in range(10):
                if self.evals >= self.budget: break
                i1, i2, i3 = np.random.choice(p_idx, 3, replace=False)
                mu = pop[i1] + 0.7 * (pop[i2] - pop[i3])
                if hess_func is not None:
                    if self.evals < self.budget:
                        H_raw = hess_func(self.best_x)
                        eigs, Q = np.linalg.eigh(H_raw)
                        eigs_pd = np.abs(eigs) + 1e-4
                        # Mutate along high-curvature eigendirections for targeted probing
                        mu[:18] += Q @ (np.random.normal(0, 0.2, 18) / np.sqrt(eigs_pd))
                mu[18:24] = np.clip(np.round(np.random.uniform(0.0, 5.99, 6)), 0, 5).astype(int)
                f_mut = self._evaluate(mu, func)
                if f_mut < pop_f[worst_idx]:
                    pop[worst_idx] = mu
                    pop_f[worst_idx] = f_mut
                    worst_idx = np.argmin(pop_f)

        return self.best_f, self.best_x