import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.hess_func = None
        self.H = None
        self.eigs = None
        self.vecs = None
        self.H_valid = False
        self.n_samples = 24
        self.stagnation_thresh = 12

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

    def _update_hessian(self, x):
        try:
            H_raw = self.hess_func(x)
            eigs, vecs = np.linalg.eigh(H_raw)
            # Regularization: force positive-definite via absolute eigenvalues
            eigs_reg = np.abs(eigs) + 1e-4
            self.H = vecs @ np.diag(eigs_reg) @ vecs.T
            self.eigs = eigs_reg
            self.vecs = vecs
            self.H_valid = True
        except Exception:
            self.H_valid = False

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        pop = np.random.uniform(-1, 1, size=(self.n_samples, self.dim))
        pbest_x = pop.copy()
        pbest_f = np.full(self.n_samples, np.inf)
        gbest_idx = 0
        gbest_f = float('inf')

        for i in range(self.n_samples):
            if self.evals >= self.budget: break
            f = self._evaluate(pop[i], func)
            pbest_f[i] = f
            if f < gbest_f:
                gbest_f = f
                gbest_idx = i

        stagnation = 0
        it = 0
        hess_cnt = 0

        while self.evals < self.budget:
            it += 1
            hess_cnt += 1

            if self.hess_func and (hess_cnt % 20 == 0 or not self.H_valid):
                if self.evals >= self.budget: break
                self._update_hessian(pop[gbest_idx])

            if gbest_f == pbest_f[gbest_idx]:
                stagnation += 1
            else:
                stagnation = 0

            for i in range(self.n_samples):
                if self.evals >= self.budget: break

                step = np.zeros(self.dim)
                if self.H_valid and stagnation > self.stagnation_thresh:
                    # Curvature-scaled escape along flat manifolds
                    z = np.random.randn(18)
                    inv_sqrt_eigs = 1.0 / np.sqrt(np.maximum(self.eigs[:18], 1e-6))
                    L = self.vecs[:18, :18] @ np.diag(inv_sqrt_eigs)
                    step[:18] = 0.3 * L @ z
                    
                    # Ascent along negative curvature directions to break traps
                    neg_idx = np.argmin(self.eigs[:18])
                    if self.eigs[neg_idx] < 1e-5:
                        step[:18] += 0.15 * self.vecs[:18, neg_idx]
                elif self.H_valid:
                    # Preconditioned diffusion for stable basins
                    z = np.random.randn(18)
                    inv_sqrt_eigs = 1.0 / np.sqrt(np.maximum(self.eigs[:18], 1e-6))
                    L = self.vecs[:18, :18] @ np.diag(inv_sqrt_eigs)
                    step[:18] = 0.08 * L @ z
                else:
                    step[:18] = np.random.randn(18) * 0.5

                # Discrete categorical jumps
                step[18:24] = np.random.choice([-1, 0, 1], size=6)

                cand = pop[i] + step
                f = self._evaluate(cand, func)

                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_x[i] = cand.copy()
                    pop[i] = cand.copy()

                if f < gbest_f:
                    gbest_f = f
                    gbest_idx = i
                    stagnation = 0

            # Local trust-region refinement on the current best basin
            if gbest_f < self.best_f * 0.3 and self.H_valid:
                xb = pop[gbest_idx][:18].copy()
                cats = np.clip(np.round(pop[gbest_idx][18:24]), 0, 5).astype(int)
                if self.evals < self.budget:
                    try:
                        res = minimize(
                            lambda xc: func(np.concatenate([xc, cats])),
                            xb,
                            method='trust-constr',
                            hess=lambda xc: self.H[:18, :18],
                            bounds=[(-1.0, 1.0)] * 18,
                            options={'maxiter': 20, 'verbose': 0}
                        )
                        if res.success and self.evals < self.budget:
                            cand = np.concatenate([res.x, cats])
                            f_c = self._evaluate(cand, func)
                            if f_c < gbest_f:
                                gbest_f = f_c
                                pop[gbest_idx] = cand
                                pbest_x[gbest_idx] = cand.copy()
                                pbest_f[gbest_idx] = f_c
                    except Exception:
                        pass

        return self.best_f, self.best_x