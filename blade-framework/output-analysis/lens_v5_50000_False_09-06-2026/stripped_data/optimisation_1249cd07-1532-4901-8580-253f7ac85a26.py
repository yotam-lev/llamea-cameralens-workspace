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

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_samples = 30
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])

        patience = 0
        stagnation_thresh = 6
        jump_base = 0.35

        while self.evals < self.budget:
            best_idx = np.argmin(pop_f)
            x_best = pop[best_idx]
            xc, cat = x_best[:18], x_best[18:24]

            # Stagnation detection
            grad_norm = 0.0
            if grad_func is not None:
                if self.evals >= self.budget: break
                g = grad_func(np.concatenate([xc, cat]))
                grad_norm = np.linalg.norm(g)

            patience = patience + 1 if grad_norm < 1e-5 else 0

            # Phase 1: Hessian-Regularized Trust-Region Exploitation
            if hess_func is not None and patience < stagnation_thresh:
                if self.evals >= self.budget: break
                if self.evals % 4 == 0:
                    full_x = np.concatenate([xc, cat])
                    H_raw = hess_func(full_x)[:18, :18]
                    eigs, Q = np.linalg.eigh(H_raw)
                    H_reg = Q @ np.diag(np.abs(eigs) + 1e-6) @ Q.T

                    def obj(xc_sub):
                        if self.evals >= self.budget: return float('inf')
                        return func(np.concatenate([xc_sub, cat]))
                    def jac(xc_sub):
                        if self.evals >= self.budget: return np.zeros(18)
                        if grad_func is not None:
                            return grad_func(np.concatenate([xc_sub, cat]))[:18]
                        return np.zeros(18)
                    def hess_local(xc_sub):
                        if self.evals >= self.budget: return H_reg
                        return H_reg

                    res = minimize(obj, xc, jac=jac, hess=hess_local, method='trust-constr',
                                   bounds=[(-1.0, 1.0)] * 18, options={'maxiter': 20, 'verbose': 0})
                    if self.evals < self.budget:
                        x_new = np.concatenate([res.x, cat])
                        f_new = self._evaluate(x_new, func)
                        pop[best_idx] = x_new
                        pop_f[best_idx] = f_new

            # Phase 2: Spectral-Filtered Basin Jumping (Escape Strategy)
            if patience >= stagnation_thresh:
                patience = 0
                if hess_func is not None:
                    if self.evals >= self.budget: break
                    full_x = np.concatenate([xc, cat])
                    H_raw = hess_func(full_x)[:18, :18]
                    eigs, Q = np.linalg.eigh(H_raw)

                    # Amplify steps along flat eigenmodes, dampen along steep ones
                    scale = 1.0 / (np.abs(eigs) + 1e-8)
                    spectral_noise = Q @ np.diag(np.sqrt(scale)) @ np.random.randn(n_samples, 18)

                    curv_spread = np.max(np.abs(eigs)) / (np.min(np.abs(eigs)) + 1e-8)
                    step_scale = jump_base * (1.0 + 0.5 * np.log10(curv_spread + 2.0))

                    for i in range(n_samples):
                        if self.evals >= self.budget: break
                        x_jump = x_best + step_scale * spectral_noise[i]
                        x_jump[18:24] = np.clip(np.round(cat + np.random.randint(-1, 2, 6)), 0, 5).astype(int)
                        f_jump = self._evaluate(x_jump, func)
                        if f_jump < pop_f[i] or np.random.rand() < 0.15:
                            pop[i] = x_jump
                            pop_f[i] = f_jump
                else:
                    pop[best_idx] = np.random.uniform(-1, 1, self.dim)
                    pop_f[best_idx] = self._evaluate(pop[best_idx], func)

            # Diversity Maintenance
            if self.evals < self.budget and np.random.rand() < 0.25:
                idx_m = np.random.randint(0, n_samples)
                x_m = pop[idx_m] + np.random.normal(0, 0.12, self.dim)
                x_m[18:24] = np.clip(np.round(x_m[18:24]), 0, 5).astype(int)
                f_m = self._evaluate(x_m, func)
                if f_m < pop_f[idx_m]:
                    pop[idx_m] = x_m
                    pop_f[idx_m] = f_m
                else:
                    idx_w = np.argmax(pop_f)
                    pop[idx_m] = pop[idx_w]
                    pop_f[idx_m] = pop_f[idx_w]

        return self.best_f, self.best_x