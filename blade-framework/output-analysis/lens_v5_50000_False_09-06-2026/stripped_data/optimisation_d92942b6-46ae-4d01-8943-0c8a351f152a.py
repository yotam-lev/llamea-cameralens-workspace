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
        self._func = None
        
        # Curvature & Escape State
        self.H_reg = np.eye(18)
        self.H_eigvecs = np.eye(18)
        self.H_eigvals = np.ones(18)
        self.stagnation = 0
        self.last_f = float('inf')
        self.jump_scale = 0.3

    def _evaluate(self, x):
        if self.evals >= self.budget: return float('inf')
        x = np.clip(x.copy(), -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = self._func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _update_curvature(self, x):
        if self.evals >= self.budget: return
        H = self._hess_func(x)
        eigs = np.linalg.eigvalsh(H)
        shift = max(0, 1e-6 - eigs.min())
        self.H_reg = H + shift * np.eye(18)
        self.H_eigvals = np.clip(np.abs(eigs), 1e-6, None)
        self.H_eigvecs = np.linalg.eigh(H)[1]

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self._hess_func = hess_func
        self._func = func
        
        n_samples = 10
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        f_pop = np.array([self._evaluate(x) for x in pop])
        base_idx = np.argmin(f_pop)
        base_x = pop[base_idx].copy()
        f_base = f_pop[base_idx]
        self.last_f = f_base

        while self.evals < self.budget:
            if f_base >= self.last_f:
                self.stagnation += 1
                self.jump_scale = np.clip(self.jump_scale * 1.15, 0.1, 2.0)
            else:
                self.stagnation = 0
                self.last_f = f_base
                self.jump_scale = np.clip(self.jump_scale * 0.9, 0.1, 0.5)

            if self.evals >= self.budget: break

            # Novel Exploration: Hessian-Metric Levy Escape vs Trust-Region Refinement
            if self.stagnation > 2 or np.random.rand() < 0.4:
                # Escape phase: jump along low-curvature eigenvector scaled by inverse sqrt curvature
                dim_idx = np.random.randint(0, 18)
                esc_dir = self.H_eigvecs[:, dim_idx]
                step_size = self.jump_scale / np.sqrt(self.H_eigvals[dim_idx])
                noise = np.random.randn()
                perturbation = esc_dir * step_size * noise * 2.0
                cand = base_x[:18] + perturbation
            else:
                # Exploitation phase: local trust-region refinement
                cat = np.clip(np.round(base_x[18:24]), 0, 5).astype(int)
                c = base_x[:18].copy()
                if self.evals >= self.budget: break
                if self._hess_func is not None and self.H_reg is not None:
                    res = minimize(
                        lambda xc: func(np.concatenate([xc, cat])),
                        c, method='trust-constr',
                        hess=lambda xc: self.H_reg,
                        bounds=[(-1.0, 1.0)]*18, options={'maxiter': 25, 'verbose': 0}
                    )
                    if res.success:
                        cand = np.concatenate([res.x, cat])
                        f_c = self._evaluate(cand)
                        if f_c < f_base:
                            base_x = cand
                            f_base = f_c
                            continue

            f_cand = self._evaluate(np.concatenate([cand, base_x[18:24]]))
            if f_cand < f_base and self.evals < self.budget:
                base_x = np.concatenate([cand, base_x[18:24]])
                f_base = f_cand

            # Update Hessian periodically
            if self.evals % 7 == 0 and self.evals < self.budget:
                self._update_curvature(base_x)

        return self.best_f, self.best_x