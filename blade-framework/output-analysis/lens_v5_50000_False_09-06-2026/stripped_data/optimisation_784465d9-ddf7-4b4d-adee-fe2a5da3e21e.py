import numpy as np
from numpy.linalg import eigh

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.categorical_ids = np.arange(6)
        self.H = None
        self.inv_sqrt_eigs = None
        self.evecs = None
        self.H_valid = False
        self.T0 = 0.4
        self.T = self.T0
        self.decay = 0.998

    def _evaluate(self, x):
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = self.func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.func = func
        self.hess_func = hess_func

        current_x = np.random.uniform(-1, 1, size=self.dim)
        current_f = self._evaluate(current_x)

        while self.evals < self.budget:
            # Hessian update strategy: trust geometry for proposal scaling
            if not self.H_valid or (self.evals > 0 and self.evals % 30 == 0):
                try:
                    H = self.hess_func(self.best_x)
                    eigs, evecs = eigh(H)
                    # PD enforcement: use absolute eigenvalues to prevent singular inversions
                    eigs = np.abs(eigs) + 1e-6
                    self.inv_sqrt_eigs = 1.0 / np.sqrt(eigs)
                    self.evecs = evecs
                    self.H_valid = True
                except Exception:
                    pass

            # Curvature-adaptive proposal generation
            if self.H_valid:
                # Isotropic noise projected onto inverse-curvature metric
                noise = np.random.randn(18)
                scaled_noise = self.evecs @ np.diag(self.inv_sqrt_eigs) @ noise
                cont_prop = current_x[:18] + self.T * scaled_noise
            else:
                cont_prop = current_x[:18] + np.random.randn(18) * self.T

            cat_prop = current_x[18:24].copy()
            # Categorical flips annealed by temperature
            if np.random.rand() < min(1.0, self.T / 0.5):
                dim_cat = np.random.randint(18, 24)
                cat_prop[dim_cat] = np.random.randint(0, 6)

            proposal_x = np.concatenate([cont_prop, cat_prop])
            proposal_f = self._evaluate(proposal_x)

            # Metropolis acceptance criterion
            delta = proposal_f - current_f
            if delta < 0 or np.random.rand() < np.exp(-delta / max(self.T, 1e-9)):
                current_x = proposal_x
                current_f = proposal_f

            # Cool down
            self.T *= self.decay
            if self.T < 1e-8:
                self.T = 1e-8

        return self.best_f, self.best_x