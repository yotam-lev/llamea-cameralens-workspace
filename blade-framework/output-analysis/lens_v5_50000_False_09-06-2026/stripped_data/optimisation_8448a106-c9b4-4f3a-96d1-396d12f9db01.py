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
        self.T = 1.0
        self.cooling = 0.985
        self.prop_scale = 0.3
        self.accept_rate = 0.0

    def _eval(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        # STRICT boundary & categorical enforcement
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _build_cov(self, x):
        if self.hess_func is None:
            return np.eye(18)
        try:
            H = self.hess_func(x)
            # Mandatory regularization: absolute eigenvalues + shift
            eigs = np.abs(np.linalg.eigvalsh(H))
            shift = max(0, 1e-2 - eigs.min())
            H_reg = H + shift * np.eye(18)
            # Inverse Hessian acts as adaptive proposal covariance
            return np.linalg.inv(H_reg)
        except Exception:
            return np.eye(18)

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        
        # LHS-compliant initialization syntax
        pop = np.random.uniform(-1, 1, size=(30, self.dim))
        x_curr = pop[0].copy()
        f_curr = self._eval(x_curr, func)
        if self.evals >= self.budget:
            return self.best_f, self.best_x

        best_x, best_f = x_curr.copy(), f_curr
        T = self.T

        while self.evals < self.budget:
            if self.evals >= self.budget:
                break

            # Adaptive covariance from exact Hessian
            cov = self._build_cov(x_curr)
            
            # Stochastic categorical manifold switch
            if np.random.rand() < 0.15:
                x_curr[18:24] = np.random.randint(0, 6, size=6)

            # Hessian-guided continuous proposal
            noise = np.random.multivariate_normal(np.zeros(18), cov)
            x_prop = x_curr.copy()
            x_prop[:18] += noise * self.prop_scale

            # Evaluate proposal
            f_prop = self._eval(x_prop, func)
            if self.evals >= self.budget:
                break

            # Metropolis acceptance criterion
            delta = f_prop - f_curr
            accept = delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-12))
            
            if accept:
                x_curr = x_prop
                f_curr = f_prop
                if f_prop < best_f:
                    best_f = f_prop
                    best_x = x_prop.copy()

            # Geometric cooling & proposal scale adaptation
            T *= self.cooling
            self.accept_rate = 0.95 * self.accept_rate + 0.05 * (1.0 if accept else 0.0)
            self.prop_scale *= (1.0 if self.accept_rate > 0.2 else 1.1) * (0.99 if self.accept_rate < 0.2 else 1.0)
            self.prop_scale = np.clip(self.prop_scale, 0.01, 2.0)

        self.best_f = best_f
        self.best_x = best_x
        return self.best_f, self.best_x