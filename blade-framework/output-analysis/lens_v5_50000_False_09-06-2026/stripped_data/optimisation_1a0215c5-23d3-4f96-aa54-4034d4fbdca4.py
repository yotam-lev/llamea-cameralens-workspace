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

    def _regularize_inverse(self, H_raw):
        eigs = np.linalg.eigvalsh(H_raw)
        eigs = np.abs(eigs) + 1e-6
        return np.diag(1.0 / eigs)

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # Initialize mixed-variable point
        x_curr = np.zeros(self.dim)
        x_curr[:18] = np.random.uniform(-1, 1, 18)
        x_curr[18:24] = np.random.randint(0, 6, 6)
        self._evaluate(x_curr, func)

        iter_count = 0
        while self.evals < self.budget:
            iter_count += 1
            cat_fixed = x_curr[18:24].copy()

            # 1. Continuous Refinement: Trust-Region conditioned on current discrete configuration
            if self.evals < self.budget and hess_func is not None:
                H_raw = hess_func(x_curr)
                H_reg_inv = self._regularize_inverse(H_raw)

                def obj(xc): return func(np.concatenate([xc, cat_fixed]))
                def jac(xc):
                    g = grad_func(np.concatenate([xc, cat_fixed])) if grad_func else np.zeros(18)
                    return g[:18]
                def hess(xc): return H_reg_inv

                res = minimize(
                    obj, x_curr[:18], jac=jac, hess=hess,
                    method='trust-constr', bounds=[(-1.0, 1.0)] * 18,
                    options={'maxiter': 4, 'verbose': 0}
                )
                x_curr[:18] = res.x

            # 2. Dynamic Discrete-Continuous Coupling
            if self.evals < self.budget and hess_func is not None:
                full_x = x_curr.copy()
                H_raw = hess_func(full_x)
                eigs = np.linalg.eigvalsh(H_raw)
                stiffness = 1.0 / (np.abs(eigs) + 1e-6)
                
                # Continuous gradient informs which discrete variables exert the strongest pull on geometry
                g = grad_func(full_x) if grad_func else np.zeros(18)
                prob_cat_change = np.zeros(6)
                
                for k in range(6):
                    # Map categorical index to influential continuous dimensions
                    cont_dims = [k, (k + 6) % 18, (k + 12) % 18]
                    grad_pressure = np.sum(np.abs(g[cont_dims]))
                    curvature_compliance = np.mean(stiffness[cont_dims])
                    # High gradient pressure + high compliance (low stiffness) = high mutation priority
                    prob_cat_change[k] = grad_pressure * curvature_compliance
                
                prob_cat_change /= (np.sum(prob_cat_change) + 1e-6)

                # Execute discrete mutation biased by continuous landscape sensitivity
                if np.random.random() < np.sum(prob_cat_change):
                    k_flip = np.random.choice(6, p=prob_cat_change)
                    valid_vals = [v for v in range(6) if v != x_curr[18 + k_flip]]
                    x_curr[18 + k_flip] = np.random.choice(valid_vals)
                    self._evaluate(x_curr, func)

            # 3. Periodic Decoupling to escape discrete-continuous lock-in
            if iter_count % 5 == 0 and self.evals < self.budget:
                x_curr[:] = np.random.uniform(-1, 1, self.dim)
                x_curr[18:24] = np.random.randint(0, 6, 6)
                self._evaluate(x_curr, func)

        return self.best_f, self.best_x