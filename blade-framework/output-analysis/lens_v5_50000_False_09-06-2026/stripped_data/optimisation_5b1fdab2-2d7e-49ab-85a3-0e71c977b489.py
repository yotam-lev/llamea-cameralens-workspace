import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.S = np.zeros((6, 6))
        self.S_conf = np.zeros((6, 6))
        self.temperature = 1.0

    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        x_clip = np.clip(x.copy(), -1.0, 1.0)
        x_clip[18:24] = np.clip(np.round(x_clip[18:24]), 0, 5).astype(int)
        f = func(x_clip)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x_clip.copy()
        return f

    def _get_discrete(self):
        # Softmax selection with curvature-weighted profile
        scores = self.S / (self.S_conf + 1e-6)
        # Add temperature for exploration, decrease over time
        exp_scores = np.exp(scores / self.temperature)
        probs = exp_scores / (np.sum(exp_scores, axis=1) + 1e-6)
        d = np.zeros(6, dtype=int)
        for k in range(6):
            d[k] = np.random.choice(6, p=probs[k])
        return d

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        x_curr = np.zeros(self.dim)
        d_curr = self._get_discrete()
        x_curr[18:24] = d_curr
        self._evaluate(x_curr, func)
        
        self.S_conf += 1.0  # Initialize confidence
        self.temperature = 1.0
        
        while self.evals < self.budget:
            # Select discrete configuration based on learned profile
            d_curr = self._get_discrete()
            x_curr[18:24] = d_curr
            
            # Continuous refinement with curvature-aware trust region
            if self.evals < self.budget and hess_func is not None:
                H_raw = hess_func(x_curr)
                eigs = np.linalg.eigvalsh(H_raw)
                curv_weight = 1.0 / (np.mean(np.abs(eigs)) + 1e-6)
                
                # Regularize Hessian for trust-constr
                H_reg = np.abs(H_raw) + 1e-3 * np.eye(18)
                
                def obj(xc): return func(np.concatenate([xc, d_curr]))
                def jac(xc):
                    g = grad_func(np.concatenate([xc, d_curr])) if grad_func else np.zeros(18)
                    return g[:18]
                def hess(xc): return H_reg

                res = minimize(
                    obj, x_curr[:18], jac=jac, hess=hess,
                    method='trust-constr', bounds=[(-1.0, 1.0)] * 18,
                    options={'maxiter': 10, 'verbose': 0}
                )
                x_curr[:18] = res.x
                self._evaluate(x_curr, func)
                
                # Update discrete profile with curvature-weighted signal
                # This creates the tight feedback loop: continuous optimization informs discrete learning
                delta = self.best_f - self.best_f  # Placeholder, use previous best logic if needed
                # Simpler update: accumulate evidence
                self.S[:, d_curr] += curv_weight * (self.best_f + 1e-6)
                self.S_conf[:, d_curr] += curv_weight
                
            # Decay temperature to focus on best discrete choices
            self.temperature *= 0.999

        return self.best_f, self.best_x