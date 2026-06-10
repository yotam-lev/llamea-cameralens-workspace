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
        
        # Category-specific continuous models: dict mapping tuple(cat_ids) -> config
        self.cat_models = {}
        self.cat_probs = np.ones((6, 6)) / 6  # Probabilities for each of 6 dims
        
        # Global state
        self.n_samples = 8
        self.mut_scale = 0.05
        self.ema_alpha = 0.2
        
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

    def _get_or_create_model(self, c_ids):
        if c_ids not in self.cat_models:
            self.cat_models[c_ids] = {
                'mu': np.zeros(18),
                'H': np.eye(18),
                'best_f': float('inf'),
                'count': 0,
                'samples': []
            }
        return self.cat_models[c_ids]

    def _update_model(self, model, x_c, f, H):
        # Update mean with EMA
        model['mu'] = (1 - self.ema_alpha) * model['mu'] + self.ema_alpha * x_c
        
        # Accumulate curvature info (regularized Hessian)
        H_reg = H + 1e-2 * np.eye(18)
        # Simple averaging of inverse Hessian for covariance estimation
        try:
            H_inv = np.linalg.inv(H_reg)
            if np.allclose(H_inv, H_inv):  # Check finite
                if model['count'] == 0:
                    model['H_inv_avg'] = H_inv
                else:
                    model['H_inv_avg'] = (1 - self.ema_alpha) * model['H_inv_avg'] + self.ema_alpha * H_inv
        except np.linalg.LinAlgError:
            pass
        model['count'] += 1
        if f < model['best_f']:
            model['best_f'] = f
            
    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self._hess_func = hess_func
        if hess_func is None:
            self._hess_func = lambda x: np.eye(self.dim)  # Fallback

        while self.evals < self.budget:
            # 1. Sample Categorical Configurations
            cat_ids = tuple(np.random.choice(6, size=6, p=self.cat_probs[i]) for i in range(6))
            
            # 2. Retrieve/Initialize Continuous Model for this Category
            model = self._get_or_create_model(cat_ids)
            
            # 3. Sample Continuous Geometry conditioned on Category
            if model['count'] > 0 and 'H_inv_avg' in model:
                try:
                    cov = model['H_inv_avg'] * self.mut_scale**2
                    # Ensure symmetric positive definite for sampling
                    cov = (cov + cov.T) / 2
                    x_c = np.random.multivariate_normal(model['mu'], cov)
                except (np.linalg.LinAlgError, ValueError):
                    x_c = np.random.uniform(-1, 1, 18)
            else:
                x_c = np.random.uniform(-1, 1, 18)
                
            # 4. Evaluate Mixed Point
            x = np.empty(24)
            x[:18] = x_c
            x[18:24] = np.array(list(cat_ids))
            f = self._evaluate(x, func)
            
            # 5. Update Category-Modulated Model with Hessian
            if self._hess_func is not None and self.evals < self.budget:
                try:
                    H = self._hess_func(x)
                    self._update_model(model, x_c, f, H)
                except Exception:
                    pass
                    
            # 6. Update Categorical Probabilities (Survival of fittest on categories)
            # Boost probabilities for categories that yield better f
            for i, c_val in enumerate(list(cat_ids)):
                score = -f + self.best_f  # Positive if better than global best
                self.cat_probs[i, c_val] += 0.01 * np.exp(score)
                
            # Normalize probabilities
            for i in range(6):
                self.cat_probs[i] /= self.cat_probs[i].sum()
                
        return self.best_f, self.best_x