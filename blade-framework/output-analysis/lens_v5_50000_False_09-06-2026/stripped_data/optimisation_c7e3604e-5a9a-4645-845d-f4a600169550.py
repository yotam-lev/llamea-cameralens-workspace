import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.acc_history = []
        self.triggered_ls = False

    def _clip_and_eval(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        x_clip = np.clip(x, -1.0, 1.0)
        x_clip[18:24] = np.clip(np.round(x_clip[18:24]), 0, 5).astype(int)
        f = func(x_clip)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x_clip.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        if self.evals < self.budget:
            x = np.random.uniform(-1, 1, self.dim)
            f = self._clip_and_eval(x, func)
        else:
            return self.best_f, self.best_x

        x_current = x
        f_current = f
        self.acc_history = []
        
        T = 1.0
        min_T = 1e-6
        alpha = 0.5
        base_decay = 0.98
        
        while T > min_T and self.evals < self.budget:
            # Adaptive temperature and step size based on recent acceptance rate
            if len(self.acc_history) >= 50:
                recent_acc = np.mean(self.acc_history[-50:])
                if recent_acc < 0.3:
                    T *= 0.95
                    alpha *= 0.9
                elif recent_acc > 0.6:
                    T *= 1.02
                    alpha *= 1.1
                self.acc_history = []
                
            T *= base_decay
            
            # Hessian preprocessing
            if hess_func is not None and self.evals < self.budget:
                H = hess_func(x_current)
                eigvals = np.linalg.eigvalsh(H)
                cond = np.max(np.abs(eigvals)) / np.min(np.abs(eigvals) + 1e-12)
                reg = np.clip(1e-5 + 0.01 * cond, 1e-4, 1.0)
                H_reg = H + reg * np.eye(18)
                H_inv = np.linalg.inv(H_reg)
            else:
                H_inv = np.eye(18)
                
            # Preconditioned proposal
            noise = np.random.randn(18)
            x_trial = x_current.copy()
            x_trial[:18] += alpha * (H_inv @ noise)
            
            # Categorical mutation
            if np.random.rand() < 0.05:
                idx = np.random.randint(18, 24)
                x_trial[idx] += np.random.choice([-1, 1])
                
            f_trial = self._clip_and_eval(x_trial, func)
            df = f_trial - f_current
            
            # Acceptance with overflow protection
            if df < 0:
                accept = True
            else:
                ratio = df / max(T, 1e-12)
                prob = np.exp(-np.clip(ratio, -700, 700))
                accept = np.random.rand() < prob
                
            if accept:
                x_current = x_trial
                f_current = f_trial
                self.acc_history.append(1.0)
            else:
                self.acc_history.append(0.0)
                
            # Memetic trigger: Convergence detection -> Trust-region refinement
            if not self.triggered_ls and len(self.acc_history) > 100 and np.mean(self.acc_history[-100:]) > 0.7:
                self.triggered_ls = True
                x_cats = np.clip(np.round(x_current[18:24]), 0, 5).astype(int)
                bounds = [(-1.0, 1.0)] * 18
                
                def obj(x_c):
                    x_full = np.concatenate([x_c, x_cats])
                    return self._clip_and_eval(x_full, func)
                    
                res = minimize(obj, x_current[:18], bounds=bounds, 
                               hess=hess_func, method='trust-constr')
                x_current[:18] = res.x
                f_current = self._clip_and_eval(x_current, func)
                self.triggered_ls = False
                self.acc_history = []
                
        return self.best_f, self.best_x