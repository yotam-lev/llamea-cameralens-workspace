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
        x_eval = np.clip(x.copy(), -1.0, 1.0)
        x_eval[18:24] = np.clip(np.round(x_eval[18:24]), 0, 5).astype(int)
        f = func(x_eval)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x_eval.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        x_curr = np.random.uniform(-1, 1, self.dim)
        f_curr = self._evaluate(x_curr, func)
        
        T = 1.0
        min_T = 1e-9
        sigma_cont = 0.3
        cat_p = 0.15
        
        perf_window = 25
        f_hist = [f_curr]
        acc_hist = [1.0]
        imp_hist = [0.0]
        
        while True:
            if self.evals >= self.budget or T < min_T:
                break
                
            # Hessian preprocessing & regularization
            H_inv = np.eye(18)
            if hess_func is not None:
                if self.evals >= self.budget: break
                H = hess_func(x_curr)
                H = 0.5 * (H + H.T)  # Ensure symmetry
                w, v = np.linalg.eigh(H)
                w = np.maximum(w, 1e-4)  # PD enforcement
                H_reg = v @ np.diag(w) @ v.T
                H_inv = np.linalg.inv(H_reg)
                H_inv /= max(np.linalg.norm(H_inv), 1e-10)  # Normalize scale
                
            # Continuous perturbation scaled by inverse Hessian
            noise = np.random.randn(18)
            step = sigma_cont * T * (H_inv @ noise)
            x_trial = x_curr.copy()
            x_trial[:18] += step
            
            # Adaptive categorical perturbation probability
            imp_rate = np.mean(imp_hist[-perf_window:]) if len(imp_hist) >= 2 else 0.0
            cat_p = max(0.05, min(0.3, 0.15 - 0.05 * imp_rate + 0.05 * np.random.rand()))
            if np.random.rand() < cat_p:
                x_trial[18:24] = np.clip(x_trial[18:24] + np.random.choice([-1, 1]), 0, 5)
                
            # Clip & repair before evaluation
            if self.evals >= self.budget: break
            f_trial = self._evaluate(x_trial, func)
            df = f_trial - f_curr
            
            # Acceptance probability with explicit overflow guard
            if df < 0:
                acc_prob = 1.0
            else:
                acc_prob = np.exp(-np.clip(df / max(T, 1e-10), 0, 700))
                
            if np.random.rand() < acc_prob:
                x_curr = x_trial.copy()
                f_curr = f_trial
                acc_hist.append(1.0)
                f_hist.append(f_curr)
            else:
                acc_hist.append(0.0)
            imp_hist.append(df)
            
            # Feedback-driven adaptation
            if len(f_hist) >= perf_window:
                recent_acc = np.mean(acc_hist[-perf_window:])
                recent_imp = np.mean(imp_hist[-perf_window:])
                
                # Temperature schedule adapts to improvement/acceptance
                T *= (0.96 if recent_imp > 0 else (1.05 if recent_acc < 0.1 else 0.98))
                T = max(T, min_T)
                
                # Step size adapts to recent acceptance rate
                sigma_cont *= (1.05 if recent_acc > 0.4 else (0.95 if recent_acc < 0.1 else 1.0))
                sigma_cont = np.clip(sigma_cont, 0.05, 2.0)
                
            # Adaptive local search trigger based on stagnation & remaining budget
            progress = self.evals / self.budget
            ls_freq = int(np.max([15, 60 * (1 - progress)]))
            should_ls = (self.evals % ls_freq == 0) and (self.evals < self.budget * 0.85) and (recent_imp > -5e-5)
            
            if should_ls and hess_func is not None:
                cat_fixed = np.clip(np.round(x_curr[18:24]), 0, 5).astype(int)
                def obj_c(xc):
                    xc_full = np.concatenate([xc, cat_fixed])
                    return self._evaluate(xc_full, func)
                bounds = [(-1.0, 1.0)] * 18
                res = minimize(obj_c, x_curr[:18], method='trust-constr', bounds=bounds, hess=lambda x: H_inv)
                if res.success and res.fun < f_curr:
                    x_curr[:18] = res.x
                    x_curr[18:24] = cat_fixed
                    f_curr = self._evaluate(x_curr, func)
                    f_hist.append(f_curr)
                    imp_hist.append(f_curr - (f_hist[-2] if len(f_hist) > 1 else f_curr))
                    
        return self.best_f, self.best_x