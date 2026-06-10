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

    def _get_hessian_info(self, x, hess_func):
        # Safety check before Hessian call
        if self.evals >= self.budget:
            return None, None, None, None
        H_raw = hess_func(x)[:18, :18]
        eigs_vals, Q = np.linalg.eigh(H_raw)
        cond_num = np.max(np.abs(eigs_vals)) / (np.min(np.abs(eigs_vals)) + 1e-8)
        # Regularization for positive-definiteness
        lam = max(0.0, -np.min(eigs_vals) + 1e-4)
        H_reg = H_raw + lam * np.eye(18)
        return H_reg, eigs_vals, Q, cond_num

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_samples = 40
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])

        stagnation_thresh = 15
        high_cond_thresh = 500
        grad_thresh = 1e-3
        
        stagnation_counters = np.zeros(n_samples)

        while self.evals < self.budget:
            # Hessian and Gradient analysis for best individual
            best_idx = np.argmin(pop_f)
            x_best = pop[best_idx].copy()
            x_best[18:24] = np.clip(np.round(x_best[18:24]), 0, 5).astype(int)
            
            H_reg, eigs_vals, Q, cond_num = None, None, None, None
            grad_norm = 1e6
            flat_dir = None

            if hess_func is not None and self.evals < self.budget:
                H_reg, eigs_vals, Q, cond_num = self._get_hessian_info(x_best, hess_func)
                if H_reg is not None:
                    flat_idx = np.argmin(np.abs(eigs_vals))
                    flat_dir = Q[:, flat_idx]
            
            if grad_func is not None and self.evals < self.budget:
                g = grad_func(x_best)
                grad_norm = np.linalg.norm(g)

            # 1. Curvature-Spectral Categorical Resonance (Escape Mechanism)
            # Detects "Categorical Mismatch": High curvature (narrow well) + Low gradient (bottom of well)
            # implies the continuous optimum is trapped in a deep local minimum defined by the current cat.
            need_cat_escape = False
            if cond_num is not None and cond_num > high_cond_thresh and grad_norm < grad_thresh:
                stagnation_counters[best_idx] += 1
                if stagnation_counters[best_idx] > stagnation_thresh:
                    need_cat_escape = True
                    # Resonance amplitude driven by landscape roughness (Trace)
                    # High trace = rough landscape, categorical variables have high impact
                    roughness = np.clip(np.trace(H_reg) / 100.0, 0.1, 1.0)
                    aggress = 0.3 * roughness
                    if np.random.rand() < aggress:
                        # Mutate categories to jump to new basins
                        mut_mask = np.random.rand(6) < 0.5
                        mut_vals = np.random.choice([-1, 1], size=6)
                        base_cat = np.clip(np.round(pop[best_idx][18:24]), 0, 5).astype(int)
                        new_cat = np.clip(base_cat + mut_mask * mut_vals, 0, 5).astype(int)
                        pop[best_idx][18:24] = new_cat
                        pop_f[best_idx] = self._evaluate(pop[best_idx], func)
                        stagnation_counters[best_idx] = 0
            else:
                stagnation_counters[best_idx] = 0

            # 2. Hessian-Projected Levy Propagation (Exploration)
            for i in range(n_samples):
                if self.evals >= self.budget: break
                
                # Levy flight with curvature adaptation
                u = np.random.randn(18)
                v = np.random.randn(18)
                levy = u / (np.abs(v)**(1/1.5))
                
                # Curvature-adaptive scaling
                if H_reg is not None:
                    # Project noise into eigenbasis and scale inversely to curvature
                    # Small eigenvalues (shallow) get large steps; large eigenvalues (steep) get small steps
                    scaled_levy = np.zeros(18)
                    for k in range(18):
                        if np.abs(eigs_vals[k]) > 1e-6:
                            scaled_levy[k] = levy[k] / np.abs(eigs_vals[k])
                        else:
                            scaled_levy[k] = levy[k] * 1e6
                    step = Q @ scaled_levy
                    
                    # Additional barrier injection along shallowest mode if condition is high
                    if cond_num > 100:
                        barrier_amp = np.random.exponential(0.2) * np.sqrt(cond_num / 100.0)
                        step += flat_dir * barrier_amp
                else:
                    step = levy * 0.1

                # Perturb continuous variables
                xc = pop[i][:18].copy()
                xc_new = np.clip(xc + step * 0.05, -1.0, 1.0)
                
                # Categorical diffusion (standard for non-escaped individuals)
                base_cat = np.clip(np.round(pop[i][18:24]), 0, 5).astype(int)
                if np.random.rand() < 0.1:
                    mut_cat = base_cat + np.random.choice([-1, 0, 1], size=6)
                    cat_new = np.clip(mut_cat, 0, 5).astype(int)
                else:
                    cat_new = base_cat

                new_ind = np.concatenate([xc_new, cat_new])
                f_new = self._evaluate(new_ind, func)
                
                if f_new < pop_f[i]:
                    pop[i] = new_ind
                    pop_f[i] = f_new

            # 3. Hessian-Regularized Trust-Region Exploitation
            best_idx = np.argmin(pop_f)
            xc = pop[best_idx][:18]
            cat = np.clip(np.round(pop[best_idx][18:24]), 0, 5).astype(int)
            
            if self.evals < self.budget:
                def obj(xc_sub):
                    if self.evals >= self.budget: return float('inf')
                    return func(np.concatenate([xc_sub, cat]))
                def jac(xc_sub):
                    if self.evals >= self.budget: return np.zeros(18)
                    if grad_func: return grad_func(np.concatenate([xc_sub, cat]))[:18]
                    return np.zeros(18)
                def hess_closure(xc_sub):
                    if self.evals >= self.budget: return np.eye(18)
                    # Return regularized Hessian
                    if H_reg is not None:
                        lam = max(0.0, -np.min(eigs_vals) + 1e-4)
                        return H_reg + lam * np.eye(18)
                    return np.eye(18)

                res = minimize(obj, xc, jac=jac, hess=hess_closure, method='trust-constr',
                               bounds=[(-1.0, 1.0)]*18, options={'maxiter': 25, 'verbose': 0})
                
                if self.evals < self.budget:
                    x_ref = np.concatenate([res.x, cat])
                    f_ref = self._evaluate(x_ref, func)
                    if f_ref < pop_f[best_idx]:
                        pop[best_idx] = x_ref
                        pop_f[best_idx] = f_ref
                    else:
                        # Stagnation on local search increases escape probability
                        stagnation_counters[best_idx] += 10

        return self.best_f, self.best_x