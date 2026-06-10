import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.hess_buffer = []  # Stores (cat_tuple, H_raw, f, x_cont)
        self.max_buffer = 20
        self.cat_marginals = np.full((6, 6), np.inf)
        self.cat_counts = np.zeros((6, 6))
        self.cat_temp = 0.5
        self.stiffness_factor = 0.0

    def _evaluate_and_update(self, x, func, hess_func):
        if self.evals >= self.budget:
            return float('inf')
        
        # CRITICAL: Boundary enforcement
        x_out = np.clip(x.copy(), -1.0, 1.0)
        cats = np.clip(np.round(x_out[18:24]), 0, 5).astype(int)
        
        f = func(x_out)
        self.evals += 1
        
        if f < self.best_f:
            self.best_f = f
            self.best_x = x_out.copy()
            
        # Update categorical statistics
        for k in range(6):
            val = cats[k]
            self.cat_marginals[k, val] += f
            self.cat_counts[k, val] += 1
            
        # Update Hessian buffer
        if hess_func is not None:
            # Compute x for hess_func with rounded cats
            x_hess = x_out.copy()
            x_hess[18:24] = cats.astype(float)
            H_raw = hess_func(x_hess)[:18, :18]
            
            # Normalize buffer
            if len(self.hess_buffer) >= self.max_buffer:
                # Remove worst cat to keep diversity if needed, or just append
                pass 
            self.hess_buffer.append({
                'cat': tuple(cats),
                'H': H_raw.copy(),
                'f': f,
                'x': x_out[:18].copy()
            })
            
        return f

    def _get_interpolated_hess(self, target_cat, hess_func):
        if not self.hess_buffer:
            return np.eye(18)
            
        w = []
        Hs = []
        for entry in self.hess_buffer:
            dist = np.sum(target_cat != np.array(entry['cat']))
            # Gaussian kernel on Hamming distance
            w.append(np.exp(-dist / 2.0))
            Hs.append(entry['H'])
            
        w = np.array(w)
        H_avg = np.average(Hs, axis=0, weights=w)
        
        # Regularization: Ensure positive definiteness by taking abs eigenvalues
        eigvals, Q = np.linalg.eigh(H_avg)
        H_reg = Q @ np.diag(np.abs(eigvals) + 1e-6) @ Q.T
        
        # Compute trace for stiffness feedback
        self.stiffness_factor = np.trace(H_reg) / 18.0
        return H_reg

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_pop = 45
        pop = np.random.uniform(-1, 1, size=(n_pop, self.dim))
        
        # Adaptive temperature for categorical sampling
        temp_cat = 0.5
        
        while self.evals < self.budget:
            # 1. Compute Categorical Biases
            means = self.cat_marginals / (self.cat_counts + 1e-12)
            # Avoid inf in exp
            means_safe = np.where(np.isinf(means), 0.0, means)
            exp_means = np.exp(-means_safe * temp_cat / np.max([np.max(means_safe), 1e-6]))
            cat_probs = exp_means / np.sum(exp_means, axis=1, keepdims=True)
            
            # 2. Coupling: Continuous stiffness modulates Categorical exploration
            # If manifold is stiff, increase cat temperature to explore materials more
            temp_cat = 0.5 + 2.0 * self.stiffness_factor
            
            # 3. Generate Candidates with Mixed-Variable Bias
            new_pop = np.empty_like(pop)
            for i in range(n_pop):
                # Continuous: Small perturbation of random or biased center
                new_pop[i, :18] = np.random.uniform(-1, 1, 18)
                
                # Categorical: Sample from bias, but allow mutations based on stiffness
                cats = np.zeros(6, dtype=int)
                for k in range(6):
                    # Bias mutation logic
                    best_val = np.argmin(means[k, :])
                    if cats[k] != best_val and np.random.rand() < (0.1 + 0.3 * self.stiffness_factor):
                        cats[k] = best_val
                    else:
                        cats[k] = np.random.choice(6, p=cat_probs[i, k])
                        
                new_pop[i, 18:24] = cats.astype(float)
                
            # 4. Evaluate Population
            pop_f = np.zeros(n_pop)
            for i in range(n_pop):
                if self.evals >= self.budget: break
                pop_f[i] = self._evaluate_and_update(pop[i], func, hess_func)
                
            if self.evals >= self.budget: break
            
            # 5. Categorical Consistency Check & Exploitation
            # Check if best point conflicts with categorical bias
            best_idx = np.argmin(pop_f)
            best_cats = np.clip(np.round(pop[best_idx][18:24]), 0, 5).astype(int)
            best_mean_cats = np.argmin(self.cat_marginals[:6, :6] / (self.cat_counts[:6, :6] + 1e-12), axis=1)
            
            if np.any(best_cats != best_mean_cats):
                # Conflict detected: continuous optimizer is stuck on a suboptimal material slice
                # Force categorical drift towards bias
                drift_cats = best_mean_cats.copy()
                # Apply drift gradually
                for k in range(6):
                    if drift_cats[k] != best_cats[k]:
                        pop[best_idx, 18+k] = float(drift_cats[k])
                pop_f[best_idx] = self._evaluate_and_update(pop[best_idx], func, hess_func)
                if self.evals >= self.budget: break

            # 6. Hessian-Interpolated Trust Region on Elites
            n_elites = 5
            elite_indices = np.argsort(pop_f)[:n_elites]
            
            for idx in elite_indices:
                if self.evals >= self.budget: break
                
                x_elite = pop[idx]
                cats_elite = np.clip(np.round(x_elite[18:24]), 0, 5).astype(int)
                
                # Interpolate Hessian for this categorical context
                H_reg = self._get_interpolated_hess(cats_elite, hess_func)
                
                # Define local objective for trust region
                cat_fixed = cats_elite.copy()
                def obj(xc_sub):
                    if self.evals >= self.budget: return float('inf')
                    return func(np.concatenate([xc_sub, cat_fixed.astype(float)]))
                
                def jac_local(xc_sub):
                    if self.evals >= self.budget: return np.zeros(18)
                    if grad_func:
                        return grad_func(np.concatenate([xc_sub, cat_fixed.astype(float)]))[:18]
                    return np.zeros(18)

                def hess_local(xc_sub):
                    # Return regularized Hessian
                    return H_reg

                res = minimize(obj, x_elite[:18], jac=jac_local, hess=hess_local, 
                               method='trust-constr',
                               bounds=[(-1.0, 1.0)] * 18,
                               options={'maxiter': 15, 'verbose': 0})
                
                if self.evals < self.budget:
                    x_new = np.concatenate([res.x, cat_fixed.astype(float)])
                    f_new = self._evaluate_and_update(x_new, func, hess_func)
                    if f_new < pop_f[idx]:
                        pop[idx] = x_new
                        pop_f[idx] = f_new

        return self.best_f, self.best_x