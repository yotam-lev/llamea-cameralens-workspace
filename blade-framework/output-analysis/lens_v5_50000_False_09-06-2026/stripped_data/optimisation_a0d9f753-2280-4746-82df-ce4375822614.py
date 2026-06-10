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

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_pop = 32
        pop = np.random.uniform(-1, 1, size=(n_pop, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])
        
        while self.evals < self.budget:
            best_idx = np.argmin(pop_f)
            
            # --- Curvature-Conditioned Categorical Drift ---
            # Compute manifold stiffness to guide categorical exploration
            cond_num = 1.0
            mut_prob = 0.25
            if hess_func is not None:
                x_best = pop[best_idx].copy()
                x_best[18:24] = np.clip(np.round(x_best[18:24]), 0, 5).astype(int)
                if self.evals < self.budget:
                    H_raw = hess_func(x_best)
                    eigs = np.linalg.eigvalsh(H_raw)
                    # Regularize eigenvalues to prevent division by zero
                    eigs_reg = np.abs(eigs) + 1e-6
                    cond_num = np.max(eigs_reg) / np.min(eigs_reg)
                    # Stiff manifolds (high cond) require more categorical exploration
                    # to jump between disjoint valleys; flat regions require exploitation
                    mut_prob = np.clip(0.05 + 0.75 / (1.0 + np.log1p(cond_num)), 0.05, 0.80)
                    
                # Apply curvature-aware categorical mutations
                cat_mutations = np.zeros((n_pop, 6), dtype=int)
                mask = np.random.rand(n_pop, 6) < mut_prob
                cat_mutations[mask] = np.random.choice([-1, 1], size=np.sum(mask))
                pop[:, 18:24] += cat_mutations
                pop[:, 18:24] = np.clip(np.round(pop[:, 18:24]), 0, 5).astype(int)
            
            # --- Dynamic Manifold Activation ---
            # If stiffness is extreme, force categorical neighbors evaluation
            # to detect if a better manifold exists
            if cond_num > 1e4 and self.evals < self.budget:
                for i in range(n_pop):
                    if self.evals >= self.budget: break
                    # Sample a categorical neighbor
                    neighbor = pop[i].copy()
                    flip_dim = np.random.randint(18, 24)
                    neighbor[flip_dim] = np.clip(neighbor[flip_dim] + np.random.choice([-1, 1]), 0, 5)
                    f_cand = self._evaluate(neighbor, func)
                    if f_cand < pop_f[i]:
                        pop[i] = neighbor
                        pop_f[i] = f_cand
                        
            # --- Continuous Refinement via Trust Region ---
            # Exploit the current manifold using Hessian information
            if hess_func is not None and self.evals < self.budget:
                best_idx = np.argmin(pop_f)
                xc = pop[best_idx][:18].copy()
                cat = np.clip(np.round(pop[best_idx][18:24]), 0, 5).astype(int)
                
                # Compute and regularize Hessian for solver
                x_best = pop[best_idx].copy()
                x_best[18:24] = cat
                H_raw = hess_func(x_best)
                eigs, Q = np.linalg.eigh(H_raw)
                # Ensure positive-definite by taking absolute eigenvalues
                eigs_reg = np.abs(eigs) + 1e-4
                H_reg = Q @ np.diag(eigs_reg) @ Q.T
                
                def obj(xs): return func(np.concatenate([xs, cat]))
                def jac(xs): return grad_func(np.concatenate([xs, cat]))[:18] if grad_func else np.zeros(18)
                def hess(xs): return H_reg
                
                try:
                    res = minimize(obj, xc, jac=jac, hess=hess, method='trust-constr',
                                   bounds=[(-1.0, 1.0)]*18, options={'maxiter': 20, 'verbose': 0})
                    
                    x_ref = np.concatenate([res.x, cat])
                    if self.evals < self.budget:
                        f_ref = self._evaluate(x_ref, func)
                        if f_ref < pop_f[best_idx]:
                            pop[best_idx] = x_ref
                            pop_f[best_idx] = f_ref
                except Exception:
                    pass  # Fallback to existing point if optimization fails
                    
        return self.best_f, self.best_x