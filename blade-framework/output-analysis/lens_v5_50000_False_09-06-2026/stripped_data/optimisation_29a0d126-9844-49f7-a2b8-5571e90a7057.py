import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.F = 0.8
        self.cr = 0.9
        self.n_pop = 40
        self.H = None
        self.H_inv = None
        self.eig_vals = None
        self.eig_vecs = None
        self.H_valid = False
        self.hess_func = None
        self.ref_cat = None

    def _clip_and_map(self, x):
        xc = np.clip(x, -1.0, 1.0)
        xc[18:24] = np.clip(np.round(xc[18:24]), 0, 5).astype(int)
        return xc

    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        xc = self._clip_and_map(x)
        f = func(xc)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = xc.copy()
        return f

    def _update_geometry(self, x):
        try:
            H = self.hess_func(x[:18])
            eigs, evecs = np.linalg.eigh(H)
            # Spectral Regularization for PD guarantee
            shift = max(0, 1e-4 - eigs.min())
            H_reg = H + shift * np.eye(18)
            self.H = H_reg
            self.H_inv = np.linalg.inv(H_reg)
            self.eig_vals = eigs
            self.eig_vecs = evecs
            self.H_valid = True
        except Exception:
            self.H_valid = False

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        self.ref_cat = pop[0, 18:24].copy()
        
        # Initial evaluation
        for i in range(self.n_pop):
            if self.evals >= self.budget: break
            self._evaluate(pop[i], func)
            
        gbest_idx = np.argmin([self._evaluate(p, func) for p in pop])
        
        it = 0
        while self.evals < self.budget:
            it += 1
            if self.evals >= self.budget: break
            
            # Categorical-Aware Hessian Invalidation
            current_cat = pop[gbest_idx, 18:24].copy()
            if not self.H_valid or not np.array_equal(current_cat, self.ref_cat):
                self._update_geometry(pop[gbest_idx])
                self.ref_cat = current_cat

            # Compute spectral metrics for anisotropic scaling
            cond = 1.0
            low_eig_proj = np.zeros(18)
            if self.H_valid:
                min_eig = max(self.eig_vals.min(), 1e-8)
                cond = np.max(self.eig_vals) / min_eig
                # Project onto low-curvature subspace (top ~log10(cond) eigenvectors)
                k = max(1, int(np.round(np.log10(cond))))
                low_eig_proj = self.eig_vecs[:, :k].sum(axis=1)

            new_pop = np.zeros_like(pop)
            for i in range(self.n_pop):
                if self.evals >= self.budget: break
                
                r1, r2, r3 = np.random.choice(self.n_pop, 3, replace=False)
                diff = pop[r1, :18] - pop[r2, :18]
                
                # Newtonian preconditioning
                d_newton = np.dot(self.H_inv, diff) if self.H_valid else diff
                
                # Curvature-Anisotropic Jump
                d_jump = np.zeros(18)
                if self.H_valid:
                    jump_scale = 0.5 * np.log2(max(cond, 2))
                    d_jump = jump_scale * low_eig_proj * np.random.randn()
                
                v_cont = pop[r3, :18] + self.F * (d_newton + d_jump)
                
                # Categorical crossover
                trial_cat = pop[i, 18:24].copy()
                cross_idx = np.random.rand(6) < self.cr
                trial_cat[cross_idx] = pop[r3, 18:24][cross_idx]
                
                # Boundary & Mapping
                trial_cont = np.clip(v_cont, -1.0, 1.0)
                trial_cat = np.clip(np.round(trial_cat), 0, 5).astype(int)
                trial = np.concatenate([trial_cont, trial_cat])
                
                f_trial = self._evaluate(trial, func)
                f_curr = self._evaluate(pop[i], func)
                
                if f_trial <= f_curr:
                    new_pop[i] = trial
                    if f_trial < self._evaluate(pop[gbest_idx], func):
                        gbest_idx = i
                else:
                    new_pop[i] = pop[i]
            
            pop = new_pop
            
            # Trust-Region Refinement with Adaptive Radius
            if self.evals < self.budget and it % 10 == 0 and self.H_valid:
                xb = pop[gbest_idx, :18].copy()
                cat = pop[gbest_idx, 18:24].copy()
                # Radius inversely proportional to peak curvature
                radius = 0.5 / np.sqrt(np.max(self.eig_vals))
                try:
                    res = minimize(
                        lambda xc: func(np.concatenate([xc, cat])),
                        xb, method='trust-constr',
                        hess=lambda xc: self.H,
                        bounds=[(-1.0, 1.0)]*18,
                        options={'maxiter': 20, 'verbose': 0}
                    )
                    if res.success:
                        cand = np.concatenate([res.x, cat])
                        f_c = self._evaluate(cand, func)
                        if f_c < self.best_f:
                            self.best_f = f_c
                            self.best_x = cand.copy()
                            pop[gbest_idx] = cand
                except Exception:
                    pass
        return self.best_f, self.best_x