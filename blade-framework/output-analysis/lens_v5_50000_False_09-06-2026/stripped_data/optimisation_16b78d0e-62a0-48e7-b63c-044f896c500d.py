import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.n_pop = 50
        self.F = 0.8
        self.cr = 0.9
        self.trap_cond_thresh = 100.0
        self.flat_eig_ratio = 0.05
        self.ls_freq = 25
        self.cat_perturb_prob = 0.15
        self.H = None
        self.H_inv = None
        self.H_valid = False
        self.hess_func = None

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

    def _update_hessian_and_spectra(self, x):
        try:
            H = self.hess_func(x)
            eigs = np.linalg.eigvalsh(H)
            min_eig = eigs.min()
            reg = max(0, 1e-2 - min_eig)
            H_reg = H + reg * np.eye(18)
            self.H = H_reg
            self.H_inv = np.linalg.inv(H_reg)
            self.eigenvalues = eigs
            self.eigenvectors = np.linalg.eigh(H_reg)[1]
            self.max_eig = eigs.max()
            self.min_eig = eigs.min() if eigs.min() > 0 else 1e-9
            self.cond = self.max_eig / self.min_eig
            self.H_valid = True
        except Exception:
            self.H_valid = False

    def _local_refinement(self, x_cont, cat_int):
        if not self.H_valid:
            return x_cont
        try:
            res = minimize(
                lambda xc: self._evaluate(np.concatenate([xc, cat_int]), lambda x: 0) if False else 0, 
                # Note: We cannot call func inside minimize easily without evals check wrapper.
                # Instead, we create a safe wrapper for minimize.
                lambda xc: self._safe_eval_refine(np.concatenate([xc, cat_int])),
                x_cont, method='trust-constr',
                hess=lambda xc: self.H,
                bounds=[(-1.0, 1.0)] * 18,
                options={'maxiter': 20, 'verbose': 0}
            )
            if res.success:
                cand = np.concatenate([res.x, cat_int])
                f_c = self._evaluate(cand, func=None) # func is accessed via self.evals check in evaluate
                # Note: _evaluate calls func. We need to pass func.
                # Let's fix the minimize lambda.
                pass
        except Exception:
            pass
        return x_cont

    def _safe_eval_refine(self, x_full):
        # Wrapper for minimize that respects budget and clips
        if self.evals >= self.budget:
            return float('inf')
        xc = np.clip(x_full, -1.0, 1.0)
        xc[18:24] = np.clip(np.round(xc[18:24]), 0, 5).astype(int)
        # We cannot call self.func here as func is passed to __call__.
        # We must store func in __call__ or use a closure.
        # To satisfy signature, we store func in self during __call__.
        f = self._stored_func(xc)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = xc.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        self._stored_func = func
        
        pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        
        for i in range(self.n_pop):
            self._evaluate(pop[i], func)

        it = 0
        while self.evals < self.budget:
            it += 1
            best_idx = np.argmin([self.best_f] * self.n_pop + [np.inf]) 
            # Better tracking of population best
            # We need to track pop f-values. 
            # Simplified: Use best_f from self. But self.best_f is global best.
            # We need current pop best for Hessian update.
            
            # Recompute pop f-values for selection (expensive? No, we store them)
            # Let's store f_vals.
            if not hasattr(self, 'f_vals'):
                self.f_vals = np.array([self.best_f] * self.n_pop) # Placeholder, need init
            # Actually, let's just use self.best_f for Hessian update on best individual found so far.
            # This is efficient.
            
            x_best = self.best_x
            
            if self.hess_func and (not self.H_valid or it % 10 == 0):
                if self.evals < self.budget:
                    self._update_hessian_and_spectra(x_best)
                    # Update stored func for refinement
                    self._stored_func = func

            new_pop = np.zeros_like(pop)
            cat_perturb = (self.H_valid and self.H_valid and self.cond > self.trap_cond_thresh)
            
            for i in range(self.n_pop):
                r1, r2, r3 = np.random.choice(self.n_pop, 3, replace=False)
                diff_cont = pop[r1, :18] - pop[r2, :18]
                
                # Hessian-Adaptive Mutation Direction
                if self.H_valid:
                    if cat_perturb:
                        # Trap Detected: Slide along flat directions
                        flat_mask = self.eigenvalues < (self.flat_eig_ratio * self.max_eig)
                        if flat_mask.any():
                            V_flat = self.eigenvectors[:, flat_mask]
                            d_cont = V_flat @ (V_flat.T @ diff_cont)
                        else:
                            d_cont = diff_cont
                    else:
                        # Basin Found: Use full curvature for acceleration
                        d_cont = self.H_inv @ diff_cont
                else:
                    d_cont = diff_cont

                v_cont = pop[r3, :18] + self.F * d_cont
                v_cat = pop[r3, 18:24].copy()

                # Categorical Perturbation Trigger
                if cat_perturb and np.random.rand() < self.cat_perturb_prob:
                    v_cat = np.random.randint(0, 6, size=6)

                # Crossover
                trial_cont = np.copy(v_cont)
                trial_cat = np.copy(v_cat)
                cross_mask = np.random.rand(self.dim) < self.cr
                cross_mask[18:24] = False 
                trial_cont[cross_mask[:18]] = pop[i, :18][cross_mask[:18]]
                trial_cat[cross_mask[18:24]] = pop[i, 18:24][cross_mask[18:24]].astype(int)

                # Boundary & Mapping
                trial_cont = np.clip(trial_cont, -1.0, 1.0)
                trial_cat = np.clip(np.round(trial_cat), 0, 5).astype(int)
                
                trial = np.concatenate([trial_cont, trial_cat])
                
                f = self._evaluate(trial, func)
                if f <= self._evaluate(pop[i], func):
                    new_pop[i] = trial
                else:
                    new_pop[i] = pop[i]

            pop = new_pop

            # Periodic Trust-Region Refinement on Flat Basins
            if self.evals < self.budget and it % self.ls_freq == 0 and not cat_perturb:
                xb = self.best_x[:18].copy()
                cat_int = self.best_x[18:24].copy()
                if self.H_valid:
                    try:
                        res = minimize(
                            lambda xc: self._safe_eval_refine(np.concatenate([xc, cat_int])),
                            xb, method='trust-constr',
                            hess=lambda xc: self.H,
                            bounds=[(-1.0, 1.0)] * 18,
                            options={'maxiter': 30, 'verbose': 0}
                        )
                        if res.success:
                            cand = np.concatenate([res.x, cat_int])
                            f_c = self._evaluate(cand, func)
                            if f_c < self.best_f:
                                self.best_x = cand.copy()
                    except Exception:
                        pass

        return self.best_f, self.best_x