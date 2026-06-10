import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        self.n_pop = 40
        self.F_base = 0.6
        self.cr_base = 0.8
        self.history_len = 8
        self.flat_eig_ratio = 0.05
        
        # Adaptive tracking
        self.f_history = []
        self.stagnation_counter = 0
        
        # Hessian tracking
        self.H = None
        self.H_reg = None
        self.H_inv = None
        self.cond = 1e9
        self.max_eig = 0.0
        self.min_eig = 1e-9
        self.H_valid = False
        self.hess_func = None
        self.last_hess_eval_x = None
        self.damping = 1e-8
        
        # Control parameters
        self.F = 0.6
        self.cr = 0.8
        self.ls_freq = 10
        self.cat_perturb_prob = 0.2

    def _clip_and_map(self, x):
        x = np.asarray(x, dtype=float)
        x[:18] = np.clip(x[:18], -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        return x

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

    def _update_hessian(self, x):
        try:
            H = self.hess_func(x)
            vals, vecs = np.linalg.eigh(H)
            # Strict PD regularization via absolute eigenvalues + adaptive damping
            min_val = np.min(vals)
            damp = max(1e-6, np.abs(min_val) * 0.05) if min_val < 0 else 1e-6
            vals_reg = np.abs(vals) + damp
            self.H_reg = vecs @ np.diag(vals_reg) @ vecs.T
            self.H_inv = np.linalg.inv(self.H_reg)
            self.max_eig = np.max(vals_reg)
            self.min_eig = np.min(vals_reg)
            self.cond = self.max_eig / max(self.min_eig, 1e-9)
            self.H_valid = True
            self.last_hess_eval_x = x.copy()
        except Exception:
            self.H_valid = False

    def _adapt_params(self):
        self.f_history.append(self.best_f)
        if len(self.f_history) > self.history_len:
            self.f_history.pop(0)
            
        progress = self.evals / self.budget
        imp = 0.0
        if len(self.f_history) >= self.history_len:
            imp = (self.f_history[-1] - self.f_history[-self.history_len]) / max(abs(self.f_history[-self.history_len]), 1e-9)
            
        stagnant = imp > -1e-7
        if stagnant:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        # Regime-aware parameter scheduling
        if self.H_valid and self.cond > 800:
            # Stiff landscape: damp mutation, frequent local search, high categorical noise
            self.F = np.clip(0.25 + 0.35 * progress, 0.2, 0.6)
            self.cr = 0.9 + 0.05 * progress
            self.ls_freq = max(5, int(22 * (1 - progress)))
            self.cat_perturb_prob = np.clip(0.25 + 0.3 * (self.cond / 1000), 0.2, 0.5)
            self.damping = max(self.damping, 0.001)
        elif stagnant:
            # Stagnant: broaden exploration, moderate local search
            self.F = np.clip(0.4 + 0.6 * (1 - progress), 0.3, 1.0)
            self.cr = 0.6 + 0.3 * (1 - progress)
            self.ls_freq = max(8, int(28 * (1 - progress)**2))
            self.cat_perturb_prob = np.clip(0.15 + 0.25 * (1 - progress), 0.1, 0.4)
            self.damping = max(self.damping, 1e-5)
        else:
            # Converging: fine-tune, reduce categorical noise, lean on trust-region
            self.F = np.clip(0.1 + 0.25 * (1 - progress), 0.05, 0.35)
            self.cr = 0.4 + 0.5 * progress
            self.ls_freq = max(6, int(18 * (1 - progress)))
            self.cat_perturb_prob = np.clip(0.05 + 0.1 * progress, 0.05, 0.15)
            self.damping = max(self.damping, 1e-6)

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        
        for i in range(self.n_pop):
            if self.evals >= self.budget: break
            self._evaluate(pop[i], func)
            
        if self.evals >= self.budget:
            return self.best_f, self.best_x

        it = 0
        while self.evals < self.budget:
            it += 1
            self._adapt_params()
            x_best = self.best_x
            
            # Update Hessian periodically or if invalid
            if self.hess_func and (not self.H_valid or it % 12 == 0):
                if self.evals < self.budget - 5:
                    self._update_hessian(x_best)

            new_pop = np.empty_like(pop)
            for i in range(self.n_pop):
                if self.evals >= self.budget: break
                
                r1, r2, r3 = np.random.choice(self.n_pop, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = np.random.choice(self.n_pop, 3, replace=False)
                    
                diff_cont = pop[r1, :18] - pop[r2, :18]
                
                # Curvature-adaptive direction
                if self.H_valid:
                    d_cont = self.H_inv @ diff_cont
                    # Dampen steps along flat eigenvectors to prevent instability
                    flat_mask = self.eigenvalues < (self.flat_eig_ratio * self.max_eig) if self.H_valid else np.zeros(18, dtype=bool)
                    if flat_mask.any():
                        d_cont[flat_mask] *= self.cond * self.flat_eig_ratio
                    # Clamp step magnitude relative to original difference
                    norm_diff = np.linalg.norm(diff_cont)
                    norm_d = np.linalg.norm(d_cont)
                    if norm_d > 0 and norm_d > 1.5 * norm_diff:
                        d_cont *= (1.5 * norm_diff / norm_d)
                    diff_cont = d_cont

                v_cont = pop[r3, :18] + self.F * diff_cont
                v_cat = pop[r3, 18:24].copy()

                # Categorical perturbation
                if np.random.rand() < self.cat_perturb_prob:
                    v_cat = np.random.randint(0, 6, size=6)

                # Crossover
                trial_cont = np.copy(v_cont)
                trial_cat = np.copy(v_cat)
                cross_mask = np.random.rand(self.dim) < self.cr
                cross_mask[18:24] = False
                trial_cont[cross_mask[:18]] = pop[i, :18][cross_mask[:18]]
                trial_cat[cross_mask[18:24]] = pop[i, 18:24][cross_mask[18:24]].astype(int)

                trial = np.concatenate([np.clip(trial_cont, -1, 1), np.clip(np.round(trial_cat), 0, 5).astype(int)])
                
                f_trial = self._evaluate(trial, func)
                f_current = self._evaluate(pop[i], func)
                new_pop[i] = trial if f_trial <= f_current else pop[i]
                
            pop = new_pop

            # Local Search Trigger
            if self.evals < self.budget - 15 and it % self.ls_freq == 0:
                xb = self.best_x[:18].copy()
                cat_int = self.best_x[18:24].copy()
                if self.H_valid:
                    try:
                        def ref_func(xc):
                            full = np.concatenate([xc, cat_int])
                            return self._evaluate(full, func)
                            
                        res = minimize(ref_func, xb, method='trust-constr',
                                      hess=lambda xc: self.H_reg,
                                      bounds=[(-1.0, 1.0)] * 18,
                                      options={'maxiter': 35, 'verbose': 0})
                        if res.success:
                            cand = np.concatenate([res.x, cat_int])
                            f_c = self._evaluate(cand, func)
                    except Exception:
                        pass
                        
        return self.best_f, self.best_x