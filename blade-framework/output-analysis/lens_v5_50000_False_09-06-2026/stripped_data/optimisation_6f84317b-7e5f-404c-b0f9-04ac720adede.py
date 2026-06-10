import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        self.n_pop = 45
        self.F_base = 0.6
        self.cr_base = 0.8
        self.history_len = 10
        self.flat_eig_ratio = 0.05
        
        # Adaptive tracking
        self.f_history = []
        self.stagnation_counter = 0
        self.grad_mag_history = []
        
        # Hessian tracking
        self.H_reg = None
        self.H_inv = None
        self.cond = 1e9
        self.max_eig = 0.0
        self.min_eig = 1e-9
        self.H_valid = False
        self.hess_func = None
        self.last_hess_eval_x = None
        self.reg_shift = 0.0
        
        # Control parameters
        self.F = 0.6
        self.cr = 0.8
        self.ls_freq = 10
        self.cat_perturb_prob = 0.2
        self.grad_weight = 0.0

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
            
            # Corrected regularization: eigenvalue shifting preserves curvature direction
            min_val = np.min(vals)
            shift = max(1e-6, -min_val + 1e-6) if min_val < 0 else 1e-6
            vals_reg = vals + shift
            
            self.reg_shift = shift
            self.H_reg = vecs @ np.diag(vals_reg) @ vecs.T
            self.H_inv = np.linalg.inv(self.H_reg)
            self.max_eig = np.max(vals_reg)
            self.min_eig = np.min(vals_reg)
            self.cond = self.max_eig / max(self.min_eig, 1e-9)
            self.H_valid = True
            self.last_hess_eval_x = x.copy()
        except Exception:
            self.H_valid = False

    def _update_gradient(self, x, grad_func):
        if grad_func is None:
            self.grad_weight = 0.0
            return
        try:
            g = np.asarray(grad_func(x[:18]), dtype=float)
            mag = np.linalg.norm(g)
            self.grad_mag_history.append(mag)
            if len(self.grad_mag_history) > 10:
                self.grad_mag_history.pop(0)
            self.grad_weight = mag / (max(mag, 1e-6) + 1e-6)
        except Exception:
            self.grad_weight = 0.0

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

        # Adaptive control logic
        if self.H_valid and self.cond > 600:
            self.F = np.clip(0.2 + 0.3 * progress, 0.15, 0.5)
            self.cr = 0.85 + 0.1 * progress
            self.ls_freq = max(4, int(18 * (1 - progress)))
            self.cat_perturb_prob = np.clip(0.2 + 0.25 * (self.cond / 1000), 0.2, 0.45)
        elif self.stagnation_counter > 3:
            self.F = np.clip(0.35 + 0.4 * (1 - progress), 0.25, 0.85)
            self.cr = 0.55 + 0.25 * (1 - progress)
            self.ls_freq = max(7, int(24 * (1 - progress)**1.5))
            self.cat_perturb_prob = np.clip(0.15 + 0.2 * (1 - progress), 0.1, 0.35)
        else:
            self.F = np.clip(0.08 + 0.2 * (1 - progress), 0.05, 0.28)
            self.cr = 0.35 + 0.45 * progress
            self.ls_freq = max(5, int(16 * (1 - progress)))
            self.cat_perturb_prob = np.clip(0.04 + 0.08 * progress, 0.04, 0.12)

        # Gradient-based scaling
        if len(self.grad_mag_history) > 0:
            avg_gm = np.mean(self.grad_mag_history)
            if avg_gm > 10.0:
                self.F *= 0.7
            elif avg_gm < 0.5:
                self.F *= 1.1

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
            
            # Update Hessian and Gradient
            if self.hess_func and (not self.H_valid or it % 10 == 0):
                if self.evals < self.budget - 5:
                    self._update_hessian(x_best)
            
            if grad_func and self.evals < self.budget - 3:
                self._update_gradient(x_best, grad_func)

            new_pop = np.empty_like(pop)
            for i in range(self.n_pop):
                if self.evals >= self.budget: break
                
                r1, r2, r3 = np.random.choice(self.n_pop, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = np.random.choice(self.n_pop, 3, replace=False)
                    
                diff_cont = pop[r1, :18] - pop[r2, :18]
                
                # Curvature-adaptive mutation with gradient augmentation
                if self.H_valid:
                    d_cont = self.H_inv @ diff_cont
                    flat_mask = np.abs(self.H_inv.diagonal()) < self.flat_eig_ratio * np.max(np.abs(self.H_inv.diagonal()))
                    if flat_mask.any():
                        d_cont[flat_mask] *= self.cond * self.flat_eig_ratio
                    
                    norm_diff = np.linalg.norm(diff_cont)
                    norm_d = np.linalg.norm(d_cont)
                    if norm_d > 0 and norm_d > 1.4 * norm_diff:
                        d_cont *= (1.4 * norm_diff / norm_d)
                    diff_cont = d_cont

                v_cont = pop[r3, :18] + self.F * diff_cont
                v_cat = pop[r3, 18:24].copy()
                
                # Gradient augmentation
                if self.grad_weight > 0.1 and len(self.grad_mag_history) > 0:
                    v_cont += self.grad_weight * 0.1 * np.random.randn(18)

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
            if self.evals < self.budget - 12 and it % self.ls_freq == 0:
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
                                      options={'maxiter': 40, 'verbose': 0})
                        if res.success:
                            cand = np.concatenate([res.x, cat_int])
                            f_c = self._evaluate(cand, func)
                    except Exception:
                        pass
                        
        return self.best_f, self.best_x