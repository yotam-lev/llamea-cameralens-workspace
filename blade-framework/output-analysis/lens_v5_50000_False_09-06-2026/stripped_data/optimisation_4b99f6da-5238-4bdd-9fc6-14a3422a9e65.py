import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.n_pop = 60
        self.F = 1.0
        self.cr = 0.8
        self.stress_thresh = 40.0
        self.ls_freq = 20
        self.H = None
        self.eigs = None
        self.evecs = None
        self.H_valid = False
        self.mat_change_counter = 0
        self.hess_func = None
        self.func = None
        self.pop_f = None

    def _clip_and_map(self, x):
        xc = np.clip(x, -1.0, 1.0)
        xc[18:24] = np.clip(np.round(xc[18:24]), 0, 5).astype(int)
        return xc

    def _evaluate(self, x):
        if self.evals >= self.budget:
            return float('inf')
        xc = self._clip_and_map(x)
        f = self.func(xc)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = xc.copy()
        return f

    def _update_spectral(self, x):
        try:
            H = self.hess_func(x)
            eigs, evecs = np.linalg.eigh(H)
            min_eig = eigs.min()
            reg = max(0.0, 1e-2 - min_eig)
            self.H = H + reg * np.eye(18)
            self.eigs = eigs
            self.evecs = evecs
            self.H_valid = True
            self.max_eig = np.max(eigs)
            self.min_eig = max(np.min(eigs), 1e-9)
        except Exception:
            self.H_valid = False

    def _get_compatible_material(self, x, stress):
        if stress > self.stress_thresh:
            return np.random.randint(0, 6, size=6)
        return x[18:24].copy()

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.func = func
        self.hess_func = hess_func
        
        pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        
        self.pop_f = np.array([self._evaluate(pop[i]) for i in range(self.n_pop)])

        it = 0
        while self.evals < self.budget:
            it += 1
            
            best_idx = np.argmin(self.pop_f)
            x_best = pop[best_idx]

            if not self.H_valid or it % 10 == 0:
                if self.evals < self.budget:
                    self._update_spectral(x_best)
                    self.pop_f[best_idx] = self._evaluate(x_best)
                    best_idx = np.argmin(self.pop_f)
                    x_best = pop[best_idx]

            if not self.H_valid:
                self.H_valid = True
                self._update_spectral(x_best)

            stress = np.sum(np.abs(self.eigs)) * np.linalg.norm(x_best[:18])
            new_mat = self._get_compatible_material(x_best, stress)
            self.mat_change_counter = 0 if np.any(new_mat != x_best[18:24]) else self.mat_change_counter + 1

            new_pop = np.zeros_like(pop)
            
            for i in range(self.n_pop):
                if self.evals >= self.budget:
                    break
                    
                r1, r2, r3 = np.random.choice(self.n_pop, 3, replace=False)
                diff = pop[r1, :18] - pop[r2, :18]
                
                if self.H_valid and self.mat_change_counter == 0:
                    inv_sqrt_eigs = 1.0 / np.sqrt(np.abs(self.eigs))
                    inv_sqrt_eigs = np.clip(inv_sqrt_eigs, 0.1, 10.0)
                    diff = self.evecs @ (inv_sqrt_eigs * (self.evecs.T @ diff))
                
                v_cont = pop[r3, :18] + self.F * diff
                v_cat = pop[r3, 18:24].copy()
                
                if self.mat_change_counter > 0 and np.random.rand() < 0.2:
                    v_cat = new_mat

                trial_cont = np.copy(v_cont)
                trial_cat = np.copy(v_cat)
                cross_mask = np.random.rand(self.dim) < self.cr
                cross_mask[18:24] = False
                trial_cont[cross_mask[:18]] = pop[i, :18][cross_mask[:18]]
                trial_cat[cross_mask[18:24]] = pop[i, 18:24][cross_mask[18:24]].astype(int)

                trial_cont = np.clip(trial_cont, -1.0, 1.0)
                trial_cat = np.clip(np.round(trial_cat), 0, 5).astype(int)
                
                trial = np.concatenate([trial_cont, trial_cat])
                
                f_trial = self._evaluate(trial)
                f_curr = self.pop_f[i]
                
                if f_trial <= f_curr:
                    new_pop[i] = trial
                    self.pop_f[i] = f_trial
                else:
                    new_pop[i] = pop[i]

            pop = new_pop

            if self.evals < self.budget and it % self.ls_freq == 0 and self.H_valid:
                xb = x_best[:18].copy()
                cat_int = x_best[18:24].copy()
                try:
                    res = minimize(
                        lambda xc: self._evaluate(np.concatenate([xc, cat_int])),
                        xb, method='trust-constr',
                        hess=lambda xc: self.H,
                        bounds=[(-1.0, 1.0)] * 18,
                        options={'maxiter': 30, 'verbose': 0}
                    )
                    if res.success:
                        cand = np.concatenate([res.x, cat_int])
                        f_c = self._evaluate(cand)
                        if f_c < self.best_f:
                            self.best_x = cand.copy()
                except Exception:
                    pass

        return self.best_f, self.best_x