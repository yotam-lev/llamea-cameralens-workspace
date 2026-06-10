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
        self.H_inv = np.eye(18)
        self.H_valid = False
        self.hess_func = None
        self.eigenvalues = np.zeros(18)
        self.eigenvectors = np.eye(18)
        self.max_eig = 0.0
        self.min_eig = 1e-9
        self.cond = 1e9

        # Adaptive tracking
        self.best_f_history = []
        self.stagnation = 0.0
        self.history_len = 5

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
            vals, vecs = np.linalg.eigh(H)
            # Strict PD regularization via absolute eigenvalues
            H_reg = vecs @ np.diag(np.abs(vals) + 1e-8) @ vecs.T
            self.H = H_reg
            self.H_inv = np.linalg.inv(H_reg)
            self.eigenvalues = np.abs(vals)
            self.eigenvectors = vecs
            self.max_eig = np.max(self.eigenvalues)
            self.min_eig = np.min(self.eigenvalues)
            self.cond = self.max_eig / max(self.min_eig, 1e-9)
            self.H_valid = True
        except Exception:
            self.H_valid = False

    def _adapt_parameters(self):
        self.best_f_history.append(self.best_f)
        if len(self.best_f_history) > self.history_len:
            self.best_f_history.pop(0)

        if len(self.best_f_history) >= self.history_len:
            rel_imp = (self.best_f_history[-1] - self.best_f_history[0]) / max(abs(self.best_f_history[-1]), 1e-9)
            self.stagnation = np.clip(1.0 - rel_imp, 0.0, 1.0)
        else:
            self.stagnation = 0.0

        progress = self.evals / self.budget

        # Adaptive Mutation Scale: Amplify F when stagnant or early; shrink when exploiting
        self.F = np.clip(0.4 + 0.5 * self.stagnation + 0.3 * (1 - progress), 0.2, 1.2)

        # Adaptive Local Search Frequency: Increase when budget depletes or stagnation rises
        self.ls_freq = max(5, int(35 * (1 - progress)**2 + 15 * self.stagnation))

        # Adaptive Categorical Perturbation: Higher when stuck in flat spectral directions or early stage
        self.cat_perturb_prob = np.clip(0.05 + 0.15 * self.stagnation + 0.1 * np.exp(-3 * progress), 0.05, 0.3)

        # Adaptive Trap Threshold: Loosen detection as optimization progresses
        self.trap_cond_thresh = np.clip(80 + 40 * progress, 80, 120)

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
            self._adapt_parameters()

            x_best = self.best_x

            if self.hess_func and (not self.H_valid or it % 10 == 0):
                if self.evals < self.budget:
                    self._update_hessian_and_spectra(x_best)

            new_pop = np.zeros_like(pop)
            cat_perturb = self.H_valid and self.cond > self.trap_cond_thresh

            for i in range(self.n_pop):
                if self.evals >= self.budget: break

                r1, r2, r3 = np.random.choice(self.n_pop, 3, replace=False)
                diff_cont = pop[r1, :18] - pop[r2, :18]

                if self.H_valid:
                    if cat_perturb:
                        flat_mask = self.eigenvalues < (self.flat_eig_ratio * self.max_eig)
                        if flat_mask.any():
                            V_flat = self.eigenvectors[:, flat_mask]
                            d_cont = V_flat @ (V_flat.T @ diff_cont)
                        else:
                            d_cont = diff_cont
                    else:
                        d_cont = self.H_inv @ diff_cont
                else:
                    d_cont = diff_cont

                v_cont = pop[r3, :18] + self.F * d_cont
                v_cat = pop[r3, 18:24].copy()

                if cat_perturb and np.random.rand() < self.cat_perturb_prob:
                    v_cat = np.random.randint(0, 6, size=6)

                trial_cont = np.copy(v_cont)
                trial_cat = np.copy(v_cat)
                cross_mask = np.random.rand(self.dim) < self.cr
                cross_mask[18:24] = False
                trial_cont[cross_mask[:18]] = pop[i, :18][cross_mask[:18]]
                trial_cat[cross_mask[18:24]] = pop[i, 18:24][cross_mask[18:24]].astype(int)

                trial_cont = np.clip(trial_cont, -1.0, 1.0)
                trial_cat = np.clip(np.round(trial_cat), 0, 5).astype(int)

                trial = np.concatenate([trial_cont, trial_cat])

                f_trial = self._evaluate(trial, func)
                f_current = self._evaluate(pop[i], func)
                if f_trial <= f_current:
                    new_pop[i] = trial
                else:
                    new_pop[i] = pop[i]

            pop = new_pop

            if self.evals < self.budget and it % self.ls_freq == 0 and not cat_perturb:
                xb = self.best_x[:18].copy()
                cat_int = self.best_x[18:24].copy()
                if self.H_valid:
                    try:
                        cat_int_fixed = cat_int.copy()
                        def refine_func(xc):
                            if self.evals >= self.budget: return float('inf')
                            full_x = np.concatenate([xc, cat_int_fixed])
                            xc_clip = np.clip(full_x, -1.0, 1.0)
                            xc_clip[18:24] = np.clip(np.round(xc_clip[18:24]), 0, 5).astype(int)
                            f = func(xc_clip)
                            self.evals += 1
                            if f < self.best_f:
                                self.best_f = f
                                self.best_x = xc_clip.copy()
                            return f

                        res = minimize(refine_func, xb, method='trust-constr',
                                      hess=lambda xc: self.H,
                                      bounds=[(-1.0, 1.0)] * 18,
                                      options={'maxiter': 30, 'verbose': 0})
                        if res.success:
                            cand = np.concatenate([res.x, cat_int_fixed])
                            f_c = self._evaluate(cand, func)
                            if f_c < self.best_f:
                                self.best_x = cand.copy()
                    except Exception:
                        pass

        return self.best_f, self.best_x