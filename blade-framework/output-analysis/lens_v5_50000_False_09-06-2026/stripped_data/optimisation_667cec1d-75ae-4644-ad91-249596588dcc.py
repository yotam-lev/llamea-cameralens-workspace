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
        self.F = 0.6
        self.cr = 0.85
        self.H = None
        self.H_reg = None
        self.V = None
        self.lam_reg = None
        self.cond = 1e9
        self.H_valid = False
        self.refine_count = 0
        self.cat_hist = np.ones(6)
        self.hess_func = None
        self.func = None

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
            self.cat_hist[xc[18:24]] += 1.0
        return f

    def _update_spectral_info(self, x):
        try:
            H_raw = self.hess_func(x)
            lam, V = np.linalg.eigh(H_raw)
            # Strict regularization: absolute value + shift ensures positive-definiteness
            self.lam_reg = np.abs(lam) + 1e-6
            self.H_reg = V @ np.diag(self.lam_reg) @ V.T
            self.V = V
            self.cond = self.lam_reg[-1] / self.lam_reg[0]
            self.H_valid = True
        except Exception:
            self.H_valid = False

    def _refine_basin(self, x_cont, cat_int):
        if not self.H_valid or self.refine_count >= 4:
            return x_cont
        if self.cond > 30.0:  # Gate refinement to well-conditioned basins only
            return x_cont
        self.refine_count += 1
        try:
            res = minimize(
                lambda xc: self._evaluate(np.concatenate([xc, cat_int])),
                x_cont, method='trust-constr',
                hess=lambda xc: self.H_reg,
                bounds=[(-1.0, 1.0)] * 18,
                options={'maxiter': 20, 'verbose': 0}
            )
            if res.success:
                cand = np.concatenate([res.x, cat_int])
                f_c = self._evaluate(cand)
                if f_c < self.best_f:
                    self.best_x = cand.copy()
                return res.x
        except Exception:
            pass
        return x_cont

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.func = func
        self.hess_func = hess_func

        n_samples = self.n_pop
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        for i in range(self.n_pop):
            if self.evals >= self.budget: break
            self._evaluate(pop[i])

        it = 0
        while self.evals < self.budget:
            it += 1
            if self.evals >= self.budget: break

            x_best = self.best_x
            if not self.H_valid or it % 10 == 0:
                if self.evals < self.budget:
                    self._update_spectral_info(x_best)

            new_pop = np.zeros_like(pop)
            is_trap = self.H_valid and self.cond > 100.0
            is_basin = self.H_valid and self.cond <= 25.0

            for i in range(self.n_pop):
                if self.evals >= self.budget: break

                r1, r2, r3 = np.random.choice(self.n_pop, 3, replace=False)
                diff = pop[r1, :18] - pop[r2, :18]

                d_cont = diff.copy()
                if is_trap:
                    # Manifold leap: project along steepest curvature eigenvector to escape ravines
                    steep_idx = np.argmax(self.lam_reg)
                    d_cont += self.F * np.random.randn() * self.V[:, steep_idx] * 0.4
                elif self.H_valid:
                    # Curvature-weighted mutation for intermediate regimes
                    d_cont = self.F * self.H_reg @ diff
                else:
                    d_cont = self.F * diff

                v_cont = pop[r3, :18] + d_cont
                v_cat = pop[r3, 18:24].copy()

                # Categorical-Jump Coupling
                if is_trap and np.random.rand() < 0.25:
                    # Bias discrete jumps toward historically successful materials
                    w = self.cat_hist / self.cat_hist.sum()
                    v_cat = np.random.choice(6, p=w)
                elif is_basin:
                    # Lock categories in stable basins to exploit continuous topology
                    v_cat = x_best[18:24].copy()

                cross = np.random.rand(self.dim) < self.cr
                cross[18:24] = False
                trial_cont = np.where(cross[:18], pop[i, :18], v_cont)
                trial_cat = np.where(cross[18:24], pop[i, 18:24].astype(int), v_cat.astype(int))

                trial_cont = np.clip(trial_cont, -1.0, 1.0)
                trial_cat = np.clip(np.round(trial_cat), 0, 5).astype(int)
                trial = np.concatenate([trial_cont, trial_cat])

                f_trial = self._evaluate(trial)
                f_curr = self._evaluate(pop[i])

                if f_trial <= f_curr:
                    new_pop[i] = trial
                    # Trigger refinement only in low-curvature basins
                    if is_basin and np.random.rand() < 0.08:
                        new_pop[i, :18] = self._refine_basin(new_pop[i, :18], new_pop[i, 18:24].astype(int))
                else:
                    new_pop[i] = pop[i]

            pop = new_pop

        return self.best_f, self.best_x