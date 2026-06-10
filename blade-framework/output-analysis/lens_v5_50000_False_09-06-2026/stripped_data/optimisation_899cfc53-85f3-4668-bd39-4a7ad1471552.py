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
        self.F = 0.75
        self.cr = 0.85
        self.stag_thresh = 6
        self.stag_counter = 0
        self.prev_best_f = float('inf')
        self.H = None
        self.H_inv = None
        self.eigvals = None
        self.eigvecs = None
        self.H_valid = False
        self.hess_func = None
        self._func = None

    def _clip_and_map(self, x):
        xc = np.clip(x, -1.0, 1.0)
        xc[18:24] = np.clip(np.round(xc[18:24]), 0, 5).astype(int)
        return xc

    def _evaluate(self, x, update_best=True):
        if self.evals >= self.budget:
            return float('inf')
        xc = self._clip_and_map(x)
        f = self._func(xc)
        self.evals += 1
        if update_best and f < self.best_f:
            self.best_f = f
            self.best_x = xc.copy()
        return f

    def _update_hessian(self, x):
        try:
            H = self.hess_func(x)
            eigs, evecs = np.linalg.eigh(H)
            eigs_pd = np.abs(eigs)
            eigs_pd[eigs_pd < 1e-3] = 1e-3
            self.H = evecs @ np.diag(eigs_pd) @ evecs.T
            self.H_inv = evecs @ np.diag(1.0 / eigs_pd) @ evecs.T
            self.eigvals = eigs
            self.eigvecs = evecs
            self.H_valid = True
        except Exception:
            self.H_valid = False

    def _detect_stagnation(self):
        if abs(self.prev_best_f - self.best_f) < 1e-4:
            self.stag_counter += 1
        else:
            self.stag_counter = 0
        self.prev_best_f = self.best_f
        return self.stag_counter >= self.stag_thresh

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self._func = func
        self.hess_func = hess_func

        n_samples = self.n_pop
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        pop_f = np.array([self._evaluate(pop[i], False) for i in range(n_samples)])

        it = 0
        while self.evals < self.budget:
            it += 1
            if self.evals >= self.budget: break

            x_best = self.best_x
            if (not self.H_valid) or (it % 7 == 0):
                self._update_hessian(x_best)

            stagnating = self._detect_stagnation()

            new_pop = np.zeros_like(pop)
            for i in range(n_samples):
                r1, r2, r3 = np.random.choice(n_samples, 3, replace=False)
                v_cont = pop[r3, :18].copy()

                if self.H_valid:
                    if stagnating:
                        min_idx = np.argmin(self.eigvals)
                        v_low = self.eigvecs[:, min_idx]
                        curv = np.abs(self.eigvals[min_idx])
                        step_scale = 1.0 / np.sqrt(curv + 1e-6)
                        v_cont += step_scale * v_low * np.random.randn()
                    else:
                        diff = pop[r1, :18] - pop[r2, :18]
                        d = self.H_inv @ diff
                        d = d / (1.0 + np.linalg.norm(d))
                        v_cont += self.F * d

                v_cat = pop[r3, 18:24].copy()
                if self.H_valid and np.random.rand() < 0.12:
                    high_curv_idx = np.argsort(np.abs(self.eigvals))[-3:]
                    if np.sum(high_curv_idx) % 2 == 0:
                        v_cat = np.clip(np.round(v_cat) + np.random.choice([-1, 1]), 0, 5).astype(int)
                    else:
                        v_cat = np.random.randint(0, 6, size=6)

                trial_cont = v_cont.copy()
                cross_mask = np.random.rand(18) < self.cr
                trial_cont[cross_mask] = pop[i, :18][cross_mask]
                trial_cat = v_cat.copy()
                cross_mask_cat = np.random.rand(6) < self.cr
                trial_cat[cross_mask_cat] = pop[i, 18:24][cross_mask_cat].astype(int)

                trial = np.concatenate([trial_cont, trial_cat])
                f_new = self._evaluate(trial, False)

                if f_new <= pop_f[i]:
                    new_pop[i] = trial
                    pop_f[i] = f_new
                else:
                    new_pop[i] = pop[i]

            pop = new_pop

            if self.evals < self.budget and it % 12 == 0 and not stagnating:
                xb = self.best_x[:18].copy()
                cat_int = self.best_x[18:24].copy()
                if self.H_valid:
                    try:
                        res = minimize(
                            lambda xc: self._evaluate(np.concatenate([xc, cat_int]), False),
                            xb, method='trust-constr',
                            hess=lambda xc: self.H,
                            bounds=[(-1.0, 1.0)] * 18,
                            options={'maxiter': 20, 'verbose': 0}
                        )
                        if res.success:
                            cand = np.concatenate([res.x, cat_int])
                            f_c = self._evaluate(cand, True)
                    except Exception:
                        pass

        return self.best_f, self.best_x