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
        self.F = 0.7
        self.cr = 0.85
        self.ls_freq = 15
        self.alpha = 0.6  # Coupling weight for discrete transitions
        
        # Hessian/State cache
        self.H_cache = None
        self.H_inv_cache = None
        self.state_cond = np.ones(6) * 5.0
        self.cat_state_cache = None

    def _clip_and_map(self, x):
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        return x

    def _eval(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        xc = self._clip_and_map(x)
        f = func(xc)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = xc.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        pop = self._clip_and_map(pop)
        f_vals = np.zeros(self.n_pop)
        for i in range(self.n_pop):
            if self.evals >= self.budget: break
            f_vals[i] = self._eval(pop[i], func)

        it = 0
        while self.evals < self.budget:
            it += 1
            best_idx = np.argmin(f_vals)
            x_best = pop[best_idx]

            # Update Hessian cache and per-state curvature statistics
            if it % self.ls_freq == 0 and hess_func is not None:
                if self.evals >= self.budget: break
                H = hess_func(x_best)
                eigs = np.linalg.eigvalsh(H)
                min_eig = eigs.min()
                reg = max(1e-2, -min_eig + 1e-5)
                self.H_cache = H + reg * np.eye(18)
                self.H_inv_cache = np.linalg.inv(self.H_cache)
                cond = eigs.max() / max(eigs.min(), 1e-9)
                
                # Update condition number profile based on visited states
                for s in x_best[18:24]:
                    self.state_cond[s] = 0.9 * self.state_cond[s] + 0.1 * cond

            new_pop = np.zeros_like(pop)
            for i in range(self.n_pop):
                if self.evals >= self.budget: break
                r1, r2, r3 = np.random.choice(self.n_pop, 3, replace=False)

                # 1. Continuous Step: Preconditioned by current state's inverse Hessian
                if self.H_inv_cache is not None:
                    diff = pop[r1, :18] - pop[r2, :18]
                    d_cont = self.H_inv_cache @ diff
                else:
                    d_cont = diff
                v_cont = pop[r3, :18] + self.F * d_cont

                # 2. Discrete Step: Curvature-Consistency Manifold Hopping
                v_cat = pop[r3, 18:24].copy()
                if np.random.rand() < 0.15:
                    old_cat = pop[i, 18:24]
                    new_cat = np.random.randint(0, 6, size=6)
                    # Accept if function improves OR if curvature profile becomes more consistent
                    f_gain = f_vals[i] - f_vals[r3]
                    cond_gain = self.alpha * (self.state_cond[old_cat] - self.state_cond[new_cat])
                    if np.random.rand() < np.exp(max(0, f_gain + cond_gain) / 2.0):
                        v_cat = new_cat

                # Crossover & Mapping
                trial_cat = v_cat.copy()
                cross_mask = np.random.rand(self.dim) < self.cr
                cross_mask[18:24] = False
                trial_cont = np.copy(v_cont)
                trial_cont[cross_mask[:18]] = pop[i, :18][cross_mask[:18]]
                trial_cat[cross_mask[18:24]] = pop[i, 18:24][cross_mask[18:24]].astype(int)
                
                trial = np.concatenate([trial_cont, trial_cat])
                if self.evals >= self.budget: break
                
                f_trial = self._eval(trial, func)
                if f_trial < f_vals[i]:
                    new_pop[i] = trial
                    f_vals[i] = f_trial
                else:
                    new_pop[i] = pop[i]

            pop = new_pop

            # 3. Local Refinement: Coupled Trust-Region on best individual's active manifold
            if self.evals < self.budget and it % 10 == 0 and best_idx == np.argmin(f_vals):
                xb = self.best_x[:18].copy()
                cat_int = self.best_x[18:24].copy()
                if self.H_cache is not None:
                    try:
                        res = minimize(
                            lambda xc: self._eval(np.concatenate([xc, cat_int]), func),
                            xb, method='trust-constr',
                            hess=lambda xc: self.H_cache,
                            bounds=[(-1.0, 1.0)] * 18,
                            options={'maxiter': 30, 'verbose': 0}
                        )
                        if res.success:
                            cand = np.concatenate([res.x, cat_int])
                            f_ref = self._eval(cand, func)
                            if f_ref < self.best_f:
                                self.best_x = cand.copy()
                    except Exception:
                        pass

        return self.best_f, self.best_x