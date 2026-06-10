import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.n_pop = 48
        self.mu = 16
        self.hess_freq = 12
        self.reg = 1e-2
        self.H = None
        self.H_inv = None
        self.H_valid = False
        self.hess_func = None

    def _clip_and_map(self, x):
        # STRICT BOUNDARY & CATEGORICAL ENFORCEMENT
        x = np.clip(x, -1.0, 1.0)
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
        if self.evals >= self.budget:
            return
        try:
            H = self.hess_func(x)
            eigs = np.linalg.eigvalsh(H)
            shift = max(0, self.reg - eigs.min())
            self.H = H + shift * np.eye(18)
            self.H_inv = np.linalg.inv(self.H)
            self.H_valid = True
        except Exception:
            self.H_valid = False

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        
        # INITIALIZE POPULATION (LHS COMPLIANT)
        pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        
        # INITIAL EVALUATION
        for i in range(self.n_pop):
            if self.evals >= self.budget: break
            self._evaluate(pop[i], func)

        it = 0
        while self.evals < self.budget:
            it += 1
            # EVALUATE & SELECT TOP INDIVIDUALS
            scores = np.array([self._evaluate(pop[i], func) for i in range(self.n_pop)])
            top_idx = np.argsort(scores)[:self.mu]
            top_pop = pop[top_idx].copy()

            # 1. MATERIAL-CONDITIONED GEOMETRY OPTIMIZATION
            for i in range(self.mu):
                if self.evals >= self.budget: break
                xb = top_pop[i, :18].copy()
                mat = top_pop[i, 18:24].copy()
                
                # Update Hessian periodically or when invalid
                if self.hess_func and (it % self.hess_freq == 0 or not self.H_valid):
                    if self.evals < self.budget:
                        self._update_hessian(top_pop[i])

                if self.H_valid:
                    # Conditional trust-region refinement for current material
                    res = minimize(
                        lambda xc, m=mat: func(np.concatenate([xc, m])),
                        xb, method='trust-constr',
                        hess=lambda xc: self.H,
                        bounds=[(-1.0, 1.0)]*18,
                        options={'maxiter': 25, 'verbose': 0}
                    )
                    if res.success and self.evals < self.budget:
                        cand = np.concatenate([res.x, mat])
                        cand = self._clip_and_map(cand)
                        self._evaluate(cand, func)
                        top_pop[i] = cand

            # 2. GEOMETRY-DRIVEN DISCRETE SWITCHING
            for i in range(self.mu):
                if self.evals >= self.budget: break
                xb = top_pop[i, :18].copy()
                curr_mat = top_pop[i, 18:24].copy()
                f_curr = self._evaluate(top_pop[i], func)
                
                best_mat = curr_mat.copy()
                min_f = f_curr
                # Curvature-guided discrete search
                for d in range(6):
                    for delta in [-1, 1]:
                        if self.evals >= self.budget: break
                        trial_mat = curr_mat.copy()
                        trial_mat[d] = np.clip(trial_mat[d] + delta, 0, 5)
                        cand = np.concatenate([xb, trial_mat])
                        f_c = self._evaluate(cand, func)
                        if f_c < min_f:
                            min_f = f_c
                            best_mat = trial_mat
                top_pop[i][18:24] = best_mat
                self._evaluate(top_pop[i], func)

            # 3. HESSIAN-CONDITIONED COUPLED MUTATION
            for i in range(self.n_pop):
                if self.evals >= self.budget: break
                p_idx = np.random.choice(top_idx)
                parent = pop[p_idx]

                # Dynamic coupling factor: inversely proportional to Hessian curvature
                if self.H_valid:
                    curv_proxy = np.linalg.norm(self.H) / (np.linalg.norm(self.H_inv, -1) + 1e-9)
                    alpha = 1.0 / (1.0 + curv_proxy)
                else:
                    alpha = 0.4

                # Continuous step: regularized Newton direction scaled by coupling factor
                if self.H_valid:
                    dx = self.H_inv @ np.random.randn(18)
                else:
                    dx = np.random.randn(18)

                # Discrete step: only aggressive when curvature is low (flat manifolds)
                mut_cat = parent[18:24].copy()
                if np.random.rand() < alpha * 0.18:
                    d = np.random.randint(18, 24)
                    mut_cat[d] = np.clip(mut_cat[d] + np.random.choice([-1, 1]), 0, 5)

                trial_cont = parent[:18] + alpha * dx
                trial_cat = np.clip(np.round(mut_cat), 0, 5).astype(int)
                trial = np.concatenate([trial_cont, trial_cat])
                trial = self._clip_and_map(trial)
                
                if self.evals < self.budget:
                    self._evaluate(trial, func)
                    pop[i] = trial

        return self.best_f, self.best_x