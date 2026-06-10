import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.f = 0.5
        self.cr = 0.8
        self.n_pop = 50
        self.hess_freq = 12
        self.ls_freq = 15
        self.reg = 1e-3
        self.H = None
        self.H_reg = None
        self.W = None  # Whitening matrix
        self.H_valid = False
        self.hess_func = None
        self.stagnation_counter = 0
        self.prev_best_f = float('inf')
        self.diversity_threshold = 0.25

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

    def _update_hessian(self, x):
        try:
            H = self.hess_func(x)
            eigs = np.linalg.eigvalsh(H)
            shift = max(0, self.reg - eigs.min())
            self.H_reg = H + shift * np.eye(18)
            
            # Whitening: Transform to amplify steps in flat directions
            e_vals, e_vecs = np.linalg.eigh(self.H_reg)
            # Use inverse sqrt to amplify low-curvature directions
            e_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(e_vals, self.reg)))
            self.W = e_vecs @ e_inv_sqrt @ e_vecs.T
            self.H_valid = True
        except Exception:
            self.H_valid = False

    def _check_categorical_escape(self, pop, gbest_idx):
        cats = np.round(pop[:, 18:24]).astype(int)
        unique_cats = np.unique([tuple(row) for row in cats], axis=0)
        diversity = len(unique_cats) / self.n_pop
        if diversity < self.diversity_threshold:
            self.stagnation_counter += 1
            if self.stagnation_counter > 10:
                # Force phase transition on worst half
                worst_idx = np.argsort(np.array([self._evaluate(pop[i], func=func) if i != gbest_idx else self.best_f for i in range(self.n_pop)]))[-(self.n_pop // 2):]
                for idx in worst_idx:
                    pop[idx, 18:24] = np.random.randint(0, 6, 6)
                self.stagnation_counter = 0
        else:
            self.stagnation_counter = 0

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        gbest_idx = np.argmin([self._evaluate(pop[i], func) for i in range(self.n_pop)])
        gbest_f = self.best_f
        
        it = 0
        while self.evals < self.budget:
            it += 1
            if self.hess_func and (it % self.hess_freq == 0 or not self.H_valid):
                if self.evals < self.budget:
                    self._update_hessian(pop[gbest_idx])

            # Check for categorical stagnation and trigger escape
            self._check_categorical_escape(pop, gbest_idx)

            new_pop = np.zeros_like(pop)
            for i in range(self.n_pop):
                if i == gbest_idx:
                    new_pop[i] = pop[i]
                    continue
                    
                r1, r2, r3 = np.random.choice(self.n_pop, 3, replace=False)
                diff_cont = pop[r1, :18] - pop[r2, :18]
                
                if self.W is not None:
                    # Apply whitening to mutation
                    d_cont = self.W @ diff_cont
                else:
                    d_cont = diff_cont

                v_cont = pop[r3, :18] + self.f * d_cont
                
                # Categorical perturbation (Mutation)
                v_cat = np.clip(np.round(pop[r3, 18:24] + np.random.normal(0, 0.4, 6)), 0, 5).astype(int)
                
                # Crossover
                cross_idx = np.random.rand(self.dim) < self.cr
                trial_cont = np.where(cross_idx[:18], v_cont, pop[i, :18])
                # Categorical dims do not crossover in this mixed formulation; 
                # they undergo mutation, so we take v_cat directly or blend via round
                trial_cat = v_cat 

                # Boundary & Mapping
                trial_cont = np.clip(trial_cont, -1.0, 1.0)
                
                trial = np.concatenate([trial_cont, trial_cat])
                
                f_trial = self._evaluate(trial, func)
                f_current = self._evaluate(pop[i], func)
                
                if f_trial <= f_current:
                    new_pop[i] = trial
                    if f_trial < gbest_f:
                        gbest_f = f_trial
                        gbest_idx = i
                else:
                    new_pop[i] = pop[i]

            pop = new_pop

            # Trust-Region Refinement
            if self.evals < self.budget and it % self.ls_freq == 0 and self.H_valid:
                xb = pop[gbest_idx][:18].copy()
                cat_int = pop[gbest_idx][18:24].copy()
                res = minimize(
                    lambda xc: func(np.concatenate([xc, cat_int])),
                    xb, method='trust-constr',
                    hess=lambda xc: self.H_reg,
                    bounds=[(-1.0, 1.0)]*18, 
                    options={'maxiter': 40, 'verbose': 0}
                )
                if res.success and self.evals < self.budget:
                    cand = np.concatenate([res.x, cat_int])
                    f_c = self._evaluate(cand, func)
                    if f_c < gbest_f:
                        gbest_f = f_c
                        gbest_idx = gbest_idx
                        pop[gbest_idx] = cand
            else:
                if self.evals < self.budget and not self.H_valid:
                    self._update_hessian(pop[gbest_idx])
                    
        return self.best_f, self.best_x