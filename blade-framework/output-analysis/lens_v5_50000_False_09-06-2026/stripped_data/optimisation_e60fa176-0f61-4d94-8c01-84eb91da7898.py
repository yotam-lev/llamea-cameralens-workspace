import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        # DE Parameters
        self.pop_size = 24
        self.F = 0.75
        self.cr = 0.85
        self.pop = np.zeros((self.pop_size, self.dim))
        self.pop_f = np.full(self.pop_size, float('inf'))
        self.g_best = np.zeros(dim)
        self.g_best_f = float('inf')
        
        # Simulated Annealing Parameters for Categorical Space
        self.T = 1.5
        self.T_min = 1e-4
        self.decay = 0.985
        
        # Thompson Sampling MAB Parameters
        self.k_ops = 3
        self.alpha_b = np.ones(self.k_ops)
        self.beta_b = np.ones(self.k_ops)
        
        # Hessian State
        self.H_reg = None
        self.eig_vals = None
        self.Q = None

    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        x = np.clip(x.copy(), -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _thompson_sample(self):
        samples = np.random.beta(self.alpha_b, self.beta_b)
        return int(np.argmax(samples))

    def _update_bandit(self, op_idx, reward):
        self.alpha_b[op_idx] += reward
        self.beta_b[op_idx] += 1.0 - reward

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # Initialize with strict LHS syntax
        self.pop = np.random.uniform(-1, 1, size=(self.pop_size, self.dim))
        
        # Initial population evaluation
        for i in range(self.pop_size):
            if self.evals >= self.budget: break
            f = self._evaluate(self.pop[i], func)
            self.pop_f[i] = f
            if f < self.g_best_f:
                self.g_best_f = f
                self.g_best = self.pop[i].copy()

        iter_count = 0
        while self.evals < self.budget:
            iter_count += 1
            
            # Decay temperature for categorical SA operator
            if self.T > self.T_min:
                self.T *= self.decay

            # Select mutation operator via Thompson Sampling
            op = self._thompson_sample()

            # Draw distinct indices for DE difference vector
            r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
            diff = self.pop[r2] - self.pop[r3]
            diff[18:24] = 0.0  # Prevent categorical leakage in continuous DE

            # Periodic Hessian computation & regularization
            if hess_func is not None and (iter_count % 5 == 0 or self.H_reg is None):
                if self.evals < self.budget:
                    H = hess_func(self.g_best)
                    eigs, Q = np.linalg.eigh(H)
                    eps = 1e-4
                    self.eig_vals = np.abs(eigs) + eps
                    self.Q = Q
                    self.H_reg = Q @ np.diag(self.eig_vals) @ Q.T

            # Operator execution
            if op == 0:  # Standard DE
                mutant = self.pop[r1] + self.F * diff
            elif op == 1:  # Hessian-conditioned Natural Gradient DE
                diff_c = diff[:18].copy()
                diff_c = self.Q @ (np.diag(1.0 / np.sqrt(self.eig_vals)) @ (self.Q.T @ diff_c))
                diff[:18] = diff_c
                mutant = self.pop[r1] + self.F * diff
            else:  # SA Categorical Jumper
                mutant = self.g_best.copy()
                for idx in range(18, 24):
                    if np.random.rand() < self.T:
                        old_val = mutant[idx]
                        mutant[idx] = np.random.randint(0, 6)
                        f_new = self._evaluate(mutant, func)
                        if f_new < self.g_best_f or np.random.rand() < np.exp((self.g_best_f - f_new) / self.T):
                            self.g_best_f = f_new
                            self.g_best = mutant.copy()
                            self.best_f = f_new
                            self.best_x = mutant.copy()
                        else:
                            mutant[idx] = old_val
                continue

            # DE Crossover
            j_rand = np.random.randint(self.dim)
            trial = np.where(np.random.rand(self.dim) < self.cr, mutant, self.pop[r1])
            trial[j_rand] = mutant[j_rand]

            # Trial Evaluation
            if self.evals >= self.budget: break
            f_trial = self._evaluate(trial, func)

            # Selection & Bandit Feedback
            success = (f_trial <= self.pop_f[r1])
            if success:
                self.pop_f[r1] = f_trial
                self.pop[r1] = trial.copy()
                if f_trial < self.g_best_f:
                    self.g_best_f = f_trial
                    self.g_best = trial.copy()
                    self.best_f = f_trial
                    self.best_x = trial.copy()
            self._update_bandit(op, float(success))

            # Periodic Local Search on Global Best
            if iter_count % 4 == 0 and hess_func is not None and self.H_reg is not None:
                if self.evals < self.budget:
                    x_c = self.g_best[:18].copy()
                    cat_ids = self.g_best[18:24].copy()
                    def obj(xc): return func(np.concatenate([xc, cat_ids]))
                    
                    if grad_func is not None:
                        def jac(xc): return grad_func(np.concatenate([xc, cat_ids]))[:18]
                        res = minimize(obj, x_c, jac=jac, hess=lambda x: self.H_reg, 
                                       method='trust-constr', bounds=[(-1.0, 1.0)]*18, options={'maxiter': 5})
                    else:
                        res = minimize(obj, x_c, hess=lambda x: self.H_reg, 
                                       method='trust-constr', bounds=[(-1.0, 1.0)]*18, options={'maxiter': 5})
                        
                    if res.success:
                        cand = np.concatenate([res.x, cat_ids])
                        cand_f = self._evaluate(cand, func)
                        if cand_f < self.g_best_f:
                            self.g_best_f = cand_f
                            self.g_best = cand
                            self.best_f = cand_f
                            self.best_x = cand

        return self.best_f, self.best_x