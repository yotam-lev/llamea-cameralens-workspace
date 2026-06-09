import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)

    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        eval_x = np.clip(x.copy(), -1.0, 1.0)
        eval_x[18:24] = np.clip(np.round(eval_x[18:24]), 0, 5).astype(int)
        f = func(eval_x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = eval_x.copy()
        return f

    def _regularize_hessian(self, H):
        eigvals = np.linalg.eigvalsh(H)
        shift = max(0.0, -eigvals.min()) + 1e-3
        return H + shift * np.eye(H.shape[0])

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_samples = 30
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        fitness = np.empty(n_samples)

        for i in range(n_samples):
            if self.evals >= self.budget: break
            fitness[i] = self._evaluate(pop[i], func)

        F = 0.85
        CR = 0.9
        iteration = 0

        while self.evals < self.budget:
            iteration += 1
            new_pop = np.empty_like(pop)
            new_fit = np.empty(n_samples)

            # Compute curvature-adaptive scaling matrix from Hessian at best point
            if hess_func is not None and iteration % 3 == 1:
                H = hess_func(self.best_x)
                e, v = np.linalg.eigh(H)
                # Positive-definite regularization via eigenvalue absolute value
                H_reg = v @ np.diag(np.abs(e) + 1e-3) @ v.T
                # Inverse square root for curvature-adaptive scaling
                C_inv_sqrt = v @ np.diag(1.0 / np.sqrt(np.abs(e) + 1e-3)) @ v.T
            else:
                C_inv_sqrt = np.eye(18)

            for i in range(n_samples):
                if self.evals >= self.budget: break

                r1, r2, r3 = np.random.choice(n_samples, 3, replace=False)
                while r1 == i: r1 = np.random.randint(n_samples)
                while r2 == i: r2 = np.random.randint(n_samples)
                while r3 == i: r3 = np.random.randint(n_samples)

                trial = pop[i].copy()

                # Hessian-adaptive continuous mutation
                diff = pop[r1][:18] - pop[r3][:18]
                trial[:18] += F * (C_inv_sqrt @ diff)

                # Curvature-weighted categorical sampling
                best_cat = self.best_x[18:24]
                cat_probs = np.ones(6)
                for c in range(6):
                    dist = np.abs(best_cat[c] - trial[18+c])
                    cat_probs[c] = 1.0 / (dist + 0.5)
                cat_probs /= cat_probs.sum()
                
                # Stochastic categorical swap weighted by curvature
                if np.random.rand() < CR:
                    trial[18 + np.random.randint(6)] = np.random.choice(6, p=cat_probs)

                # Standard crossover
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, trial, pop[i])

                f = self._evaluate(trial, func)
                new_pop[i] = trial
                new_fit[i] = f

                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial

            pop = new_pop
            fitness = new_fit

            # Strictly limited second-order refinement to prevent resource exhaustion
            if iteration % 10 == 0 and self.evals < self.budget:
                elite_idx = np.argmin(fitness)
                x_c = pop[elite_idx][:18].copy()
                cat_fixed = pop[elite_idx][18:24].copy()

                def sub_func(x_val):
                    full = np.empty(self.dim)
                    full[:18] = x_val
                    full[18:24] = cat_fixed
                    return self._evaluate(full, func)

                def sub_grad(x_val):
                    if grad_func is None: return np.zeros(18)
                    full = np.empty(self.dim)
                    full[:18] = x_val
                    full[18:24] = cat_fixed
                    return grad_func(full)[:18]

                def sub_hess(x_val):
                    if hess_func is None: return np.eye(18)
                    full = np.empty(self.dim)
                    full[:18] = x_val
                    full[18:24] = cat_fixed
                    return self._regularize_hessian(hess_func(full))

                res = minimize(sub_func, x_c, jac=sub_grad, hess=sub_hess,
                               method='trust-constr', bounds=[(-1.0, 1.0)]*18,
                               options={'maxiter': 25, 'verbose': 0})

                if self.evals < self.budget:
                    refined_x = np.empty(self.dim)
                    refined_x[:18] = res.x
                    refined_x[18:24] = cat_fixed
                    f_ref = self._evaluate(refined_x, func)
                    if f_ref < fitness[elite_idx]:
                        fitness[elite_idx] = f_ref
                        pop[elite_idx] = refined_x

            F = np.clip(F * 1.02, 0.5, 1.2)
            CR = np.clip(CR + 0.005, 0.6, 0.98)

        return self.best_f, self.best_x