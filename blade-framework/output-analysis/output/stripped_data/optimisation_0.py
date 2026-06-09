import numpy as np
from scipy.optimize import minimize
import cma

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)

    def _evaluate(self, x, func):
        if self.evals >= self.budget: return float('inf')
        eval_x = np.clip(x.copy(), -1.0, 1.0)
        eval_x[18:24] = np.clip(np.round(eval_x[18:24]), 0, 5).astype(int)
        
        f = func(eval_x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = eval_x.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # Initial LHS exploration phase
        n_init = min(10, max(1, self.budget // 15))
        pop = np.random.uniform(-1, 1, size=(n_init, self.dim))
        for x in pop:
            if self.evals >= self.budget: break
            self._evaluate(x, func)

        # Global exploration via CMA-ES on continuous subspace (0:18)
        mean = self.best_x[:18].copy()
        sigma0 = 0.35
        opts = {'bounds': [(-1, 1)] * 18, 'popsize': 10, 'verbose': -1}
        es = cma.CMAEvolutionStrategy(mean, sigma0, opts)

        cat_state = np.random.randint(0, 6, 6)
        gen = 0
        while self.evals < self.budget:
            candidates = es.ask()
            if candidates is None: break

            fitnesses = []
            # Periodic categorical perturbation for diversity
            if gen % 6 == 0:
                cat_state = np.random.randint(0, 6, 6)

            for x_c in candidates:
                x_full = np.concatenate([x_c, cat_state])
                f = self._evaluate(x_full, func)
                fitnesses.append(f)

            es.tell(candidates, fitnesses)

            # Second-Order Newton Correction (Memetic Hybrid)
            if gen % 4 == 0 and hess_func is not None and grad_func is not None:
                if self.evals >= self.budget: break
                grad = grad_func(self.best_x)
                if self.evals >= self.budget: break
                hess = hess_func(self.best_x)
                if self.evals >= self.budget: break

                # Regularize Hessian: absolute eigenvalues to guarantee PD
                eigvals, eigvecs = np.linalg.eigh(hess)
                eigvals = np.abs(eigvals) + 1e-8
                H_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T

                step = -np.linalg.solve(H_reg, grad)
                x_new_c = np.clip(self.best_x[:18] + step, -1, 1)

                # Categorical refinement
                cat_ref = cat_state.copy()
                if np.random.rand() < 0.3:
                    idx = np.random.randint(0, 6)
                    cat_ref[idx] = np.clip(cat_state[idx] + np.random.randint(-1, 2), 0, 5)

                self._evaluate(np.concatenate([x_new_c, cat_ref]), func)

            # Trust-Constr Local Search on best continuous subspace
            if gen % 8 == 0 and self.evals < self.budget - 2:
                if self.evals >= self.budget: break
                res = minimize(
                    lambda xc: func(np.concatenate([xc, self.best_x[18:]])),
                    self.best_x[:18],
                    jac=lambda xc: grad_func(np.concatenate([xc, self.best_x[18:]])) if grad_func else None,
                    hess=lambda xc: hess_func(np.concatenate([xc, self.best_x[18:]])) if hess_func else None,
                    bounds=[(-1, 1)] * 18,
                    method='trust-constr',
                    options={'maxiter': 20, 'verbose': 0}
                )
                if res.success:
                    self._evaluate(np.concatenate([res.x, self.best_x[18:]]), func)

            es.disp = -1
            gen += 1

        return self.best_f, self.best_x
