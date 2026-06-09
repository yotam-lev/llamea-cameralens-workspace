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

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # Adaptive parameters
        sigma_init = 0.35
        sigma = sigma_init
        popsize = 10
        min_local_search_freq = 2
        max_local_search_freq = 1
        
        # LHS initialization
        n_init = min(10, max(1, self.budget // 15))
        pop = np.random.uniform(-1, 1, size=(n_init, self.dim))
        for x in pop:
            if self.evals >= self.budget: break
            self._evaluate(x, func)

        mean = self.best_x[:18].copy()
        cat_state = np.random.randint(0, 6, 6)
        
        # Adaptive CMA-ES
        es = cma.CMAEvolutionStrategy(mean, sigma, {'popsize': popsize, 'verbose': -1})
        gen = 0
        stagnation_counter = 0
        best_f_history = []
        
        while self.evals < self.budget:
            # Adaptive population size
            if gen % 5 == 0 and popsize < 20:
                popsize = min(20, popsize + 2)
                es.set_options('popsize', popsize)

            candidates = es.ask()
            if candidates is None: break

            fitnesses = []
            
            # Adaptive categorical mutation probability
            p_cat_mut = max(0.1, 0.5 - (self.evals / self.budget) * 0.3)
            if gen % 4 == 0 or np.random.rand() < p_cat_mut:
                cat_state = np.random.randint(0, 6, 6)

            for x_c in candidates:
                x_full = np.concatenate([x_c, cat_state])
                f = self._evaluate(x_full, func)
                fitnesses.append(f)

            es.tell(candidates, fitnesses)

            # Adaptive improvement tracking
            if self.evals > 10:
                best_f_history.append(self.best_f)
                if len(best_f_history) > 5:
                    best_f_history.pop(0)
                improv_rate = (best_f_history[-2] - best_f_history[-1]) / (abs(best_f_history[-2]) + 1e-12)
                if improv_rate < 1e-4:
                    stagnation_counter += 1
                    sigma *= 0.8
                else:
                    stagnation_counter = 0
                    sigma *= 1.05
                    sigma = np.clip(sigma, sigma_init * 0.1, sigma_init * 5)
            else:
                best_f_history.append(self.best_f)

            # Adaptive local search frequency based on budget remaining
            budget_frac = self.evals / self.budget
            freq = max(min_local_search_freq, int(min_local_search_freq + (max_local_search_freq - min_local_search_freq) * (1 - budget_frac)**2))
            
            # Adaptive Newton Correction
            if gen % freq == 0 and hess_func is not None and grad_func is not None:
                if self.evals >= self.budget: break
                grad = grad_func(self.best_x)
                if self.evals >= self.budget: break
                hess = hess_func(self.best_x)
                if self.evals >= self.budget: break

                # Adaptive regularization based on condition number
                eigvals, eigvecs = np.linalg.eigh(hess)
                max_eig = np.max(np.abs(eigvals))
                min_eig = np.min(np.abs(eigvals))
                cond_num = max_eig / (min_eig + 1e-12)
                reg_lambda = max(1e-6, min_eig * 0.1, 1e-3 * cond_num * max_eig * 1e-4)
                
                eigvals_reg = np.abs(eigvals) + reg_lambda
                H_reg = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T
                step = -np.linalg.solve(H_reg, grad)
                
                # Adaptive step damping based on progress
                step_scale = 1.0 if stagnation_counter > 2 else 0.8
                x_new_c = np.clip(self.best_x[:18] + step * step_scale, -1, 1)

                # Adaptive categorical refinement
                cat_ref = cat_state.copy()
                if np.random.rand() < p_cat_mut:
                    idx = np.random.randint(0, 6)
                    cat_ref[idx] = np.clip(cat_state[idx] + np.random.randint(-1, 2), 0, 5)

                self._evaluate(np.concatenate([x_new_c, cat_ref]), func)

            # Adaptive Trust-Constr Local Search
            if gen % (freq * 2) == 0 and self.evals < self.budget - 5:
                if self.evals >= self.budget: break
                # Adaptive maxiter based on budget
                local_budget = max(10, int(20 * (1 - budget_frac)))
                res = minimize(
                    lambda xc: func(np.concatenate([xc, self.best_x[18:]])),
                    self.best_x[:18],
                    jac=lambda xc: grad_func(np.concatenate([xc, self.best_x[18:]])) if grad_func else None,
                    hess=lambda xc: hess_func(np.concatenate([xc, self.best_x[18:]])) if hess_func else None,
                    bounds=[(-1.0, 1.0)] * 18,
                    method='trust-constr',
                    options={'maxiter': local_budget, 'verbose': 0}
                )
                if res.success:
                    self._evaluate(np.concatenate([res.x, self.best_x[18:]]), func)

            es.disp = -1
            gen += 1

        return self.best_f, self.best_x
