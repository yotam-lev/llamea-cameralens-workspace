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
        self.dim_c = 18
        self.dim_cat = 6
        
        # Manifold tracking: best loss found for each category value in each slot
        self.manifold_scores = np.full((self.dim_cat, 6), float('inf'))
        self.manifold_probs = np.ones((self.dim_cat, 6)) / 6.0
        
        # Continuous state memory for categorical conditioning
        self.best_cont_for_cat = np.zeros((6, self.dim_c))
        
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
        # LHS Initialization
        n_init = min(20, max(2, self.budget // 10))
        pop = np.random.uniform(-1, 1, size=(n_init, self.dim))
        for x in pop:
            if self.evals >= self.budget: break
            self._evaluate(x, func)

        # Manifold-Conditioned CMA-ES with Score-Driven Categorical Updates
        mean_c = self.best_x[:18].copy()
        sigma0 = 0.3
        opts = {'bounds': (np.full(self.dim_c, -1.0), np.full(self.dim_c, 1.0)), 
                'popsize': 15, 'verbose': -1}
        es = cma.CMAEvolutionStrategy(mean_c, sigma0, opts)

        # Initialize categorical state from best found
        current_cat = self.best_x[18:24].copy().astype(int)
        
        gen = 0
        while self.evals < self.budget:
            # 1. Sample categorical state from manifold probabilities
            cat_state = np.array([np.random.choice(6, p=self.manifold_probs[i]) for i in range(self.dim_cat)])
            
            # 2. Continuous Optimization (CMA) conditioned on current manifold
            # Use history of best continuous points to bias initial mean
            hist_mean = np.zeros(self.dim_c)
            weights = np.zeros(6)
            for i in range(self.dim_cat):
                w = 1.0 / (self.manifold_scores[i, cat_state[i]] + 1e-12)
                weights[i] = w
                hist_mean += w * self.best_cont_for_cat[i]
            hist_mean /= (weights.sum() + 1e-12)
            
            # Set CMA mean and adapt sigma based on Hessian condition (if available)
            es.set(xmean=hist_mean)
            if hess_func is not None and self.evals < self.budget:
                try:
                    x_test = np.concatenate([hist_mean, cat_state])
                    hess = hess_func(x_test)
                    eigvals = np.linalg.eigvalsh(hess)
                    cond = np.max(np.abs(eigvals)) / (np.min(np.abs(eigvals)) + 1e-8)
                    # Scale sigma inversely to condition number: stiff manifolds get tighter search
                    es.set('sigma', sigma0 / np.sqrt(cond) * 0.5)
                except Exception:
                    pass
            
            # CMA Iterations
            budget_cma = min(5, max(1, (self.budget - self.evals) // 4))
            for _ in range(budget_cma):
                if self.evals >= self.budget: break
                candidates = es.ask()
                if candidates is None: break
                fitnesses = []
                for x_c in candidates:
                    x_full = np.concatenate([x_c, cat_state])
                    f = self._evaluate(x_full, func)
                    fitnesses.append(f)
                es.tell(candidates, fitnesses)
                
                # Track best continuous for this categorical slot
                best_idx = np.argmin(fitnesses)
                best_c = candidates[best_idx]
                best_f_c = fitnesses[best_idx]
                
                if best_f_c < self.manifold_scores[es.best.xindex, cat_state[es.best.xindex]]:
                    # Update manifold scores and history
                    slot = es.best.xindex # Use best individual index as proxy for manifold visit
                    if self.manifold_scores[slot, cat_state[slot]] > best_f_c:
                        self.manifold_scores[slot, cat_state[slot]] = best_f_c
                        self.best_cont_for_cat[slot] = best_c.copy()

            # 3. Dynamic Categorical Propagation (Score-Driven)
            # Update probabilities based on manifold scores
            score_diff = self.manifold_scores - self.manifold_scores.min(axis=1, keepdims=True)
            self.manifold_probs = np.exp(-0.1 * score_diff)
            self.manifold_probs += 0.01 # Exploration bias
            self.manifold_probs /= self.manifold_probs.sum(axis=1, keepdims=True)

            # 4. Trust-Region Refinement on Best Manifold
            if gen % 5 == 0 and self.evals < self.budget - 3:
                # Pick slot with lowest manifold score for refinement
                best_slot = np.argmin(self.manifold_scores.min(axis=1))
                best_cat_val = np.argmin(self.manifold_scores[best_slot])
                # Ensure categorical state includes this slot
                cat_ref = cat_state.copy()
                cat_ref[best_slot] = best_cat_val
                
                x_ref_c = self.best_cont_for_cat[best_slot].copy()
                
                if hess_func is not None and grad_func is not None:
                    try:
                        hess_ref = hess_func(np.concatenate([x_ref_c, cat_ref]))
                        eigvals, eigvecs = np.linalg.eigh(hess_ref)
                        eigvals = np.abs(eigvals) + 1e-8
                        H_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
                        
                        grad = grad_func(np.concatenate([x_ref_c, cat_ref]))
                        step = -np.linalg.solve(H_reg, grad)
                        
                        res = minimize(
                            lambda xc: func(np.concatenate([xc, cat_ref])),
                            x_ref_c + step,
                            jac=lambda xc: grad_func(np.concatenate([xc, cat_ref])),
                            bounds=(np.full(self.dim_c, -1.0), np.full(self.dim_c, 1.0)),
                            method='trust-constr',
                            options={'maxiter': 15, 'verbose': 0}
                        )
                        if res.success:
                            f_ref = self._evaluate(np.concatenate([res.x, cat_ref]), func)
                            if f_ref < self.manifold_scores[best_slot, best_cat_val]:
                                self.manifold_scores[best_slot, best_cat_val] = f_ref
                                self.best_cont_for_cat[best_slot] = res.x.copy()
                    except Exception:
                        pass

            es.disp = -1
            gen += 1

        return self.best_f, self.best_x
