import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.hess_func = None
        self.H = None
        self.H_inv = None
        self.eigs = None
        self.vecs = None
        self.H_valid = False
        self.reg = 1e-4
        
        # Adaptive parameters
        self.cat_probs = np.ones(6) / 6.0  # Probability for categorical dims
        self.base_switch_prob = 0.1
        self.trust_refine_interval = 15
        self.trust_counter = 0

    def _clip_and_map(self, x):
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        return x

    def _evaluate(self, x_raw, func):
        if self.evals >= self.budget:
            return float('inf')
        x = self._clip_and_map(x_raw)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _update_hessian(self, x):
        if self.evals >= self.budget: return
        try:
            H_raw = self.hess_func(x)
            eigs, vecs = np.linalg.eigh(H_raw)
            # Regularization: Ensure positive definiteness by clamping eigenvalues
            eigs_reg = np.maximum(np.abs(eigs), self.reg)
            self.H = vecs @ np.diag(eigs_reg) @ vecs.T
            self.H_inv = vecs @ np.diag(1.0 / eigs_reg) @ vecs.T
            self.eigs = eigs_reg
            self.vecs = vecs
            self.H_valid = True
        except Exception:
            self.H_valid = False

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        
        # Population of candidates to maintain diversity
        n_pop = 25
        pop = np.random.uniform(-1, 1, size=(n_pop, self.dim))
        
        # Pre-allocate memory for spectral stats
        grad_norms = np.full(n_pop, np.inf)
        quad_preds = np.full(n_pop, np.inf)
        trap_depths = np.full(n_pop, np.inf)
        
        it = 0
        while self.evals < self.budget:
            it += 1
            
            for i in range(n_pop):
                if self.evals >= self.budget: break
                
                # 1. Adaptive Categorical Mutation based on Spectral Gap
                # Compute curvature ratio: sum of smallest 3 eigs / sum of all eigs
                if self.H_valid and np.sum(self.eigs) > 0:
                    curv_ratio = np.sum(self.eigs[:3]) / np.sum(self.eigs)
                    # High ratio (flat manifold) -> Low switch prob
                    # Low ratio (sharp basin) -> High switch prob (likely trap)
                    switch_p = self.base_switch_prob / (1.0 + 10.0 * curv_ratio)
                else:
                    switch_p = 0.2  # Default exploration
                    
                # Apply mutations
                for d in range(18, 24):
                    if np.random.rand() < switch_p:
                        # Sample category from adaptive distribution
                        cat_val = np.random.choice(6, p=self.cat_probs)
                        pop[i][d] = cat_val
                        
                        # Update categorical probabilities based on recent history
                        # Simple bias: if this category was part of best_x recently, reinforce
                        if self.evals > 10:
                            best_cat = np.clip(np.round(self.best_x[18:24]), 0, 5).astype(int)
                            if cat_val == best_cat[d]:
                                self.cat_probs[cat_val] += 0.05
                            self.cat_probs = np.maximum(self.cat_probs, 0.05)
                            self.cat_probs /= self.cat_probs.sum()

                # 2. Continuous Hessian-Preconditioned Step
                r1, r2 = np.random.rand(2)
                damp = np.sqrt(r1)  # Random damping to prevent overshoot
                
                if grad_func is not None:
                    try:
                        g = grad_func(pop[i])
                        grad_norms[i] = np.linalg.norm(g)
                        
                        if self.H_valid and grad_norms[i] > 1e-6:
                            # Newton step with damping
                            n_step = -self.H_inv @ g
                            step_len = np.linalg.norm(n_step)
                            if step_len > 0:
                                # Normalize and apply random damping
                                move = n_step * (damp * min(0.8, 0.5 / step_len))
                                pop[i][:18] += move
                                
                                # Predict quadratic improvement
                                pred_improve = 0.5 * move @ self.H @ move
                                quad_preds[i] = self.best_f - pred_improve # Approx
                                
                                # Trap Depth Index: Ratio of predicted improvement to gradient norm
                                # High ratio suggests deceptive basin
                                trap_depths[i] = pred_improve / (grad_norms[i] + 1e-12)
                    except Exception:
                        pass

                # 3. Boundary Enforcement
                pop[i] = np.clip(pop[i], -1.0, 1.0)
                pop[i][18:24] = np.clip(np.round(pop[i][18:24]), 0, 5)
                
                # 4. Evaluation
                f = self._evaluate(pop[i], func)
                
                # 5. Hessian Update on high-potential candidates
                # Update Hessian if candidate is promising or random probe
                if (f < self.best_f + 0.1) or np.random.rand() < 0.05:
                    if self.evals < self.budget:
                        self._update_hessian(pop[i])
                        
                # 6. Trust-Region Injection (Local Refinement)
                # Only inject if spectral model predicts a deep basin (low trap_depth)
                if (self.trust_counter % self.trust_refine_interval == 0 and 
                    self.H_valid and 
                    trap_depths[i] < 0.5 and 
                    grad_norms[i] < 1e-4):
                    
                    try:
                        cat_int = np.clip(np.round(pop[i][18:24]), 0, 5).astype(int)
                        
                        # Define local objective
                        def local_obj(xc): return func(np.concatenate([xc, cat_int]))
                        
                        res = minimize(local_obj, pop[i][:18], method='trust-constr',
                                       hess=lambda xc: self.H,
                                       bounds=[(-1.0, 1.0)]*18,
                                       options={'maxiter': 30, 'verbose': 0})
                        
                        if res.success:
                            cand = np.concatenate([res.x, cat_int])
                            f_c = self._evaluate(cand, func)
                            if f_c < f:
                                pop[i] = cand
                                # Update stats for this candidate
                                if grad_func is not None:
                                    g = grad_func(cand)
                                    grad_norms[i] = np.linalg.norm(g)
                                    # Reset trap_depth as we are now in a refined basin
                                    trap_depths[i] = 0.0 
                    except Exception:
                        pass
                        
                self.trust_counter += 1

        return self.best_f, self.best_x