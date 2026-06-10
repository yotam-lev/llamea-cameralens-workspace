import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        # Adaptive state
        self.smoothed_improve = 1.0
        self.cond_num = 1.0
        self.curvature_ratio = 1.0
        self.phase = 0  # 0: Explore, 1: Transition, 2: Exploit
        self.phase_hist = []
        
        # Control parameters
        self.base_scale = 0.8
        self.local_search_freq = 5
        self.exploration_ratio = 0.6
        self.barrier_thresh = 500

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

    def _get_hessian_metric(self, hess_func, x):
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        if self.evals >= self.budget: return None, 1.0, 1.0
        
        H_raw = hess_func(x)[:18, :18]
        H_sym = (H_raw + H_raw.T) / 2.0
        eigs, Q = np.linalg.eigh(H_sym)
        min_abs = np.min(np.abs(eigs))
        max_abs = np.max(np.abs(eigs))
        self.cond_num = max_abs / (min_abs + 1e-8)
        self.curvature_ratio = max_abs / (np.mean(np.abs(eigs)) + 1e-8)
        
        # PD regularization via absolute eigenvalues
        H_reg = Q @ np.diag(np.abs(eigs) + 1e-6) @ Q.T
        M_inv = Q @ np.diag(1.0 / np.sqrt(np.abs(eigs) + 1e-6)) @ Q.T
        return H_reg, M_inv, self.cond_num

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        pop_size = int(np.clip(self.budget * 0.05, 30, 100))
        pop = np.random.uniform(-1, 1, size=(pop_size, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])
        
        # Initial curvature reference
        if hess_func is not None:
            center_idx = np.argmin(pop_f)
            self.H_reg, self.M_inv, self.cond_num = self._get_hessian_metric(hess_func, pop[center_idx])

        while self.evals < self.budget:
            # --- Phase Transition & Adaptive Controller ---
            if self.evals % 10 == 0:
                delta = max(1e-8, (pop_f.min() - self.smoothed_improve))
                self.smoothed_improve = 0.9 * self.smoothed_improve + 0.1 * pop_f.min()
                
                stiffness_heavy = self.cond_num > self.barrier_thresh
                stagnation = self.curvature_ratio > 10.0
                progress = (self.smoothed_improve - pop_f.min()) < 1e-4
                
                if stagnation and progress:
                    self.phase = 0
                elif stiffness_heavy:
                    self.phase = 1
                else:
                    self.phase = 2
                    
                # Adaptive Scales via Budget-Aware Sigmoid
                budget_frac = 1.0 - self.evals / self.budget
                self.exploration_ratio = 0.9 / (1.0 + np.exp(5 * (self.phase - 0.5)))
                self.base_scale = 0.3 + 0.5 * self.exploration_ratio * budget_frac
                self.local_search_freq = max(3, int(10 * (1.0 - self.exploration_ratio)))
                
                self.phase_hist.append(self.phase)

            best_idx = np.argmin(pop_f)
            elite = pop[best_idx]
            elite_c = elite[:18]
            elite_cat = elite[18:24]
            
            # --- Hessian Update & Metric Prep ---
            if hess_func is not None and self.evals % 8 == 0:
                probe = elite.copy()
                self.H_reg, self.M_inv, self.cond_num = self._get_hessian_metric(hess_func, probe)
                
            # --- Curvature-Whitened Mutation ---
            n_mut = int(np.ceil(pop_size * self.exploration_ratio))
            noise = np.random.randn(n_mut, 18)
            whitened_noise = (self.M_inv @ noise.T).T if self.M_inv is not None else noise
            
            # Levy scaling for long jumps during exploration
            levy_pow = np.random.pareto(0.8, n_mut)
            levy_scale = levy_pow / np.mean(levy_pow)
            
            scale = np.clip(self.base_scale * (1.0 + self.cond_num / 1000.0), 0.05, 2.0)
            steps = whitened_noise * scale * levy_scale[:, np.newaxis]
            
            new_c = np.clip(elite_c + steps, -1.0, 1.0)
            
            # --- Adaptive Categorical Mutation ---
            new_cat_base = np.clip(np.round(elite_cat), 0, 5).astype(int)
            new_cats = np.tile(new_cat_base, (n_mut, 1))
            cat_mut_prob = 0.1 * self.exploration_ratio + 0.05 * (self.cond_num > self.barrier_thresh)
            flip_mask = np.random.rand(n_mut, 6) < cat_mut_prob
            flips = np.where(flip_mask, np.random.choice([-1, 1], size=(n_mut, 6)), 0)
            new_cats += flips
            new_cats = np.clip(new_cats, 0, 5).astype(int)
            
            # Evaluate mutants
            new_pop_candidates = np.hstack([new_c, new_cats])
            new_f = np.array([self._evaluate(np.hstack([new_c[i], new_cats[i]]), func) for i in range(n_mut)])
            
            # Swarm Update (Tournament/Elitist hybrid)
            for i in range(n_mut):
                cand_x = np.hstack([new_c[i], new_cats[i]])
                cand_f = new_f[i]
                target_idx = np.random.randint(pop_size)
                if cand_f < pop_f[target_idx]:
                    pop[target_idx] = cand_x
                    pop_f[target_idx] = cand_f
                elif cand_f < pop[best_idx][:18].copy(): # Maintain elite consistency
                     pass # Keep elite

            # --- Hessian-Accelerated Trust-Region Exploitation ---
            if self.evals % self.local_search_freq == 0 and hess_func is not None:
                best_idx = np.argmin(pop_f)
                xc = pop[best_idx][:18]
                cat = np.clip(np.round(pop[best_idx][18:24]), 0, 5).astype(int)
                
                # Ensure PD for trust-constr
                H_pd = self.H_reg if self.H_reg is not None else np.eye(18)
                
                def obj(xs):
                    if self.evals >= self.budget: return float('inf')
                    return func(np.concatenate([xs, cat]))
                def jac(xs):
                    if self.evals >= self.budget: return np.zeros(18)
                    if grad_func: return grad_func(np.concatenate([xs, cat]))[:18]
                    return np.zeros(18)
                def hess(xs):
                    return H_pd

                res = minimize(obj, xc, jac=jac, hess=hess, method='trust-constr',
                               bounds=[(-1.0, 1.0)]*18, options={'maxiter': 15, 'verbose': 0})
                
                if self.evals < self.budget:
                    x_ref = np.concatenate([res.x, cat])
                    f_ref = self._evaluate(x_ref, func)
                    if f_ref < pop_f[best_idx]:
                        pop[best_idx] = x_ref
                        pop_f[best_idx] = f_ref
                else:
                    break

        return self.best_f, self.best_x