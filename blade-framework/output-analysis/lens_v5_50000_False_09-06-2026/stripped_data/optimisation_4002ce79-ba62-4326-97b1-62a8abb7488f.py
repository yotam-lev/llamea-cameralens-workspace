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
        x = np.clip(x.copy(), -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _levy_distance(self, size):
        u = np.random.standard_normal(size)
        v = np.random.standard_normal(size)
        return np.abs(u / np.power(np.abs(v), 0.5))

    def _hessian_levy_jump(self, x, func, hess_func):
        if hess_func is None:
            return x
        
        full_x = np.concatenate([x[:18], x[18:24]])
        H_raw = hess_func(full_x)
        eigs, Q = np.linalg.eigh(H_raw)
        
        # Anisotropic scaling: stretch steps in flat directions (low curvature)
        # to traverse wide plateaus, dampen in steep directions.
        scale = np.abs(eigs) + 1e-8
        D_inv = np.diag(1.0 / np.sqrt(scale))
        levy_noise = self._levy_distance(18)
        s_levy = Q @ D_inv @ levy_noise
        
        # Deflection along strongest negative curvature to escape saddles
        defl = np.zeros(18)
        idx_neg = np.argmin(eigs)
        if eigs[idx_neg] < -1e-5:
            v_neg = Q[:, idx_neg]
            defl = np.abs(eigs[idx_neg]) * 0.3 * v_neg
        
        # Repulsive force: scale by inverse square root of min eigenvalue (basin width)
        # to push away from current minimum with strength proportional to trap narrowness
        rep = np.zeros(18)
        dist = x[:18] - self.best_x[:18]
        norm_d = np.linalg.norm(dist)
        if norm_d > 1e-6 and eigs.min() > 1e-8:
            R = 1.0 / np.sqrt(eigs.min() + 1e-8)
            rep = (dist / norm_d) * R * 0.4
            
        step = s_levy + defl + rep
        x_new = x.copy()
        x_new[:18] = np.clip(x_new[:18] + 1.8 * step, -1.0, 1.0)
        return x_new

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_samples = 40
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])

        stagnation = 0
        prev_best = self.best_f

        while self.evals < self.budget:
            if self.best_f == prev_best:
                stagnation += 1
            else:
                stagnation = 0
                prev_best = self.best_f

            if stagnation >= 10 and hess_func is not None:
                # Hessian-Guided Levy-Deflation Jump phase
                new_pop = []
                new_f = []
                for x in pop:
                    if self.evals >= self.budget: break
                    x_new = self._hessian_levy_jump(x, func, hess_func)
                    f = self._evaluate(x_new, func)
                    new_pop.append(x_new)
                    new_f.append(f)
                if self.evals >= self.budget: break
                pop = np.array(new_pop)
                pop_f = np.array(new_f)
                stagnation = 0
                continue

            # Phase 2: Levy-Differential Evolution + Hessian Trust Region
            best_idx = np.argmin(pop_f)
            x_best = pop[best_idx]
            x_c, cat = x_best[:18], x_best[18:24]

            if hess_func is not None and self.evals + 50 < self.budget:
                # Periodic Trust Region refinement on best
                full_x = np.concatenate([x_c, cat])
                H_raw = hess_func(full_x)
                eigs, Q = np.linalg.eigh(H_raw)
                H_reg = Q @ np.diag(np.abs(eigs) + 1e-6) @ Q.T
                
                def obj(xc):
                    if self.evals >= self.budget: return float('inf')
                    return func(np.concatenate([xc, cat]))
                def jac(xc):
                    if self.evals >= self.budget: return np.zeros_like(xc)
                    return grad_func(np.concatenate([xc, cat]))
                def hess(xc):
                    return H_reg

                res = minimize(
                    obj, x_c, jac=jac, hess=hess,
                    method='trust-constr',
                    bounds=[(-1.0, 1.0)] * 18,
                    options={'maxiter': 20, 'verbose': 0}
                )
                if self.evals < self.budget:
                    x_refined = np.concatenate([res.x, cat])
                    f_refined = self._evaluate(x_refined, func)
                    pop[best_idx] = x_refined
                    pop_f[best_idx] = f_refined

            # Levy-Differential Evolution for diversity
            p_idx = np.argsort(pop_f)[:20]
            offspring = []
            levy_scale = np.random.uniform(1.5, 2.5)
            for _ in range(n_samples):
                idx_rand = np.random.choice(p_idx, 3, replace=False)
                diff = pop[idx_rand[1]] - pop[idx_rand[2]]
                lev_dist = self._levy_distance(18) * levy_scale
                mu = pop[idx_rand[0]].copy()
                mu[:18] += lev_dist * diff[:18]
                # Categorical mutation with Levy-like probability
                if np.random.rand() < 0.1:
                    mu[18:24] = np.random.randint(0, 6, 6)
                offspring.append(mu)
            
            offspring = np.array(offspring)
            pop_f_off = np.array([self._evaluate(x, func) for x in offspring])
            
            combined = np.vstack([pop, offspring])
            combined_f = np.concatenate([pop_f, pop_f_off])
            keep = np.argsort(combined_f)[:n_samples]
            pop = combined[keep]
            pop_f = combined_f[keep]

        return self.best_f, self.best_x