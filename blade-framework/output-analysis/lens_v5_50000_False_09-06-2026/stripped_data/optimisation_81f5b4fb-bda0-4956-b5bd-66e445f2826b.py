import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.stagnation_thresh = 1e-8
        self.stagnation_cnt = 0
        self.hess_freq = 10
        self.ls_freq = 20
        self.reg = 1e-4
        self.hess_func = None

    def _clip_map(self, x):
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        return x

    def _eval(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        xc = self._clip_map(x)
        f = func(xc)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = xc.copy()
        return f

    def _get_H_reg(self, x):
        try:
            H = self.hess_func(x)
            eigs = np.linalg.eigvalsh(H)
            shift = max(0, self.reg - eigs.min())
            return H + shift * np.eye(18), eigs
        except Exception:
            return None, None

    def _categorical_jump(self, pop):
        for i in range(len(pop)):
            if np.random.rand() < 0.5:
                pop[i, 18:24] = np.random.randint(0, 6, size=6)
        return pop

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        n_pop = 30
        pop = np.random.uniform(-1, 1, size=(n_pop, self.dim))
        
        pbest_x = pop.copy()
        pbest_f = np.full(n_pop, np.inf)
        gbest_idx = 0
        gbest_f = float('inf')
        
        best_hist = [float('inf')]
        
        for i in range(n_pop):
            f = self._eval(pop[i], func)
            pbest_f[i] = f
            if f < gbest_f:
                gbest_f = f
                gbest_idx = i
        
        it = 0
        while self.evals < self.budget:
            it += 1
            improved = False
            
            # Hessian update
            H_reg = None
            H_eigs = None
            if self.hess_func and (it % self.hess_freq == 0 or H_reg is None):
                if self.evals < self.budget:
                    H_reg, H_eigs = self._get_H_reg(pop[gbest_idx])
            
            # Stagnation detection
            if len(best_hist) > 5 and best_hist[-1] - best_hist[-5] < self.stagnation_thresh:
                self.stagnation_cnt += 1
            else:
                self.stagnation_cnt = 0
                best_hist.append(gbest_f)
                
            # Escape strategy: Categorical Jump + Hessian-Guided Geometry Expansion
            if self.stagnation_cnt > 10:
                pop = self._categorical_jump(pop)
                
                # Expansion along flat directions
                flat_idx = np.argsort(H_eigs)[:5] # Indices of smallest eigenvalues
                if H_eigs is not None and np.any(H_eigs < 1e-2):
                    for i in range(n_pop):
                        if i == gbest_idx: continue
                        direction = H_eigs[flat_idx].sum() * 0.5
                        # Random combination of flat eigenvectors
                        mask = np.zeros(18)
                        mask[flat_idx] = 1
                        noise = np.random.randn(18) * mask
                        noise = noise / (np.linalg.norm(noise) + 1e-9)
                        pop[i, :18] = pop[gbest_idx, :18] + direction * noise * (np.random.rand() - 0.5)
                        pop[i, :18] = np.clip(pop[i, :18], -1, 1)
                        improved = True

            # Standard evolution step
            w = 0.4 + 0.4 * np.random.rand()
            c1, c2 = 1.6, 1.6
            
            for i in range(n_pop):
                r1, r2 = np.random.rand(), np.random.rand()
                
                diff_g = pop[gbest_idx, :18] - pop[i, :18]
                diff_p = pbest_x[i, :18] - pop[i, :18]
                
                # Hessian preconditioning
                if H_reg is not None:
                    try:
                        H_inv = np.linalg.inv(H_reg)
                        diff_g = H_inv @ diff_g
                    except:
                        pass
                
                vel = w * (pop[i, :18] - pop[i, :18]) + c1 * r1 * diff_p + c2 * r2 * diff_g
                pop[i, :18] += vel
                pop[i, :18] = np.clip(pop[i, :18], -1, 1)
                
                f = self._eval(pop[i], func)
                
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_x[i] = pop[i].copy()
                    improved = True
                    
                if f < gbest_f:
                    gbest_f = f
                    gbest_idx = i
                    improved = True
                    
            if self.evals < self.budget and (improved or it % self.ls_freq == 0):
                xb = pop[gbest_idx, :18].copy()
                cat = pop[gbest_idx, 18:24].astype(int)
                
                # Local refinement with Trust-Constr
                if H_reg is not None:
                    def obj(xc):
                        return func(np.concatenate([xc, cat]))
                    
                    res = minimize(
                        obj, xb, method='trust-constr',
                        hess=lambda xc: H_reg,
                        bounds=[(-1.0, 1.0)] * 18,
                        options={'maxiter': 50, 'verbose': 0}
                    )
                    
                    if res.success and self.evals < self.budget:
                        cand = np.concatenate([res.x, cat])
                        f_c = self._eval(cand, func)
                        if f_c < gbest_f:
                            gbest_f = f_c
                            pop[gbest_idx] = cand
                            pbest_x[gbest_idx] = cand.copy()
                            pbest_f[gbest_idx] = f_c
                            
        return self.best_f, self.best_x