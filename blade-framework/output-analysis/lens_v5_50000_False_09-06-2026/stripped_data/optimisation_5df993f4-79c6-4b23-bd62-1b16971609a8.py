import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.pop_size = 40
        self.stag_thresh = 1e-7
        self.stag_cnt = 0
        self.hess_freq = 15
        self.ls_freq = 10
        self.esc_prob = 0.4
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

    def _get_H_pd(self, x):
        if self.hess_func is None:
            return None, None, None
        try:
            H = self.hess_func(x)
            eigs, vecs = np.linalg.eigh(H)
            # Enforce positive definiteness via absolute eigenvalues + ridge
            H_pd = vecs @ np.diag(np.abs(eigs)) @ vecs.T
            H_pd += 1e-6 * np.eye(18)
            return H_pd, eigs, vecs
        except Exception:
            return None, None, None

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        pop = np.random.uniform(-1, 1, size=(self.pop_size, self.dim))
        
        pbest_x = pop.copy()
        pbest_f = np.full(self.pop_size, np.inf)
        gbest_idx = 0
        gbest_f = float('inf')
        best_hist = [float('inf')]
        cond_hist = []
        
        for i in range(self.pop_size):
            if self.evals >= self.budget: break
            f = self._eval(pop[i], func)
            pbest_f[i] = f
            if f < gbest_f:
                gbest_f = f
                gbest_idx = i
                
        it = 0
        while self.evals < self.budget:
            it += 1
            improved = False
            
            H_pd, eigs, vecs = None, None, None
            if it % self.hess_freq == 0:
                if self.evals >= self.budget: break
                H_pd, eigs, vecs = self._get_H_pd(pop[gbest_idx])
                if eigs is not None:
                    cond_hist.append(eigs[-1] / max(eigs[0], 1e-9))
                    
            if len(best_hist) > 5:
                if best_hist[-1] - best_hist[-5] < self.stag_thresh:
                    self.stag_cnt += 1
                else:
                    self.stag_cnt = 0
            best_hist.append(gbest_f)
            
            # Novel Escape: Spectral Basin-Hopping
            if self.stag_cnt > 8 or (len(cond_hist) > 3 and np.mean(cond_hist[-3:]) > 100):
                self.stag_cnt = 0
                cond_hist.clear()
                for i in range(self.pop_size):
                    if np.random.rand() < self.esc_prob:
                        # Target stiffest eigenvectors to break high-curvature traps
                        stiff_idx = np.argsort(eigs)[-3:]
                        stiff_vecs = vecs[:, stiff_idx]
                        stiff_eigs = eigs[stiff_idx]
                        
                        # Curvature-damped step size prevents overshooting flat regions
                        step = 1.5 / np.sqrt(np.abs(stiff_eigs).max() + 1e-6)
                        coeffs = np.random.randn(3)
                        direction = stiff_vecs @ coeffs
                        direction /= (np.linalg.norm(direction) + 1e-9)
                        
                        pop[i, :18] += step * direction
                        pop[i, :18] = np.clip(pop[i, :18], -1, 1)
                        
                        # Adaptive categorical perturbation when geometry landscape is flat
                        if eigs.min() < 0.01:
                            pop[i, 18:24] = np.random.randint(0, 6, size=6)
                            
                        f = self._eval(pop[i], func)
                        if f < pbest_f[i]:
                            pbest_f[i] = f
                            pbest_x[i] = pop[i].copy()
                            improved = True
                        if f < gbest_f:
                            gbest_f = f
                            gbest_idx = i
                            improved = True
                            
            # Hessian-Preconditioned Evolution
            for i in range(self.pop_size):
                if self.evals >= self.budget: break
                j = (i + 1) % self.pop_size
                diff = pop[j, :18] - pop[i, :18]
                
                if H_pd is not None:
                    try:
                        diff = np.linalg.solve(H_pd, diff)
                    except:
                        pass
                        
                trial = pop[i, :18] + 0.6 * diff
                trial = np.clip(trial, -1, 1)
                
                cand = np.concatenate([trial, pop[i, 18:24]])
                f_t = self._eval(cand, func)
                
                if f_t < pbest_f[i]:
                    pop[i, :18] = trial
                    pbest_f[i] = f_t
                    pbest_x[i] = pop[i].copy()
                    improved = True
                if f_t < gbest_f:
                    gbest_f = f_t
                    gbest_idx = i
                    improved = True
                    
            # Local Search: Hessian-Conditioned Trust Region
            if self.evals < self.budget and (improved or it % self.ls_freq == 0):
                xb = pop[gbest_idx, :18].copy()
                cat = pop[gbest_idx, 18:24].astype(int)
                
                if H_pd is not None:
                    def obj(xc):
                        return func(np.concatenate([xc, cat]))
                        
                    # Dynamic ridge ensures PD during local refinement
                    H_loc = H_pd + 1e-4 * np.eye(18)
                    
                    res = minimize(
                        obj, xb, method='trust-constr',
                        hess=lambda xc: H_loc,
                        bounds=[(-1.0, 1.0)] * 18,
                        options={'maxiter': 30, 'verbose': 0}
                    )
                    
                    if res.success and self.evals < self.budget:
                        cand = np.concatenate([res.x, cat])
                        f_c = self._eval(cand, func)
                        if f_c < gbest_f:
                            gbest_f = f_c
                            pop[gbest_idx] = cand
                            pbest_x[gbest_idx] = cand.copy()
                            pbest_f[gbest_idx] = f_c
                            improved = True
                            H_pd, eigs, vecs = self._get_H_pd(cand)
                            
        return self.best_f, self.best_x