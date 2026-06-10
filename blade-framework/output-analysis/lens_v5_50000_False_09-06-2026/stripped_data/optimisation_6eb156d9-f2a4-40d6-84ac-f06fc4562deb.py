import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)

    def _eval(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_samples = 40
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        pop_f = np.array([self._eval(x, func) for x in pop])

        # Mixed-variable coupling state
        cat_probs = np.ones(6) / 6.0
        cat_perf = np.ones(6) + 1e-8
        cont_mu = np.mean(pop[:, :18], axis=0)
        cont_S = np.eye(18) * 0.5

        while self.evals < self.budget:
            if self.evals >= self.budget: break

            # 1. Sample discrete categories dynamically
            cats = np.random.choice(6, size=n_samples, p=cat_probs)

            # 2. Condition continuous covariance scale on categorical history
            eps = np.exp(-cat_perf)
            S_cond = cont_S * np.outer(eps, eps)
            L = np.linalg.cholesky(S_cond + 1e-6 * np.eye(18))
            z = np.random.randn(n_samples, 18)
            x_cont = cont_mu + z @ L.T
            x_cont *= eps[cats, np.newaxis]
            x_cont += np.random.normal(0, 0.05, (n_samples, 18))

            # 3. Form joint candidates
            x_cand = np.zeros((n_samples, self.dim))
            x_cand[:, :18] = np.clip(x_cont, -1.0, 1.0)
            x_cand[:, 18:24] = cats.astype(int)

            # 4. Evaluate candidates
            cand_f = np.array([self._eval(x_cand[i], func) for i in range(n_samples)])

            # 5. Update categorical performance tracking
            for i in range(n_samples):
                cat_perf[cats[i]] += np.exp(-cand_f[i])

            # 6. Update continuous distribution & covariance via Hessian metric
            best_mask = cand_f < np.percentile(cand_f, 30)
            if np.any(best_mask):
                best_xs = x_cand[best_mask]
                cat_perf_best = cats[best_mask]
                weights = np.exp(-cand_f[best_mask])
                weights /= weights.sum()
                cont_mu = np.mean(weights[:, None] * best_xs[:, :18], axis=0)

                if hess_func is not None and self.evals < self.budget:
                    temp_x = np.concatenate([cont_mu, np.zeros(6)])
                    H_raw = hess_func(temp_x)
                    eigs, Q = np.linalg.eigh(H_raw)
                    H_reg = Q @ np.diag(np.abs(eigs) + 1e-4) @ Q.T
                    cont_S = 0.6 * cont_S + 0.4 * np.linalg.inv(H_reg + np.eye(18)*1e-4)
                else:
                    cov_est = np.cov((best_xs[:, :18] - cont_mu).T)
                    if np.linalg.det(cov_est) > 0:
                        cont_S = 0.8 * cont_S + 0.2 * cov_est

            # 7. Local Trust-Region Exploitation (Conditioned on exact discrete config)
            if self.evals < self.budget and np.random.rand() < 0.15:
                idx_best = np.argmin(cand_f)
                xc, xcat = x_cand[idx_best, :18], x_cand[idx_best, 18:24].astype(int)

                if self.evals >= self.budget: break
                full_ref = np.concatenate([xc, xcat])
                H_raw = hess_func(full_ref)
                eigs, Q = np.linalg.eigh(H_raw)
                H_reg = Q @ np.diag(np.abs(eigs) + 1e-5) @ Q.T

                def obj(xc_s):
                    if self.evals >= self.budget: return float('inf')
                    return func(np.concatenate([xc_s, xcat]))
                def jac(xc_s):
                    if self.evals >= self.budget: return np.zeros(18)
                    if grad_func is not None: return grad_func(np.concatenate([xc_s, xcat]))[:18]
                    return np.zeros(18)
                def hess(xc_s):
                    if self.evals >= self.budget: return H_reg
                    return H_reg

                res = minimize(obj, xc, jac=jac, hess=hess, method='trust-constr',
                               bounds=[(-1.0, 1.0)] * 18, options={'maxiter': 15, 'verbose': 0})

                if self.evals < self.budget:
                    x_ref = np.concatenate([res.x, xcat])
                    f_ref = self._eval(x_ref, func)
                    if f_ref < cand_f[idx_best]:
                        x_cand[idx_best] = x_ref
                        cand_f[idx_best] = f_ref

            # 8. Population Update
            worst_idx = np.argmax(pop_f)
            if self.evals < self.budget:
                pop[worst_idx] = x_cand[np.argmin(cand_f)]
                pop_f[worst_idx] = cand_f[np.argmin(cand_f)]

            # Update categorical probabilities via softmax
            cat_probs = np.exp(cat_perf - np.max(cat_perf))
            cat_probs /= np.sum(cat_probs)

        return self.best_f, self.best_x