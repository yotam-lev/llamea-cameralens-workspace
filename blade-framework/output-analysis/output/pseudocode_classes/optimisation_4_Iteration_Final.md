```plaintext
class Optimizer:
    function Initialize(budget, dim):
        this.budget = budget
        this.dim = dim
        this.evals = 0
        this.best_f = infinity
        this.best_x = array of zeros with length dim

    function _evaluate(x, func):
        eval_x = CLIP(x, -1.0, 1.0)
        eval_x[18:24] = ROUND(eval_x[18:24]) CLAMPED between 0 and 5
        f = func(eval_x)
        evaluations += 1
        if f < this.best_f:
            this.best_f = f
            this.best_x = COPY(eval_x)
        return f

    function _regularize_hessian(H):
        eigvals = compute_eigenvalues(H)
        shift = MAX(0, -MIN(eigvals) + 1e-4)
        return H + (shift + 1) * IDENTITY_MATRIX(H.shape[0])

    function __call__(func, hess_func=None, grad_func=None):
        es = Initialize CMA-ES with this.dim
        while evaluations < budget:
            candidates = es.ask()
            fit_vals = []
            for c in candidates:
                if evaluations >= budget:
                    fit_vals.append(infinity)
                    break
                fit_vals.append(_evaluate(c, func))
            
            if all(fit_vals == infinity):
                break
            
            es.tell(candidates, fit_vals)
            es.disp()
            
            if hess_func is not None:
                best_c = es.result.xbest
                best_cat = CLAMP(ROUND(best_c[18:24]), 0, 5)
                fb = np.concatenate((best_c[:18], best_cat))
                H = hess_func(fb)
                H_reg = _regularize_hessian(H)
                
                def sub_hess(x):
                    return H_reg
                
                def sub_grad(x):
                    if grad_func is not None:
                        fb_x = np.concatenate((x, best_cat))
                        return grad_func(fb_x)[:18]
                    else:
                        return np.zeros(18)
                
                result = minimize(lambda x: _evaluate(np.concatenate((x, best_cat)), func), 
                                best_c[:18], 
                                method='trust-constr', 
                                jac=sub_grad, 
                                hess=sub_hess,
                                bounds=[(-1.0, 1.0)] * 18,
                                options={'maxiter': 30, 'disp': False})
                
                if result.fun < this.best_f:
                    this.best_f = result.fun
                    this.best_x = np.concatenate((result.x, best_cat))
        
        return this.best_f, this.best_x
```