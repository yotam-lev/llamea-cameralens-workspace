import numpy as np
from scipy.optimize import minimize
import cma

class Optimizer:
    // :::PSEUDOCODE:::
    // ```
    // // Pseudocode for __init__ method
    // 
    // INITIALIZE FUNCTION [METHOD_NAME](SELF, [VAR_3]: INTEGER, [VAR_4]: INTEGER)
    //     SET SELF.[VAR_3] TO [VAR_3]
    //     SET SELF.[VAR_4] TO [VAR_4]
    //     SET SELF.[VAR_53] TO 0
    //     SET SELF.[VAR_54] TO INFINITY
    //     SET SELF.[VAR_55] TO ARRAY OF ZEROS WITH SIZE [VAR_4]
    // END FUNCTION
    // 
    // // END Pseudocode for __init__ method
    // ```
    // :::END_PSEUDOCODE:::
        
    def _evaluate(self, x, func):
        // :::PSEUDOCODE:::
        // ```
        // IF self.[VAR_38] IS GREATER THAN OR EQUAL TO self.[VAR_4]
        //     RETURN INFINITY
        // ```
        // :::END_PSEUDOCODE:::
        x_clip = np.clip(x.copy(), -1.0, 1.0)
        x_clip[18:24] = np.clip(np.round(x_clip[18:24]), 0, 5).astype(int)
        f = func(x_clip)
        self.evals += 1
        // :::PSEUDOCODE:::
        // IF [VAR_27] LESS THAN self.[VAR_24] THEN  
        //     self.[VAR_24] ASSIGN [VAR_27]  
        //     self.[VAR_25] ASSIGN COPY OF [VAR_26]  
        // END IF
        // :::END_PSEUDOCODE:::
        return f

    // :::PSEUDOCODE:::
    // ```
    // FUNCTION _regularize_hessian(VAR_8)
    //     VAR_43 = numpy.linalg.eigvalsh(VAR_8)
    //     VAR_44 = MAX(0.0, -MIN(VAR_43) + 1e-4)
    //     RETURN VAR_8 + (VAR_44 + 1.0) * numpy.eye(SHAPE(VAR_8)[0])
    // END FUNCTION
    // ```
    // :::END_PSEUDOCODE:::

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        dim_cont = 18
        sigma0 = 0.5
        x0 = np.zeros(dim_cont)
        es = cma.CMAEvolutionStrategy(x0, sigma0)
        
        cat_theta = np.ones(6) / 6.0
        mu = max(4, 4 + int(3 + np.log(dim_cont)))
        
        local_search_count = 0
        
        while self.evals < self.budget:
            candidates = es.ask()
            fit_vals = np.empty(len(candidates))
            
            for i, c in enumerate(candidates):
                // :::PSEUDOCODE:::
                // IF self.[VAR_23] >= self.[VAR_3] THEN  
                //     [VAR_39][[VAR_11]] = OP_TYPECAST(float, 'inf')  
                //     BREAK  
                // END IF
                // :::END_PSEUDOCODE:::
                full_x = np.zeros(self.dim)
                full_x[:18] = c
                
                cat_samples = np.random.choice(6, size=6, p=cat_theta)
                full_x[18:24] = cat_samples
                
                f = self._evaluate(full_x, func)
                fit_vals[i] = f
                
                if np.isfinite(f):
                    elite_mask = fit_vals < np.mean(fit_vals)
                    cat_theta = np.zeros(6)
                    // :::PSEUDOCODE:::
                    // ```
                    // IF [VAR_42].any() THEN
                    //     [VAR_43] = [VAR_40][[VAR_42]][:, 18:24]
                    //     [VAR_44] = np.bincount([VAR_43].flatten(), [VAR_15]=6)
                    //     [VAR_35] = np.exp([VAR_44] / (np.max([VAR_44]) + 1e-3))
                    //     [VAR_35] /= [VAR_35].sum()
                    // END IF
                    // ```
                    // :::END_PSEUDOCODE:::
            
            // :::PSEUDOCODE:::
            // IF ALL [VAR_39] EQUAL TO FLOAT(INFINITY) THEN
            //     BREAK
            // END IF
            // :::END_PSEUDOCODE:::
                
            // :::PSEUDOCODE:::
            // ```
            // FOR EACH [VAR_11], [VAR_12] IN ENUMERATE([VAR_38])
            //     IF np.ISFINITE([VAR_39][[VAR_11]])
            //         ADD ([VAR_12]) TO LIST
            // END FOR
            // 
            // [VAR_34].TELL(LIST, [VAR_39][np.ISFINITE([VAR_39])])
            // ```
            // :::END_PSEUDOCODE:::
            
            es.disp()
            
            if es.sigma < 1e-4 and local_search_count < 5:
                best_idx = np.argmin(fit_vals)
                if np.isfinite(fit_vals[best_idx]):
                    best_c = candidates[best_idx]
                    best_cat = full_x[best_idx, 18:24]
                    
                    if hess_func is not None:
                        full_b = np.empty(self.dim)
                        full_b[:18] = best_c
                        full_b[18:24] = best_cat
                        H = hess_func(full_b)
                        H_reg = self._regularize_hessian(H)
                        
                        def sub_hess(x): return H_reg
                        def sub_grad(x):
                            // :::PSEUDOCODE:::
                            // IF [VAR_30] THEN  
                            //     [VAR_50] = np.empty(self.[VAR_4])  
                            //     [VAR_50][:18] = [VAR_5]  
                            //     [VAR_50][18:24] = [VAR_47]  
                            //     RETURN [VAR_30]([VAR_50])[:18]  
                            // END IF
                            // :::END_PSEUDOCODE:::
                            return np.zeros(18)
                            
                        bounds = [(-1.0, 1.0)] * 18
                        // :::PSEUDOCODE:::
                        // ```
                        // SET [VAR_46] TO [VAR_57]
                        // SET [VAR_53] TO [VAR_17]
                        // SET [VAR_17] TO [VAR_58]
                        // ```
                        // :::END_PSEUDOCODE:::
                        
                        if res.fun < fit_vals[best_idx]:
                            final = np.empty(self.dim)
                            final[:18] = res.x
                            final[18:24] = best_cat
                            f_ref = self._evaluate(final, func)
                            // :::PSEUDOCODE:::
                            // IF [VAR_56] < self.[VAR_24] THEN  
                            //     self.[VAR_24] = [VAR_56]  
                            //     self.[VAR_25] = [VAR_55]  
                            // END IF
                            // :::END_PSEUDOCODE:::
                            fit_vals[best_idx] = f_ref
                            es.tell([res.x], [f_ref])
                            local_search_count += 1
            
        return self.best_f, self.best_x