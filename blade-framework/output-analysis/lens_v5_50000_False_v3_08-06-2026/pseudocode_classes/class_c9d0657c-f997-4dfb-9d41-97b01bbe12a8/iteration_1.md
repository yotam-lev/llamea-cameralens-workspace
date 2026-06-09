import numpy as np
from scipy.optimize import minimize

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
        if self.evals >= self.budget: return float('inf')

        x = np.clip(x.copy(), -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        
        f = func(x)
        self.evals += 1
        // :::PSEUDOCODE:::
        // ```
        // IF [VAR_46] < self.[VAR_44] THEN
        //     self.[VAR_44] = [VAR_46]
        //     self.[VAR_45] = COPY([VAR_5])
        // END IF
        // ```
        // :::END_PSEUDOCODE:::
        return f

    // :::PSEUDOCODE:::
    // ```
    // FUNCTION _regularize([VAR_11]):
    // 
    //     [VAR_47], [VAR_48] = np.[VAR_19].eigh([VAR_11])
    //     
    //     RETURN [VAR_48] @ np.diag(np.abs([VAR_47])) @ [VAR_48].T
    // ```
    // :::END_PSEUDOCODE:::

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):

        pop = np.random.uniform(-1, 1, size=(20, self.dim))
        pop_f = []
        // :::PSEUDOCODE:::
        // ```
        // FOR each [VAR_5] in pop DO
        //     IF self.[VAR_43] >= self.[VAR_3] THEN
        //         BREAK
        //     END IF
        //     [VAR_46] = self._evaluate([VAR_5], [VAR_6])
        //     APPEND [VAR_46] TO [VAR_50]
        // END FOR
        // ```
        // :::END_PSEUDOCODE:::
        pop_f = np.array(pop_f)

        while self.evals < self.budget:

            if hess_func is not None and grad_func is not None:
                best_idx = np.argmin(pop_f)
                x_best = pop[best_idx]
                cat_ids = x_best[18:24].copy()
                x_c = x_best[:18].copy()
                
                if self.evals < self.budget:
                    H_raw = hess_func(np.concatenate([x_c, cat_ids]))
                    H_reg = self._regularize(H_raw)
                    
                    def obj(xc): return func(np.concatenate([xc, cat_ids]))
                    def jac(xc): return grad_func(np.concatenate([xc, cat_ids]))
                    def hess(xc): return H_reg
                    
                    if self.evals < self.budget:
                        // :::PSEUDOCODE:::
                        // ```
                        // SET [VAR_72] TO [VAR_72]
                        // SET [VAR_54] TO [VAR_54]
                        // SET [VAR_73] TO [VAR_73]
                        // SET [VAR_74] TO [VAR_74]
                        // ```
                        // :::END_PSEUDOCODE:::
                        )
                        // :::PSEUDOCODE:::
                        // IF self.[VAR_43] < self.[VAR_3] THEN  
                        //     [VAR_61] = CONCATENATE([[[VAR_57].[VAR_5]], [VAR_53]])  
                        //     [VAR_62] = self._evaluate([VAR_61], [VAR_6])  
                        //     pop[[VAR_51]] = [VAR_61]  
                        //     [VAR_50][[VAR_51]] = [VAR_62]  
                        // END IF
                        // :::END_PSEUDOCODE:::


            p_idx = np.argsort(pop_f)[:10]
            parents = pop[p_idx]
            
            offspring = []
            // :::PSEUDOCODE:::
            // ```
            // FOR each [VAR_66] in range from 0 to 19 DO
            //     [VAR_67] = ([VAR_64][0] + [VAR_64][1] + [VAR_64][2] + [VAR_64][3]) / 4.0
            //     [VAR_67][0:18] = [VAR_67][0:18] + np.[VAR_26].normal(0, 0.12, 18)
            //     [VAR_67][18:24] = [OP_BOUND](np.round(np.[VAR_26].uniform(0.0, INTEGER, INTEGER)), 0, 5).astype(int)
            //     APPEND [VAR_67] to [VAR_65]
            // END FOR
            // ```
            // :::END_PSEUDOCODE:::
            offspring = np.array(offspring)
            
            pop_f_off = []
            // :::PSEUDOCODE:::
            // ```
            // FOR each [VAR_5] in [VAR_65]
            //     IF self.[VAR_43] >= self.[VAR_3] THEN
            //         BREAK
            //     END IF
            //     [VAR_46] = self._evaluate([VAR_5], [VAR_6])
            //     APPEND [VAR_46] TO [VAR_68]
            // END FOR
            // ```
            // :::END_PSEUDOCODE:::
            pop_f_off = np.array(pop_f_off)
            
            combined = np.vstack([pop, offspring])
            combined_f = np.concatenate([pop_f, pop_f_off])
            keep = np.argsort(combined_f)[:20]
            pop = combined[keep]
            pop_f = combined_f[keep]

        return self.best_f, self.best_x