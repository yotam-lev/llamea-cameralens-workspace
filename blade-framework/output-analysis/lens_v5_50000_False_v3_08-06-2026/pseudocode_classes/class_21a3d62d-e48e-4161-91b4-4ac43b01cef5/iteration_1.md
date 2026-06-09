import numpy as n
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
        // :::PSEUDOCODE:::
        // ```
        // IF self.[VAR_38] IS GREATER THAN OR EQUAL TO self.[VAR_4]
        //     RETURN INFINITY
        // ```
        // :::END_PSEUDOCODE:::
        eval_x = np.clip(x.copy(), -1.0, 1.0)
        eval_x[18:24] = np.clip(np.round(eval_x[18:24]), 0, 5).astype(int)
        f = func(eval_x)
        self.evals += 1
        // :::PSEUDOCODE:::
        // IF [VAR_57] IS LESS THAN self.[VAR_54] THEN  
        //     self.[VAR_54] ASSIGNED TO [VAR_57]  
        //     self.[VAR_55] ASSIGNED TO COPY OF [VAR_56]  
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
        n_samples = 40
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        fitness = np.empty(n_samples)
        
        // :::PSEUDOCODE:::
        // ```
        // FOR [VAR_13] FROM 0 TO [VAR_46] - 1 DO
        //     IF self.[VAR_38] >= self.[VAR_4] THEN BREAK
        //     [VAR_47][[VAR_13]] = self._evaluate(pop[[VAR_13]], [VAR_7])
        // END FOR
        // ```
        // :::END_PSEUDOCODE:::
            
        F = 0.5
        CR = 0.9
        iter_count = 0
        
        while self.evals < self.budget:
            iter_count += 1
            new_pop = np.empty_like(pop)
            new_fit = np.empty(n_samples)
            
            for i in range(n_samples):
                if self.evals >= self.budget: break
                
                r1, r2, r3 = np.random.choice(n_samples, 3, replace=False)
                while r1 == i: r1 = np.random.randint(n_samples)
                while r2 == i: r2 = np.random.randint(n_samples)
                while r3 == i: r3 = np.random.randint(n_samples)
                
                trial = pop[i].copy()
                

                mut = pop[r1][:18] + F * (pop[r2][:18] - pop[r3][:18])
                trial[:18] += mut
                

                cat_idx = np.random.randint(6)
                trial[18 + cat_idx] = np.random.randint(0, 6)
                

                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, trial, pop[i])
                
                f = self._evaluate(trial, func)
                new_pop[i] = trial
                new_fit[i] = f
                
                // :::PSEUDOCODE:::
                // IF [VAR_42] < [VAR_47][[VAR_13]]
                //     THEN 
                //         [VAR_47][[VAR_13]] = [VAR_42]
                //         pop[[VAR_13]] = [VAR_56]
                // :::END_PSEUDOCODE:::

            pop = new_pop
            fitness = new_fit
            

            if iter_count % 5 == 0:
                elite_indices = np.argsort(fitness)[:5]
                for idx in elite_indices:
                    if self.evals >= self.budget: break
                    x_c = pop[idx][:18].copy()
                    cat_fixed = pop[idx][18:24].copy()
                    
                    // :::PSEUDOCODE:::
                    // ```
                    // FUNCTION [VAR_71]([VAR_63], [VAR_64]=[VAR_62]):
                    //     [VAR_65] = ALLOCATE EMPTY ARRAY OF SIZE self.[VAR_5]
                    //     SET [VAR_65][0 TO 17] TO [VAR_63]
                    //     SET [VAR_65][18 TO 23] TO [VAR_64]
                    //     RETURN RESULT OF CALLING self._evaluate([VAR_65], [VAR_7])
                    // END FUNCTION
                    // ```
                    // :::END_PSEUDOCODE:::
                        
                    // :::PSEUDOCODE:::
                    // ```
                    // FUNCTION [VAR_72]([VAR_63], [VAR_64]=[VAR_62])
                    //     IF [VAR_45] IS NONE THEN
                    //         RETURN np.zeros(18)
                    //     ENDIF
                    // 
                    //     [VAR_65] = np.empty(self.[VAR_5])
                    //     [VAR_65][:18] = [VAR_63]
                    //     [VAR_65][18:24] = [VAR_64]
                    // 
                    //     RETURN [VAR_45]([VAR_65])[:18]
                    // END FUNCTION
                    // ```
                    // :::END_PSEUDOCODE:::
                        
                    // :::PSEUDOCODE:::
                    // ```
                    // FUNCTION [VAR_73]([VAR_63], [VAR_64]=[VAR_62]) RETURNS MATRIX:
                    //     IF [VAR_10] IS NONE THEN
                    //         RETURN IDENTITY_MATRIX(18)
                    //     END IF
                    //     
                    //     [VAR_65] = EMPTY_MATRIX(self.[VAR_5])
                    //     [VAR_65][:18] = [VAR_63]
                    //     [VAR_65][18:24] = [VAR_64]
                    //     
                    //     [VAR_8] = [VAR_10]([VAR_65])
                    //     
                    //     RETURN self._regularize_hessian([VAR_8])
                    // END FUNCTION
                    // ```
                    // :::END_PSEUDOCODE:::

                    bounds = [(-1.0, 1.0) for _ in range(18)]
                    
                    if self.evals >= self.budget: break
                    
                    // :::PSEUDOCODE:::
                    // ```
                    // // No code provided to translate.
                    // ```
                    // :::END_PSEUDOCODE:::
                    
                    if res.success or res.fun < fitness[idx]:
                        final_x = np.empty(self.dim)
                        final_x[:18] = res.x
                        final_x[18:24] = cat_fixed
                        f_refined = self._evaluate(final_x, func)
                        // :::PSEUDOCODE:::
                        // IF [VAR_70] < [VAR_47][[VAR_28]]
                        //     THEN
                        //         [VAR_47][[VAR_28]] = [VAR_70]
                        //         pop[[VAR_28]] = [VAR_69]
                        // END IF
                        // :::END_PSEUDOCODE:::

            F = np.clip(F * 1.05, 0.4, 1.0)
            CR = np.clip(CR + 0.01, 0.5, 0.95)
            
        return self.best_f, self.best_x