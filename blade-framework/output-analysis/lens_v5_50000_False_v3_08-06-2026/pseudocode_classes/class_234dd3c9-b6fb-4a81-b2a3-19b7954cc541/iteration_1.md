import os
import numpy as np
from scipy.optimize import minimize


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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
                

                cat_diversity = np.std(pop[:, 18:24])
                mut_scale = 1.0 + 0.2 * cat_diversity
                mut = pop[r1][:18] + F * mut_scale * (pop[r2][:18] - pop[r3][:18])
                trial[:18] += mut
                

                cat_idx = np.random.randint(6)
                trial[18 + cat_idx] = np.clip(trial[18 + cat_idx] + np.random.choice([-1, 1]), 0, 5)
                

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
            

            if iter_count % 4 == 0:
                elite_indices = np.argsort(fitness)[:4]
                for idx in elite_indices:
                    if self.evals >= self.budget: break
                    

                    x_c = pop[idx][:18].copy()
                    cat_fixed = pop[idx][18:24].copy()
                    
                    def make_sub_funcs(c_fixed):
                        // :::PSEUDOCODE:::
                        // ```
                        // FUNCTION [VAR_124]([VAR_46]):
                        //     [VAR_107] = ALLOCATE ARRAY OF SIZE (self.[VAR_14])
                        //     [VAR_107][0 TO 17] = [VAR_46]
                        //     [VAR_107][18 TO 23] = [VAR_45]
                        //     RETURN self._evaluate([VAR_107], [VAR_16])
                        // END FUNCTION
                        // ```
                        // :::END_PSEUDOCODE:::
                        // :::PSEUDOCODE:::
                        // ```
                        // FUNCTION [VAR_125]([VAR_46])
                        //     IF [VAR_87] IS EQUAL TO None THEN
                        //         RETURN np.zeros(18)
                        //     ENDIF
                        // 
                        //     [VAR_107] = np.empty(self.[VAR_14])
                        //     [VAR_107][0:18] = [VAR_46]
                        //     [VAR_107][18:24] = [VAR_45]
                        // 
                        //     RETURN [VAR_87]([VAR_107])[0:18]
                        // END FUNCTION
                        // ```
                        // :::END_PSEUDOCODE:::
                        // :::PSEUDOCODE:::
                        // ```
                        // FUNCTION [VAR_126]([VAR_46])
                        //     IF [VAR_24] IS EQUAL TO None THEN
                        //         RETURN np.eye(18)
                        //     END IF
                        // 
                        //     [VAR_107] = np.empty(self.[VAR_14])
                        //     [VAR_107][FROM 0 TO 17] = [VAR_46]
                        //     [VAR_107][FROM 18 TO 23] = [VAR_45]
                        //     
                        //     RETURN self._regularize_hessian([VAR_24]([VAR_107]))
                        // END FUNCTION
                        // ```
                        // :::END_PSEUDOCODE:::
                        return sub_func, sub_grad, sub_hess

                    if self.evals >= self.budget: break
                    sub_func, sub_grad, sub_hess = make_sub_funcs(cat_fixed)
                    bounds = [(-1.0, 1.0) for _ in range(18)]
                    
                    // :::PSEUDOCODE:::
                    // ```
                    // BEGIN PSEUDOCODE
                    // 
                    // END PSEUDOCODE
                    // ```
                    // :::END_PSEUDOCODE:::
                    
                    if res.success or res.fun < fitness[idx]:
                        x_c_opt = res.x
                        f_cont = res.fun
                        

                        best_cat = cat_fixed.copy()
                        f_best_cat = f_cont
                        

                        for c_dim in range(6):
                            for delta in [-1, 1]:
                                trial_cat = cat_fixed.copy()
                                trial_cat[c_dim] = np.clip(trial_cat[c_dim] + delta, 0, 5)
                                

                                // :::PSEUDOCODE:::
                                // ```
                                // FUNCTION [VAR_123](INPUT [VAR_46], INPUT [VAR_118] DEFAULT [VAR_117])
                                //     DECLARE [VAR_107] AS ARRAY OF LENGTH self.[VAR_14]
                                //     
                                //     SET [VAR_107][0:18] TO [VAR_46]
                                //     SET [VAR_107][18:24] TO [VAR_118]
                                //     
                                //     RETURN self._evaluate([VAR_107], [VAR_16])
                                // END FUNCTION
                                // ```
                                // :::END_PSEUDOCODE:::
                                
                                // :::PSEUDOCODE:::
                                // ```
                                // ASSIGN [VAR_108] TO [VAR_108]
                                // ASSIGN {'[VAR_52]': 5, '[VAR_53]': 10} TO [VAR_111]
                                // ```
                                // :::END_PSEUDOCODE:::
                                f_swapped = res_cat.fun
                                
                                // :::PSEUDOCODE:::
                                // IF [VAR_120] IS LESS THAN [VAR_115] THEN
                                // :::END_PSEUDOCODE:::
                                    
                        final_x = np.empty(self.dim)
                        final_x[:18] = x_c_opt
                        final_x[18:24] = best_cat
                        f_refined = self._evaluate(final_x, func)
                        
                        // :::PSEUDOCODE:::
                        // IF [VAR_70] < [VAR_47][[VAR_28]]
                        //     THEN
                        //         [VAR_47][[VAR_28]] = [VAR_70]
                        //         pop[[VAR_28]] = [VAR_69]
                        // END IF
                        // :::END_PSEUDOCODE:::

            F = np.clip(F * 1.02, 0.4, 1.0)
            CR = np.clip(CR + 0.005, 0.5, 0.95)
            
        return self.best_f, self.best_x