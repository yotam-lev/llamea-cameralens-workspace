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

    // :::PSEUDOCODE:::
    // ```
    // FUNCTION _evaluate(self, [VAR_10], [VAR_11])
    //     IF self.[VAR_38] IS GREATER THAN OR EQUAL TO self.[VAR_4]
    //         RETURN INFINITY
    //     END IF
    // 
    //     [VAR_10] = [OP_BOUND](copy([VAR_10]), INTEGER, INTEGER)
    //     [VAR_10][18:24] = [OP_BOUND](np.round([VAR_10][18:24]), INTEGER, INTEGER).astype(int)
    //     [VAR_51] = [VAR_11]([VAR_10])
    //     self.[VAR_36] += 1
    // 
    //     IF [VAR_46] < self.[VAR_44]
    //         self.[VAR_44] = [VAR_46]
    //         self.[VAR_45] = COPY([VAR_5])
    //     END IF
    // 
    //     RETURN [VAR_51]
    // END FUNCTION
    // ```
    // :::END_PSEUDOCODE:::

    def _adaptive_step(self, x, func, grad_func, hess_func):
        x_c = x[:18]
        cat_ids = x[18:24]
        H_raw = hess_func(np.concatenate([x_c, cat_ids]))
        eigs, Q = np.linalg.eigh(H_raw)
        
        min_eig = eigs[0]
        max_eig = eigs[-1]
        eps = 1e-8
        scale = 0.8
        alpha_saddle = 0.5
        thresh_cond = 50.0
        

        // :::PSEUDOCODE:::
        // IF [VAR_62] < -[VAR_64] THEN
        //     // :::PSEUDOCODE:::
        //     // Existing pseudocode block here
        //     // :::END_PSEUDOCODE:::
        // END IF
        // :::END_PSEUDOCODE:::

        // :::PSEUDOCODE:::
        // ELSE IF [VAR_62] is greater than [VAR_64] AND ([VAR_63] divided by ([VAR_62] plus [VAR_64])) is greater than [VAR_67]:
        //     // :::PSEUDOCODE:::
        //     // Existing pseudocode block would be integrated here
        //     // :::END_PSEUDOCODE:::
        // :::END_PSEUDOCODE:::

        // :::PSEUDOCODE:::
        // ```
        // ELSE
        //     [VAR_73] = [VAR_61] @ np.diag(np.abs([VAR_60]) + 1e-6) @ [VAR_61].T
        //     FUNCTION [VAR_96]([VAR_24])
        //         RETURN [VAR_6](np.concatenate([[VAR_24], [VAR_58]]))
        //     END FUNCTION
        //     
        //     FUNCTION [VAR_94]([VAR_24])
        //         RETURN [VAR_7](np.concatenate([[VAR_24], [VAR_58]]))
        //     END FUNCTION
        //     
        //     FUNCTION [VAR_95]([VAR_24])
        //         RETURN [VAR_73]
        //     END FUNCTION
        // 
        //     ASSIGN [VAR_96] = [VAR_94]
        //     ASSIGN [VAR_57] = [VAR_95]
        //     ASSIGN [VAR_94] = [VAR_95]
        // END ELSE
        // ```
        // :::END_PSEUDOCODE:::
            // :::PSEUDOCODE:::
            // IF [VAR_53] IS LESS THAN [VAR_3] THEN
            // :::END_PSEUDOCODE:::
            // :::PSEUDOCODE:::
            // ```
            // ELSE:
            // ```
            // :::END_PSEUDOCODE:::
            
        return np.concatenate([x_new_c, cat_ids])

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):

        pop = np.random.uniform(-1, 1, size=(25, self.dim))
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

        stagnation_count = 0
        prev_best = self.best_f

        while self.evals < self.budget:
            // :::PSEUDOCODE:::
            // IF self.[VAR_54] EQUALS [VAR_80] THEN
            //     [VAR_79] INCREMENT BY 1
            // END IF
            // :::END_PSEUDOCODE:::
            // :::PSEUDOCODE:::
            // IF condition THEN  
            //     [VAR_80] ← self.[VAR_80]  
            // END IF
            // :::END_PSEUDOCODE:::


            // :::PSEUDOCODE:::
            // IF [VAR_79] IS GREATER THAN 15 THEN  
            //     pop = np.[VAR_21].uniform(-1, 1, size=(25, self.[VAR_4]))  
            //     [VAR_78] = []  
            //     // :::PSEUDOCODE:::  
            //     FOR each [VAR_5] IN pop DO  
            //         IF self.[VAR_53] IS GREATER THAN OR EQUAL TO self.[VAR_3] THEN  
            //             BREAK  
            //         END IF  
            //         [VAR_56] = self._evaluate([VAR_5], [VAR_6])  
            //         APPEND [VAR_56] TO [VAR_78]  
            //     END FOR  
            //     // :::END_PSEUDOCODE:::  
            //     [VAR_78] = np.array([VAR_78])  
            //     [VAR_79] = 0  
            //     CONTINUE  
            // END IF
            // :::END_PSEUDOCODE:::


            worst_idx = np.argmax(pop_f)
            x_worst = pop[worst_idx]
            
            // :::PSEUDOCODE:::
            // ```
            // IF [VAR_8] IS NOT NONE AND [VAR_7] IS NOT NONE THEN
            // 
            //     [VAR_83] = SELF._ADAPTIVE_STEP([VAR_82], [VAR_6], [VAR_7], [VAR_8])
            //     [VAR_84] = SELF._EVALUATE([VAR_83], [VAR_6])
            //     pop[[VAR_81]] = [VAR_83]
            //     [VAR_78][[VAR_81]] = [VAR_84]
            // 
            // END IF
            // ```
            // :::END_PSEUDOCODE:::


            p_idx = np.argsort(pop_f)[:12]
            offspring = []
            // :::PSEUDOCODE:::
            // ```
            // FOR each [VAR_87] in range from 0 to 24 DO
            //     [VAR_88] = np.[VAR_21].choice([VAR_85], 3, replace=False)
            //     [VAR_89] = pop[[VAR_88][0]] + 0.8 * (pop[[VAR_88][1]] - pop[[VAR_88][2]])
            //     [VAR_89][:18] += np.[VAR_21].normal(0, 0.1, 18)
            //     [VAR_89][18:24] = [OP_BOUND](np.round(np.[VAR_21].uniform(0.0, INTEGER, INTEGER)), 0, 5).astype(int)
            //     APPEND [VAR_89] TO [VAR_86]
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
            keep = np.argsort(combined_f)[:25]
            pop = combined[keep]
            pop_f = combined_f[keep]

        return self.best_f, self.best_x