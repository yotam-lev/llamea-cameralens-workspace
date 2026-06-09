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
        // IF [VAR_47] >= [VAR_3] THEN  
        //     RETURN [OP_TYPECAST](FLOAT, 'inf')  
        // END IF
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

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):

        sigma_init = 0.35
        sigma = sigma_init
        popsize = 10
        min_local_search_freq = 2
        max_local_search_freq = 1
        

        n_init = min(10, max(1, self.budget // 15))
        pop = np.random.uniform(-1, 1, size=(n_init, self.dim))
        // :::PSEUDOCODE:::
        // FOR EACH [VAR_5] IN pop DO
        //     IF self.[VAR_53] IS GREATER THAN OR EQUAL TO self.[VAR_3] THEN
        //         BREAK
        //     END IF
        //     CALL self._evaluate WITH ARGUMENTS ([VAR_5], [VAR_6])
        // END FOR
        // :::END_PSEUDOCODE:::

        mean = self.best_x[:18].copy()
        cat_state = np.random.randint(0, 6, 6)
        

        es = cma.CMAEvolutionStrategy(mean, sigma, {'popsize': popsize, 'verbose': -1})
        gen = 0
        stagnation_counter = 0
        best_f_history = []
        
        while self.evals < self.budget:

            // :::PSEUDOCODE:::
            // ```
            // IF [VAR_62] % 5 == 0 AND [VAR_55] < 20 THEN
            //     [VAR_55] = MIN(20, [VAR_55] + 2)
            //     [VAR_61].set_options('[VAR_55]', [VAR_55])
            // END IF
            // ```
            // :::END_PSEUDOCODE:::

            candidates = es.ask()
            if candidates is None: break

            fitnesses = []
            

            p_cat_mut = max(0.1, 0.5 - (self.evals / self.budget) * 0.3)
            // :::PSEUDOCODE:::
            // ```
            // IF [VAR_62] MODULUS 4 EQUALS 0 OR np.[VAR_13].RAND() LESS THAN [VAR_67]:
            // ```
            // :::END_PSEUDOCODE:::

            // :::PSEUDOCODE:::
            // ```
            // FOR EACH [VAR_68] IN [VAR_66]
            //     [VAR_69] = CONCATENATE([[VAR_68], [VAR_64]])
            //     [VAR_57] = self._evaluate([VAR_69], [VAR_6])
            //     APPEND [VAR_57] TO [VAR_67]
            // END FOR
            // ```
            // :::END_PSEUDOCODE:::

            es.tell(candidates, fitnesses)


            if self.evals > 10:
                best_f_history.append(self.best_f)
                // :::PSEUDOCODE:::
                // ```
                // IF LENGTH OF [VAR_64] IS GREATER THAN 5 THEN
                //     REMOVE FIRST ELEMENT FROM [VAR_64]
                // END IF
                // ```
                // :::END_PSEUDOCODE:::
                improv_rate = (best_f_history[-2] - best_f_history[-1]) / (abs(best_f_history[-2]) + 1e-12)
                // :::PSEUDOCODE:::
                // ```
                // IF [VAR_70] < 1e-4 THEN
                //     [VAR_63] = [VAR_63] + 1
                //     [VAR_54] = [VAR_54] * 0.8
                // END IF
                // ```
                // :::END_PSEUDOCODE:::
                // :::PSEUDOCODE:::
                // ```
                // ELSE
                //     [VAR_54] = [VAR_54] * 1.05
                //     [VAR_54] = [OP_BOUND]([VAR_54], [VAR_53] * 0.1, [VAR_53] * 5)
                // END ELSE
                // ```
                // :::END_PSEUDOCODE:::
            // :::PSEUDOCODE:::
            // ELSE
            //     APPEND self.[VAR_48] TO [VAR_64]
            // END ELSE
            // :::END_PSEUDOCODE:::


            budget_frac = self.evals / self.budget
            freq = max(min_local_search_freq, int(min_local_search_freq + (max_local_search_freq - min_local_search_freq) * (1 - budget_frac)**2))
            

            if gen % freq == 0 and hess_func is not None and grad_func is not None:
                if self.evals >= self.budget: break
                grad = grad_func(self.best_x)
                if self.evals >= self.budget: break
                hess = hess_func(self.best_x)
                if self.evals >= self.budget: break


                eigvals, eigvecs = np.linalg.eigh(hess)
                max_eig = np.max(np.abs(eigvals))
                min_eig = np.min(np.abs(eigvals))
                cond_num = max_eig / (min_eig + 1e-12)
                reg_lambda = max(1e-6, min_eig * 0.1, 1e-3 * cond_num * max_eig * 1e-4)
                
                eigvals_reg = np.abs(eigvals) + reg_lambda
                H_reg = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T
                step = -np.linalg.solve(H_reg, grad)
                

                step_scale = 1.0 if stagnation_counter > 2 else 0.8
                x_new_c = np.clip(self.best_x[:18] + step * step_scale, -1, 1)


                cat_ref = cat_state.copy()
                // :::PSEUDOCODE:::
                // ```
                // IF np.[VAR_13].rand() < [VAR_67] THEN
                //     [VAR_87] = np.[VAR_13].randint(0, 6)
                //     [VAR_86][[VAR_87]] = [OP_BOUND]([VAR_60][[VAR_87]] + np.[VAR_13].randint(-1, 2), 0, 5)
                // END IF
                // ```
                // :::END_PSEUDOCODE:::

                self._evaluate(np.concatenate([x_new_c, cat_ref]), func)


            if gen % (freq * 2) == 0 and self.evals < self.budget - 5:
                if self.evals >= self.budget: break

                local_budget = max(10, int(20 * (1 - budget_frac)))
                // :::PSEUDOCODE:::
                // // :::PSEUDOCODE:::
                // LAMBDA FUNCTION WITH PARAMETER [VAR_43] DO
                //     RETURN [VAR_6](CONCATENATE([VAR_43], self.[VAR_49][FROM INDEX 18 TO END]))
                // END LAMBDA FUNCTION
                // 
                // RETURN self.[VAR_49][FROM INDEX 0 TO 17]
                // // :::END_PSEUDOCODE:::
                // :::END_PSEUDOCODE:::
                )
                // :::PSEUDOCODE:::
                // ```
                // IF [VAR_79].[VAR_52] THEN
                //     // :::PSEUDOCODE:::
                //     self._evaluate(np.concatenate([[VAR_79].[VAR_5], self.[VAR_55][18:]]), [VAR_6])
                //     // :::END_PSEUDOCODE:::
                // ```
                // :::END_PSEUDOCODE:::

            es.disp = -1
            gen += 1

        return self.best_f, self.best_x