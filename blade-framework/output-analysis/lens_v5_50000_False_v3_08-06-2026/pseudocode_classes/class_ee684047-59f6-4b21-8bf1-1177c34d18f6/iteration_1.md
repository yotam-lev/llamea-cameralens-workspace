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
        if self.evals >= self.budget: return float('inf')
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
        sigma0 = 0.35
        opts = {'bounds': [(-1, 1)] * 18, 'popsize': 10, 'verbose': -1}
        es = cma.CMAEvolutionStrategy(mean, sigma0, opts)

        cat_state = np.random.randint(0, 6, 6)
        gen = 0
        while self.evals < self.budget:
            candidates = es.ask()
            if candidates is None: break

            fitnesses = []

            // :::PSEUDOCODE:::
            // IF [VAR_65] MODULUS 6 EQUALS 0 THEN
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


            if gen % 4 == 0 and hess_func is not None and grad_func is not None:
                if self.evals >= self.budget: break
                grad = grad_func(self.best_x)
                if self.evals >= self.budget: break
                hess = hess_func(self.best_x)
                if self.evals >= self.budget: break


                eigvals, eigvecs = np.linalg.eigh(hess)
                eigvals = np.abs(eigvals) + 1e-8
                H_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T

                step = -np.linalg.solve(H_reg, grad)
                x_new_c = np.clip(self.best_x[:18] + step, -1, 1)


                cat_ref = cat_state.copy()
                // :::PSEUDOCODE:::
                // ```
                // IF np.[VAR_13].rand() < 0.3 THEN
                //     [VAR_78] = np.[VAR_13].randint(0, 6)
                //     [VAR_77][[VAR_78]] = [OP_BOUND]([VAR_64][[VAR_78]] + np.[VAR_13].randint(-1, 2), 0, 5)
                // END IF
                // ```
                // :::END_PSEUDOCODE:::

                self._evaluate(np.concatenate([x_new_c, cat_ref]), func)


            if gen % 8 == 0 and self.evals < self.budget - 2:
                if self.evals >= self.budget: break
                // :::PSEUDOCODE:::
                // ```
                // LAMBDA [VAR_48]:
                //     RETURN [VAR_6](CONCATENATE([VAR_48], SELF.[VAR_55][18:]))
                // RETURN SELF.[VAR_55][:18]
                // ```
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