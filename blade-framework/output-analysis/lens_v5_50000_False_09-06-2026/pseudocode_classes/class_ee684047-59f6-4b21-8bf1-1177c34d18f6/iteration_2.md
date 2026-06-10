import numpy as np
from scipy.optimize import minimize
import cma

class Optimizer:
    // :::PSEUDOCODE:::
    // ```
    // FUNCTION INIT(VAR_3: INTEGER, VAR_4: INTEGER)
    //     SET SELF.VAR_3 TO VAR_3
    //     SET SELF.VAR_4 TO VAR_4
    //     SET SELF.VAR_53 TO 0
    //     SET SELF.VAR_54 TO INFINITY
    //     SET SELF.VAR_55 TO NUMPY.ZEROS(VAR_4)
    // END FUNCTION
    // ```
    // :::END_PSEUDOCODE:::

    // :::PSEUDOCODE:::
    // ```
    // FUNCTION _evaluate(VAR_5, VAR_6)
    //     IF self.VAR_53 >= self.VAR_3 THEN
    //         RETURN INFINITY
    //     END IF
    // 
    //     VAR_56 = OP_BOUND(COPY(VAR_5), INTEGER, INTEGER)
    //     VAR_56[18:24] = OP_BOUND(ROUND(VAR_56[18:24]), INTEGER, INTEGER).TYPECAST(int)
    // 
    //     VAR_57 = VAR_6(VAR_56)
    //     self.VAR_53 += 1
    // 
    //     // :::PSEUDOCODE:::
    //     IF VAR_57 < self.VAR_54 THEN
    //         self.VAR_54 = VAR_57
    //         self.VAR_55 = COPY(VAR_56)
    //     END IF
    //     // :::END_PSEUDOCODE:::
    // 
    //     RETURN VAR_57
    // END FUNCTION
    // ```
    // :::END_PSEUDOCODE:::

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):

        n_init = min(10, max(1, self.budget // 15))
        pop = np.random.uniform(-1, 1, size=(n_init, self.dim))
        // :::PSEUDOCODE:::
        // FOR each [VAR_5] in pop DO
        //     IF self.[VAR_53] is greater than or equal to self.[VAR_3] THEN
        //         BREAK
        //     END IF
        //     self._evaluate([VAR_5], [VAR_6])
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
            // ```
            // IF [VAR_65] MODULO 6 EQUALS 0 THEN
            //     // :::PSEUDOCODE:::
            //     // Existing pseudocode block goes here
            //     // :::END_PSEUDOCODE:::
            // END IF
            // ```
            // :::END_PSEUDOCODE:::

            // :::PSEUDOCODE:::
            // FOR EACH element [VAR_68] IN [VAR_66] DO  
            //     [VAR_69] ← CONCATENATE([[VAR_68], [VAR_64]])  
            //     [VAR_57] ← self._evaluate([VAR_69], [VAR_6])  
            //     APPEND [VAR_57] TO [VAR_67]  
            // END FOR
            // :::END_PSEUDOCODE:::

            es.tell(candidates, fitnesses)


            // :::PSEUDOCODE:::
            // ```
            // IF [VAR_65] % 4 == 0 AND [VAR_7] IS NOT NONE AND [VAR_58] IS NOT NONE THEN
            //     IF self.[VAR_53] >= self.[VAR_3] THEN BREAK
            //     [VAR_70] = [VAR_58](self.[VAR_55])
            //     IF self.[VAR_53] >= self.[VAR_3] THEN BREAK
            //     [VAR_71] = [VAR_7](self.[VAR_55])
            //     IF self.[VAR_53] >= self.[VAR_3] THEN BREAK
            // 
            //     [VAR_72], [VAR_73] = np.[VAR_39].eigh([VAR_71])
            //     [VAR_72] = np.abs([VAR_72]) + 1e-8
            //     [VAR_74] = [VAR_73] @ np.diag([VAR_72]) @ [VAR_73].T
            // 
            //     [VAR_75] = -np.[VAR_39].solve([VAR_74], [VAR_70])
            //     [VAR_76] = [OP_BOUND](self.[VAR_55][:18] + [VAR_75], INTEGER, INTEGER)
            // 
            //     [VAR_77] = copy([VAR_64])
            //     // :::PSEUDOCODE:::
            //     IF [VAR_13].[OP_RAND]() < 0.3 THEN
            //         [VAR_78] = [VAR_13].[OP_RANDINT](0, 6)
            //         [VAR_77][[VAR_78]] = [OP_BOUND]([VAR_64][[VAR_78]] + [VAR_13].[OP_RANDINT](-1, 2), 0, 5)
            //     END IF
            //     // :::END_PSEUDOCODE:::
            // 
            //     self._evaluate(np.concatenate([[VAR_76], [VAR_77]]), [VAR_6])
            // END IF
            // ```
            // :::END_PSEUDOCODE:::


            // :::PSEUDOCODE:::
            // ```
            // IF [VAR_65] MODULO 8 EQUALS 0 AND self.[VAR_53] LESS THAN self.[VAR_3] MINUS 2 THEN
            //     IF self.[VAR_53] GREATER THAN OR EQUAL TO self.[VAR_3] THEN
            //         BREAK
            //     END IF
            //     // :::PSEUDOCODE:::
            //     // LAMBDA FUNCTION [VAR_48]:
            //     //     RETURN [VAR_6](CONCATENATE([[VAR_48], SELF.[VAR_55][INDEX 18 TO END]]))
            //     // SELF.[VAR_55][INDEX 0 TO 17]
            //     // :::END_PSEUDOCODE:::
            // END IF
            // ```
            // :::END_PSEUDOCODE:::
                // :::PSEUDOCODE:::
                // ```
                // IF [VAR_79].[VAR_52] THEN
                //     // :::PSEUDOCODE:::
                //     self._evaluate(np.concatenate([[VAR_79].[VAR_5], self.[VAR_55][18:]]), [VAR_6])
                //     // :::END_PSEUDOCODE:::
                // END IF
                // ```
                // :::END_PSEUDOCODE:::

            es.disp = -1
            gen += 1

        return self.best_f, self.best_x