import numpy as np
from scipy.optimize import minimize

class Optimizer:
    // :::PSEUDOCODE:::
    // ```
    // CLASS Constructor([VAR_3]: int, [VAR_4]: int):
    //     SET SELF.[VAR_3] TO [VAR_3]
    //     SET SELF.[VAR_4] TO [VAR_4]
    //     SET SELF.[VAR_114] TO 0
    //     SET SELF.[VAR_115] TO INFINITY
    //     SET SELF.[VAR_116] TO np.zeros([VAR_4])
    //     SET SELF.[VAR_117] TO 60
    //     SET SELF.[VAR_118] TO 0.7
    //     SET SELF.[VAR_119] TO 1.4
    //     SET SELF.[VAR_120] TO 1.4
    //     SET SELF.[VAR_121] TO 0.1
    //     SET SELF.[VAR_122] TO 25
    //     SET SELF.[VAR_123] TO 15
    //     SET SELF.[VAR_124] TO 0.05
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
    // FUNCTION _regularize_hessian([VAR_11])
    //     // :::PSEUDOCODE:::
    //     [VAR_127], [VAR_128] = np.[VAR_12].eigh([VAR_11])
    //     
    //     FOR i FROM 0 TO LENGTH([VAR_127]) - 1 DO
    //         [VAR_127][i] = ABS([VAR_127][i]) + 1e-6
    //     END FOR
    //     
    //     RETURN [VAR_128] @ np.diag([VAR_127]) @ TRANSPOSE([VAR_128])
    //     // :::END_PSEUDOCODE:::
    // END FUNCTION
    // ```
    // :::END_PSEUDOCODE:::

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):

        n_init = min(20, max(1, self.budget // 20))
        X = np.random.uniform(-1, 1, size=(self.swarm_size, self.dim))
        

        V = np.zeros_like(X)
        

        X_cont = X[:, :18].copy()
        X_cat = np.round(np.clip(X[:, 18:24], 0, 5)).astype(int)
        

        pbest_X = X.copy()
        pbest_f = np.full(self.swarm_size, float('inf'))
        

        for i in range(self.swarm_size):
            x_full = np.concatenate([X_cont[i], X_cat[i]])
            f = self._evaluate(x_full, func)
            pbest_f[i] = f
            // :::PSEUDOCODE:::
            // ```
            // IF [VAR_126] < self.[VAR_115] THEN
            //     self.[VAR_115] ← [VAR_126]
            //     self.[VAR_116] ← [VAR_137]
            // END IF
            // ```
            // :::END_PSEUDOCODE:::

        gen = 0
        hess_counter = 0
        lso_counter = 0

        while self.evals < self.budget:

            // :::PSEUDOCODE:::
            // IF self.[VAR_114] >= self.[VAR_3]
            //     THEN BREAK
            // :::END_PSEUDOCODE:::




            grad_step = np.zeros(18)
            if hess_func is not None and hess_counter % self.hess_freq == 0:
                hess_counter = 0

                if self.evals >= self.budget: break
                

                full_best = np.concatenate([self.best_x[:18], self.best_x[18:24]])
                hess = hess_func(full_best)
                
                // :::PSEUDOCODE:::
                // ```
                // IF [VAR_129] IS NOT NONE THEN
                //     IF self.[VAR_114] >= self.[VAR_3] THEN
                //         BREAK
                //     END IF
                // END IF
                // ```
                // :::END_PSEUDOCODE:::
                // :::PSEUDOCODE:::
                // ```
                // ELSE:
                // ```
                // :::END_PSEUDOCODE:::
            hess_counter += 1

            gbest_cont = self.best_x[:18].copy()

            for i in range(self.swarm_size):

                r1 = np.random.rand(18)
                r2 = np.random.rand(18)
                
                // :::PSEUDOCODE:::
                // ```
                // // :::PSEUDOCODE:::
                // [self.[VAR_119] * [VAR_147] * ([VAR_135][[VAR_37], :18] - [VAR_133][[VAR_37]]) + 
                //  self.[VAR_120] * [VAR_148] * ([VAR_146] - [VAR_133][[VAR_37]]))]
                // // :::END_PSEUDOCODE:::
                // ```
                // :::END_PSEUDOCODE:::
                

                V[i] = v_update + self.lambda_h * grad_step
                X_cont[i] += V[i]


            X_cont = np.clip(X_cont, -1.0, 1.0)



            for i in range(self.swarm_size):
                for j in range(6):
                    // :::PSEUDOCODE:::
                    // ```
                    // IF np.[VAR_23].rand() < self.[VAR_124] THEN
                    // 
                    //     [VAR_134][[VAR_37], [VAR_150]] = np.[VAR_23].randint(0, 6)
                    // 
                    // END IF
                    // ```
                    // :::END_PSEUDOCODE:::
                    // :::PSEUDOCODE:::
                    // IF np.[VAR_23].rand() < 0.02 THEN  
                    //     [VAR_54] = np.[VAR_23].choice([-1, 1])  
                    //     [VAR_134][[VAR_37], [VAR_150]] = [OP_BOUND]([VAR_134][[VAR_37], [VAR_150]] + [VAR_54], 0, 5)  
                    // ENDIF
                    // :::END_PSEUDOCODE:::


            for i in range(self.swarm_size):
                x_full = np.concatenate([X_cont[i], X_cat[i]])
                f = self._evaluate(x_full, func)
                

                // :::PSEUDOCODE:::
                // ```
                // IF [VAR_126] < [VAR_136][[VAR_37]]
                //     THEN [VAR_136][[VAR_37]] = [VAR_126]
                //          [VAR_135][[VAR_37]] = [VAR_137]
                // ```
                // :::END_PSEUDOCODE:::


                // :::PSEUDOCODE:::
                // ```
                // IF [VAR_126] < self.[VAR_115]
                //     THEN
                //         self.[VAR_115] = [VAR_126]
                //         self.[VAR_116] = [VAR_137]
                //         [VAR_146] = COPY([VAR_137][:18])
                // 
                //         [VAR_132][[VAR_37]] += self.[VAR_120] * ([VAR_146] - [VAR_133][[VAR_37]])
                // ```
                // :::END_PSEUDOCODE:::


            lso_counter += 1
            if lso_counter % self.lso_freq == 0 and self.evals < self.budget - 10:
                if self.evals >= self.budget: break
                

                // :::PSEUDOCODE:::
                // ```
                // LAMBDA [VAR_107]:
                //     RETURN [VAR_6]([
                //         CONCATENATE(
                //             [[VAR_107]],
                //             self.[VAR_116][18:]
                //         )
                //     ])
                // END LAMBDA
                // 
                // RETURN self.[VAR_116][:18]
                // ```
                // :::END_PSEUDOCODE:::
                )
                // :::PSEUDOCODE:::
                // ```
                // IF [VAR_151].[VAR_112] AND [VAR_151].[VAR_113] < self.[VAR_115]
                //     THEN
                //         self._evaluate(np.concatenate([[VAR_151].[VAR_5], self.[VAR_116][18:]]), [VAR_6])
                // ```
                // :::END_PSEUDOCODE:::

            gen += 1

        return self.best_f, self.best_x