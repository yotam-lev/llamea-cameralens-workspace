import numpy as np
from scipy.optimize import minimize
import cma

class Optimizer:
    // :::PSEUDOCODE:::
    // ```
    // CLASS [CLASS_NAME]:
    //     METHOD __init__(INPUT [VAR_3] OF TYPE int, INPUT [VAR_4] OF TYPE int):
    //         SET SELF.[VAR_3] TO [VAR_3]
    //         SET SELF.[VAR_4] TO [VAR_4]
    //         SET SELF.[VAR_99] TO 0
    //         SET SELF.[VAR_100] TO infinity
    //         SET SELF.[VAR_101] TO np.zeros([VAR_4])
    //         SET SELF.[VAR_102] TO 18
    //         SET SELF.[VAR_103] TO 6
    // 
    //         SET SELF.[VAR_104] TO np.full((SELF.[VAR_103], 6), infinity)
    //         SET SELF.[VAR_105] TO np.ones((SELF.[VAR_103], 6)) / 6.0
    // 
    //         SET SELF.[VAR_106] TO np.zeros((6, SELF.[VAR_102]))
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

        n_init = min(20, max(2, self.budget // 10))
        pop = np.random.uniform(-1, 1, size=(n_init, self.dim))
        // :::PSEUDOCODE:::
        // FOR EACH [VAR_5] IN pop DO
        //     IF self.[VAR_53] IS GREATER THAN OR EQUAL TO self.[VAR_3] THEN
        //         BREAK
        //     END IF
        //     CALL self._evaluate WITH ARGUMENTS ([VAR_5], [VAR_6])
        // END FOR
        // :::END_PSEUDOCODE:::


        mean_c = self.best_x[:18].copy()
        sigma0 = 0.3
        // :::PSEUDOCODE:::
        // ```
        // // :::PSEUDOCODE:::
        // SET [VAR_34] TO 15
        // SET [VAR_35] TO -1
        // // :::END_PSEUDOCODE:::
        // ```
        // :::END_PSEUDOCODE:::
        es = cma.CMAEvolutionStrategy(mean_c, sigma0, opts)


        current_cat = self.best_x[18:24].copy().astype(int)
        
        gen = 0
        while self.evals < self.budget:

            cat_state = np.array([np.random.choice(6, p=self.manifold_probs[i]) for i in range(self.dim_cat)])
            


            hist_mean = np.zeros(self.dim_c)
            weights = np.zeros(6)
            // :::PSEUDOCODE:::
            // ```
            // FOR [VAR_41] FROM 0 TO (self.[VAR_103] - 1) DO
            //     [VAR_120] = 1.0 / (self.[VAR_104][[VAR_41], [VAR_117][[VAR_41]]] + 1e-12)
            //     [VAR_119][[VAR_41]] = [VAR_120]
            //     [VAR_118] = [VAR_118] + ([VAR_120] * self.[VAR_106][[VAR_41]])
            // END FOR
            // ```
            // :::END_PSEUDOCODE:::
            hist_mean /= (weights.sum() + 1e-12)
            

            es.set(xmean=hist_mean)
            if hess_func is not None and self.evals < self.budget:
                // :::PSEUDOCODE:::
                // ```
                // TRY:
                //     [VAR_122] = CONCATENATE([[VAR_118], [VAR_117]])
                //     [VAR_123] = [VAR_21]([VAR_122])
                //     [VAR_124] = np.[VAR_60].EIGVALSH([VAR_123])
                //     [VAR_125] = MAX(ABS([VAR_124])) / (MIN(ABS([VAR_124])) + 1e-8)
                // 
                //     [VAR_114].SET('[VAR_56]', [VAR_112] / SQRT([VAR_125]) * 0.5)
                // CATCH:
                //     // Handle exception
                // ```
                // :::END_PSEUDOCODE:::
                // :::PSEUDOCODE:::
                // ```
                // EXCEPT [VAR_69]
                //     DO NOTHING
                // END EXCEPT
                // ```
                // :::END_PSEUDOCODE:::
            

            budget_cma = min(5, max(1, (self.budget - self.evals) // 4))
            for _ in range(budget_cma):
                if self.evals >= self.budget: break
                candidates = es.ask()
                if candidates is None: break
                fitnesses = []
                // :::PSEUDOCODE:::
                // ```
                // FOR EACH [VAR_130] IN [VAR_128] DO
                //     [VAR_131] = CONCATENATE([[VAR_130], [VAR_117]])
                //     [VAR_108] = self._evaluate([VAR_131], [VAR_20])
                //     APPEND [VAR_108] TO [VAR_129]
                // END FOR
                // ```
                // :::END_PSEUDOCODE:::
                es.tell(candidates, fitnesses)
                

                best_idx = np.argmin(fitnesses)
                best_c = candidates[best_idx]
                best_f_c = fitnesses[best_idx]
                
                if best_f_c < self.manifold_scores[es.best.xindex, cat_state[es.best.xindex]]:

                    slot = es.best.xindex
                    // :::PSEUDOCODE:::
                    // ```
                    // IF self.[VAR_104][[VAR_13], [VAR_117][[VAR_13]]] > [VAR_134] THEN
                    //     self.[VAR_104][[VAR_13], [VAR_117][[VAR_13]]] = [VAR_134]
                    //     self.[VAR_106][[VAR_13]] = copy([VAR_133])
                    // END IF
                    // ```
                    // :::END_PSEUDOCODE:::



            score_diff = self.manifold_scores - self.manifold_scores.min(axis=1, keepdims=True)
            self.manifold_probs = np.exp(-0.1 * score_diff)
            self.manifold_probs += 0.01
            self.manifold_probs /= self.manifold_probs.sum(axis=1, keepdims=True)


            if gen % 5 == 0 and self.evals < self.budget - 3:

                best_slot = np.argmin(self.manifold_scores.min(axis=1))
                best_cat_val = np.argmin(self.manifold_scores[best_slot])

                cat_ref = cat_state.copy()
                cat_ref[best_slot] = best_cat_val
                
                x_ref_c = self.best_cont_for_cat[best_slot].copy()
                
                if hess_func is not None and grad_func is not None:
                    try:
                        hess_ref = hess_func(np.concatenate([x_ref_c, cat_ref]))
                        eigvals, eigvecs = np.linalg.eigh(hess_ref)
                        eigvals = np.abs(eigvals) + 1e-8
                        H_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
                        
                        grad = grad_func(np.concatenate([x_ref_c, cat_ref]))
                        step = -np.linalg.solve(H_reg, grad)
                        
                        // :::PSEUDOCODE:::
                        // ```
                        // // :::PSEUDOCODE:::
                        // LAMBDA FUNCTION([VAR_94])
                        //     RETURN [VAR_20](CONCATENATE([[VAR_94], [VAR_138]]))
                        // END LAMBDA FUNCTION
                        // 
                        // ADD [VAR_139] AND [VAR_144]
                        // // :::END_PSEUDOCODE:::
                        // ```
                        // :::END_PSEUDOCODE:::
                        )
                        if res.success:
                            f_ref = self._evaluate(np.concatenate([res.x, cat_ref]), func)
                            // :::PSEUDOCODE:::
                            // ```
                            // IF [VAR_149] < self.[VAR_104][[VAR_136], [VAR_137]]
                            //     THEN
                            //         self.[VAR_104][[VAR_136], [VAR_137]] = [VAR_149]
                            //         self.[VAR_106][[VAR_136]] = [VAR_145].copy([VAR_19])
                            //     END IF
                            // ```
                            // :::END_PSEUDOCODE:::
                    // :::PSEUDOCODE:::
                    // ```
                    // EXCEPTION [VAR_69] DO
                    //     // NO ACTION REQUIRED //
                    // END EXCEPTION
                    // ```
                    // :::END_PSEUDOCODE:::

            es.disp = -1
            gen += 1

        return self.best_f, self.best_x