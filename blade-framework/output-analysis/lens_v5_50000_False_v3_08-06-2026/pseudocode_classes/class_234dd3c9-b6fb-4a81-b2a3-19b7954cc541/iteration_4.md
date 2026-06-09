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

    // :::PSEUDOCODE:::
    // ```
    // FUNCTION _evaluate([VAR_15], [VAR_16])
    //     // :::PSEUDOCODE:::
    //     // ```
    //     // IF self.[VAR_38] IS GREATER THAN OR EQUAL TO self.[VAR_4]
    //     //     RETURN INFINITY
    //     // ```
    //     // :::END_PSEUDOCODE:::
    // 
    //     [VAR_83] = [OP_BOUND](COPY([VAR_15]), INTEGER, INTEGER)
    // 
    //     [VAR_83][18:24] = [OP_BOUND](NP.ROUND([VAR_83][18:24]), INTEGER, INTEGER).ASTYPE(INTEGER)
    //     [VAR_84] = [VAR_16]([VAR_83])
    //     self.[VAR_80] += 1
    //     // :::PSEUDOCODE:::
    //     // IF [VAR_57] IS LESS THAN self.[VAR_54] THEN  
    //     //     self.[VAR_54] ASSIGNED TO [VAR_57]  
    //     //     self.[VAR_55] ASSIGNED TO COPY OF [VAR_56]  
    //     // END IF
    //     // :::END_PSEUDOCODE:::
    //     RETURN [VAR_84]
    // END FUNCTION
    // ```
    // :::END_PSEUDOCODE:::

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
            
            // :::PSEUDOCODE:::
            // ```
            // FOR [VAR_27] FROM 0 TO [VAR_88] - 1 DO
            //     IF self.[VAR_80] >= self.[VAR_13] THEN
            //         BREAK
            //     END IF
            //     
            //     [VAR_95], [VAR_96], [VAR_97] = np.[VAR_26].choice([VAR_88], 3, [VAR_28]=False)
            //     WHILE [VAR_95] == [VAR_27] DO
            //         [VAR_95] = np.[VAR_26].randint([VAR_88])
            //     END WHILE
            //     WHILE [VAR_96] == [VAR_27] DO
            //         [VAR_96] = np.[VAR_26].randint([VAR_88])
            //     END WHILE
            //     WHILE [VAR_97] == [VAR_27] DO
            //         [VAR_97] = np.[VAR_26].randint([VAR_88])
            //     END WHILE
            //     
            //     [VAR_98] = COPY(pop[[VAR_27]])
            //     
            //     [VAR_99] = np.std(pop[:, 18:24])
            //     [VAR_100] = 1.0 + 0.2 * [VAR_99]
            //     [VAR_101] = pop[[VAR_95]][:18] + [VAR_90] * [VAR_100] * (pop[[VAR_96]][:18] - pop[[VAR_97]][:18])
            //     [VAR_98][:18] += [VAR_101]
            //     
            //     [VAR_102] = np.[VAR_26].randint(6)
            //     [VAR_98][18 + [VAR_102]] = [OP_BOUND]([VAR_98][18 + [VAR_102]] + np.[VAR_26].choice([-1, 1]), 0, 5)
            //     
            //     [VAR_103] = np.[VAR_26].rand(self.[VAR_14]) < [VAR_91]
            //     [VAR_98] = np.where([VAR_103], [VAR_98], pop[[VAR_27]])
            //     
            //     [VAR_84] = self._evaluate([VAR_98], [VAR_16])
            //     [VAR_93][[VAR_27]] = [VAR_98]
            //     [VAR_94][[VAR_27]] = [VAR_84]
            // END FOR
            // ```
            // :::END_PSEUDOCODE:::
                
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
                // :::PSEUDOCODE:::
                // ```
                // FOR EACH [VAR_44] IN [VAR_104]
                //     IF self.[VAR_80] IS GREATER THAN OR EQUAL TO self.[VAR_13] THEN
                //         BREAK
                //     ENDIF
                // 
                //     [VAR_106] = pop[[VAR_44]][FROM 18 TO 23].COPY()
                // 
                //     // :::PSEUDOCODE:::
                //     // FUNCTION make_sub_funcs([VAR_45]):
                //     //     // :::PSEUDOCODE:::
                //     //     // FUNCTION [VAR_124]([VAR_46]):
                //     //     //     [VAR_107] = ALLOCATE ARRAY OF SIZE (self.[VAR_14])
                //     //     //     [VAR_107][FROM 0 TO 17] = [VAR_46]
                //     //     //     [VAR_107][FROM 18 TO 23] = [VAR_45]
                //     //     //     RETURN self._evaluate([VAR_107], [VAR_16])
                //     //     // END FUNCTION
                //     //     // :::END_PSEUDOCODE:::
                //     //     
                //     //     // :::PSEUDOCODE:::
                //     //     // FUNCTION [VAR_125]([VAR_46]):
                //     //     //     IF [VAR_87] IS EQUAL TO None THEN
                //     //     //         RETURN np.zeros(18)
                //     //     //     ENDIF
                //     //     // 
                //     //     //     [VAR_107] = np.empty(self.[VAR_14])
                //     //     //     [VAR_107][FROM 0 TO 17] = [VAR_46]
                //     //     //     [VAR_107][FROM 18 TO 23] = [VAR_45]
                //     //     // 
                //     //     //     RETURN [VAR_87]([VAR_107])[FROM 0 TO 17]
                //     //     // END FUNCTION
                //     //     // :::END_PSEUDOCODE:::
                //     //     
                //     //     // :::PSEUDOCODE:::
                //     //     // FUNCTION [VAR_126]([VAR_46]):
                //     //     //     IF [VAR_24] IS EQUAL TO None THEN
                //     //     //         RETURN np.eye(18)
                //     //     //     END IF
                //     //     // 
                //     //     //     [VAR_107] = np.empty(self.[VAR_14])
                //     //     //     [VAR_107][FROM 0 TO 17] = [VAR_46]
                //     //     //     [VAR_107][FROM 18 TO 23] = [VAR_45]
                //     //     //     
                //     //     //     RETURN self._regularize_hessian([VAR_24]([VAR_107]))
                //     //     // END FUNCTION
                //     //     // :::END_PSEUDOCODE:::
                //     //     
                //     //     RETURN [VAR_124], [VAR_125], [VAR_126]
                //     // END FUNCTION
                //     // :::END_PSEUDOCODE:::
                // 
                //     IF self.[VAR_80] IS GREATER THAN OR EQUAL TO self.[VAR_13] THEN
                //         BREAK
                //     ENDIF
                // 
                //     [VAR_124], [VAR_125], [VAR_126] = make_sub_funcs([VAR_106])
                // ENDFOR
                // ```
                // :::END_PSEUDOCODE:::
                    
                    // :::PSEUDOCODE:::
                    // ```
                    // BEGIN PSEUDOCODE
                    // 
                    // END PSEUDOCODE
                    // ```
                    // :::END_PSEUDOCODE:::
                    
                    // :::PSEUDOCODE:::
                    // ```
                    // IF [VAR_109].[VAR_54] OR [VAR_109].[VAR_55] < [VAR_89][[VAR_44]] THEN
                    //     SET [VAR_112] TO [VAR_109].[VAR_15]
                    //     SET [VAR_113] TO [VAR_109].[VAR_55]
                    // 
                    //     SET [VAR_114] TO COPY OF [VAR_106]
                    // 
                    //     FOR EACH [VAR_68] IN RANGE(0, 6) DO
                    //         // :::PSEUDOCODE:::
                    //         // FOR EACH [VAR_116] IN [-1, 1]
                    //         //     SET [VAR_117] TO COPY OF [VAR_106]
                    //         //     SET [VAR_117][[VAR_68]] TO [OP_BOUND]([VAR_117][[VAR_68]] + [VAR_116], INTEGER, INTEGER)
                    //         // 
                    //         //     // :::PSEUDOCODE:::
                    //         //     // FUNCTION [VAR_123](INPUT [VAR_46], INPUT [VAR_118] DEFAULT [VAR_117])
                    //         //     //     DECLARE [VAR_107] AS ARRAY OF LENGTH self.[VAR_14]
                    //         //     //     
                    //         //     //     SET [VAR_107][0:18] TO [VAR_46]
                    //         //     //     SET [VAR_107][18:24] TO [VAR_118]
                    //         //     //     
                    //         //     //     RETURN self._evaluate([VAR_107], [VAR_16])
                    //         //     // END FUNCTION
                    //         //     // :::END_PSEUDOCODE:::
                    //         // 
                    //         //     // :::PSEUDOCODE:::
                    //         //     // ASSIGN [VAR_108] TO [VAR_108]
                    //         //     // ASSIGN {'[VAR_52]': 5, '[VAR_53]': 10} TO [VAR_111]
                    //         //     // :::END_PSEUDOCODE::
                    //         // END FOR
                    //         // :::END_PSEUDOCODE:::
                    // 
                    //         IF [VAR_120] IS LESS THAN [VAR_113] THEN
                    //             // :::PSEUDOCODE:::
                    //             SET [VAR_121] TO np.empty(self.[VAR_14])
                    //             SET [VAR_121][:18] TO [VAR_112]
                    //             SET [VAR_121][18:24] TO [VAR_114]
                    //             // :::END_PSEUDOCODE:::
                    //         END IF
                    //     END FOR
                    // END IF
                    // ```
                    // :::END_PSEUDOCODE:::
                        
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