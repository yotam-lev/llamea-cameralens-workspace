```
CLASS Optimizer:
    METHOD __init__(INPUT [VAR_3] OF TYPE int, INPUT [VAR_4] OF TYPE int):
        SET SELF.[VAR_3] TO [VAR_3]
        SET SELF.[VAR_4] TO [VAR_4]
        SET SELF.[VAR_99] TO 0
        SET SELF.[VAR_100] TO infinity
        SET SELF.[VAR_101] TO np.zeros([VAR_4])
        SET SELF.[VAR_102] TO 18
        SET SELF.[VAR_103] TO 6
        SET SELF.[VAR_104] TO np.full((SELF.[VAR_103], 6), infinity)
        SET SELF.[VAR_105] TO np.ones((SELF.[VAR_103], 6)) / 6.0
        SET SELF.[VAR_106] TO np.zeros((6, SELF.[VAR_102]))
    END METHOD
    FUNCTION _evaluate([VAR_5], [VAR_6]):
        [VAR_50] = [OP_BOUND](COPY([VAR_5]), INTEGER, INTEGER)
        [VAR_50][18:24] = [OP_BOUND](ROUND([VAR_50][18:24]), INTEGER, INTEGER).astype(int)
        [VAR_51] = [VAR_6]([VAR_50])
        self.[VAR_47] += 1
        RETURN [VAR_51]
    END FUNCTION
    FUNCTION [VAR_0](INPUT [VAR_20], INPUT [VAR_109] = NONE, INPUT [VAR_21] = NONE, **[VAR_22]):
        SET [VAR_110] TO MIN(20, MAX(2, FLOOR([VAR_3] / 10)))
        SET pop TO np.[VAR_25].uniform(-1, 1, SIZE = ([VAR_110], [VAR_4]))
        FOR EACH [VAR_5] IN pop DO
            IF self.[VAR_53] IS GREATER THAN OR EQUAL TO [VAR_3] THEN
                BREAK
            END IF
            CALL self._evaluate WITH ARGUMENTS ([VAR_5], [VAR_6])
        END FOR
        SET [VAR_111] TO self.copy([VAR_101][:18])
        SET [VAR_112] TO 0.3
        SET [VAR_34] TO 15
        SET [VAR_35] TO -1
        SET [VAR_114] TO cma.CMAEvolutionStrategy([VAR_111], [VAR_112], [VAR_113])
        SET [VAR_116] TO 0
        WHILE self.[VAR_99] < [VAR_3] DO
            SET [VAR_117] TO ARRAY([np.[VAR_25].choice(6, [VAR_40]=self.[VAR_105][[VAR_41]]) FOR [VAR_41] FROM 0 TO (self.[VAR_103] - 1)])
            SET [VAR_118] TO ARRAY OF ZEROS WITH LENGTH self.[VAR_102]
            SET [VAR_119] TO ARRAY OF ZEROS WITH LENGTH 6
            FOR [VAR_41] FROM 0 TO (self.[VAR_103] - 1) DO
                SET [VAR_120] TO 1.0 / (self.[VAR_104][[VAR_41], [VAR_117][[VAR_41]]] + 1e-12)
                SET [VAR_119][[VAR_117][[VAR_41]]] TO [VAR_120]
                SET [VAR_118] TO [VAR_118] + ([VAR_120] * self.[VAR_106][[VAR_41]])
            END FOR
            SET [VAR_118] TO [VAR_118] / ([VAR_119].sum() + 1e-12)
            CALL [VAR_114].SET WITH ARGUMENTS ([VAR_121]=[VAR_118])
            IF [VAR_21] IS NOT NONE AND self.[VAR_99] < [VAR_3] THEN
                TRY:
                    SET [VAR_122] TO CONCATENATE([[VAR_118], [VAR_117]])
                    SET [VAR_123] TO [VAR_21]([VAR_122])
                    SET [VAR_124] TO np.[VAR_60].EIGVALSH([VAR_123])
                    SET [VAR_125] TO MAX(ABS([VAR_124])) / (MIN(ABS([VAR_124])) + 1e-8)
                    CALL [VAR_114].SET WITH ARGUMENTS ([VAR_56], [VAR_112] / SQRT([VAR_125]) * 0.5)
                CATCH:
                    // Handle exception
                END TRY
            END IF
            SET [VAR_135] TO self.[VAR_104] - MIN(self.[VAR_104], [VAR_81]=1, [VAR_82]=True)
            self.[VAR_105] ← np.exp(-0.1 * [VAR_135])
            self.[VAR_105] ← self.[VAR_105] + 0.01
            self.[VAR_105] ← self.[VAR_105] / SUM(self.[VAR_105], [VAR_81]=1, [VAR_82]=True)
            IF ([VAR_116] MOD 5 == 0) AND (self.[VAR_99] < [VAR_3] - 3) THEN
                SET [VAR_136] TO np.argmin(MIN(self.[VAR_104], [VAR_81]=1))
                SET [VAR_137] TO np.argmin(self.[VAR_104][[VAR_136]])
                SET [VAR_138] TO COPY([VAR_117])
                SET [VAR_138][[VAR_136]] TO [VAR_137]
                IF [VAR_21] IS NOT NONE AND [VAR_109] IS NOT NONE THEN
                    TRY:
                        SET [VAR_140] TO [VAR_21](CONCATENATE([[VAR_139], [VAR_138]]))
                        SET [VAR_124], [VAR_141] TO np.[VAR_60].eigh([VAR_140])
                        SET [VAR_124] TO ABS([VAR_124]) + 1e-8
                        SET [VAR_142] TO [VAR_141] @ DIAG([VAR_124]) @ [VAR_141].T
                        SET [VAR_143] TO [VAR_109](CONCATENATE([[VAR_139], [VAR_138]]))
                        SET [VAR_144] TO -np.[VAR_60].solve([VAR_142], [VAR_143])
                    CATCH:
                        // HANDLE EXCEPTION
                    END TRY
                END IF
            END IF
            SET [VAR_114].[VAR_150] TO -1
            INCREMENT [VAR_116] BY 1
        END WHILE
        RETURN self.[VAR_100], self.[VAR_101]
    END FUNCTION
END CLASS
```