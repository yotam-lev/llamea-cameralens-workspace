# Agentic Debugging Architecture: Framework Environment Discrepancy

**Objective:** Identify the structural root cause of evaluation divergence between `lens_v4.py` (using `ContextualLensOptimisation`) and the standalone script (using `LensOptimisation`). The discrepancy is confirmed *not* to be caused by the optimization class itself, the budget factor, or seed variances.

## Investigative Vectors
The agents will ignore the optimizer's logic and instead audit the execution environments for the following:
1. **Module Resolution (The Import Trap):** The standalone script manually injects `CAMERA_LENS_ROOT`. Are both scripts resolving to the exact same physical `double_gauss_objective.py` file, or is one loading a stale/different version from another path?
2. **Wrapper Mechanics (`ContextualLensOptimisation` vs `LensOptimisation`):** Does the `Contextual` subclass override `_build_objective`, apply different penalty scaling for `inf` losses, or alter the `[-1, 1]` boundary projections?
3. **State Leakage:** `lens_v4.py` evaluates across multiple instances sequentially. Is the internal state of the optical simulation (e.g., ray tracing kernel data or gradient baseline) failing to reset between LLaMEA evaluations, causing cascading errors?

## Agent Roles
* **Lead Architect (Coordinator):** Directs the investigation, ensuring the focus remains strictly on the framework's architecture and objective wrappers, not the generated LLM code.
* **Software Engineer (Execution):** Injects telemetry into the BLADE framework source code, modifies paths, and runs the parallel executions.
* **Checker (Telemetry):** Wraps the underlying `DoubleGaussObjective.objective_theta` to capture the exact raw inputs it receives and the exact raw outputs it yields *before* the framework processes them.
* **Analyst (Diff Engine):** Compares the I/O telemetry line-by-line to pinpoint the exact moment the environments diverge.