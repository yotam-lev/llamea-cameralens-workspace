# Local Stress Test Plan: LLaMEA Framework Stability

**Objective:** Validate that the `lens_optimisation.py` framework can handle continuous, multi-generational execution using a local Ollama model to generate syntactically correct Python optimizers, which are then evaluated at scale ($50,000$ to $100,000$ budget) without performance degradation.

## Phase 1: Local Model Configuration
* We must bypass external API calls and hook the LLaMEA framework into a local Ollama instance.
* We will use a lightweight, fast local model (e.g., `llama3`, `codellama`, or `qwen2.5-coder` depending on local availability) to ensure rapid code generation during the test.

## Phase 2: The Simplified Experiment (`lens_local_test.py`)
* **Base Architecture:** Create a simplified version of the experiment (analogous to a `lens_v2.py`) that strips away complex memetic mutation prompts.
* **Problem Constraints:** * `budget_factor`: $50,000$ to $100,000$.
  * `eval_timeout`: Ensure the sandbox allows enough time for the JAX engine to run $10^5$ iterations (e.g., 300 seconds).
* **LLaMEA Constraints:**
  * `budget` (Generations): 3 to 5. We only need a few generations to prove consistency and memory stability.
  * `n_parents`: 1
  * `n_offspring`: 2 (Keep population small to accelerate the test).

## Phase 3: Execution & Profiling
* The IDE will execute the run and monitor standard output.
* **Success Criteria:**
  1. Ollama successfully generates valid `Optimizer` classes.
  2. The BLADE framework evaluates the $10^5$ budget smoothly.
  3. The time-per-evaluation remains consistent across generations (no memory leaks).