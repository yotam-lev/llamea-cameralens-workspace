# Step 1: Refactor `lens_v5.py`
Open `lens_v5.py` and make the following precise adjustments to the prompts and configuration:

1.  **Fix Categorical Bounds in `task_prompt`:**
    * Change the categorical instructions to match V4. Replace the `[-1, 1]` rounding logic with: `"The categorical variables x[18:24] MUST be mapped to valid glass integers [0, 5] using np.clip(np.round(x[18:24]), 0, 5).astype(int)."`
2.  **Strengthen Hessian Guardrails in `task_prompt`:**
    * Update the regularization instruction to: `"The exact Hessian will often be indefinite. You MUST ensure it is positive-definite by taking the absolute value of its eigenvalues or adding a sufficiently large identity matrix before passing it to any solver or Newton step."`
3.  **Update `example_prompt` `_evaluate` Method:**
    * Replace the `eval_x[18:24]` rounding logic with: `eval_x[18:24] = np.clip(np.round(eval_x[18:24]), 0, 5).astype(int)`
4.  **Restore Exploration Budgets:**
    * In `configure_run`, set `budget = 10` (for testing) and `budget_factor = 5000` (enough to ensure solvers don't instantly starve, but low enough for rapid validation). 

# Step 2: Create `validate_v5_fast.py`
Create a new file named `validate_v5_fast.py` in the root directory. This script will import the configured experiment from `lens_v5.py` but override parameters to force a rapid, dry-run style validation to check syntax and guardrails.

**Requirements for `validate_v5_fast.py`:**
* Import `configure_run` from `lens_v5`.
* Initialize the LLM using a fast, cheap model if available in `config.py` (e.g., GPT-4o-mini or equivalent).
* Override the budget to strictly `budget = 2` (only 2 generations) and `budget_factor = 100`.
* Set `training_seeds = [(1,)]` and `test_seeds = []` to minimize problem loading time.
* Wrap the `experiment()` call in a try/except block that prints a detailed traceback if the LLM-generated code violates the `__call__` signature or crashes the scipy solver.

# Step 3: Run Validation
Execute `python validate_v5_fast.py` in the terminal.
* Verify that the LLM successfully generates a class with the exact `__call__(self, func, grad_func=None, hess_func=None, **kwargs)` signature.
* Check the standard output to confirm that `_evaluate` is successfully parsing the integer categorical bounds without returning `inf`.
* Observe the variation in `best_f` between generation 1 and 2 to ensure the algorithm is successfully exploring the continuous subspace.