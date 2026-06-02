# Workflow: /debug
# Description: Isolate, diagnose, and resolve runtime errors, configuration drift, or algorithmic failures.

## Step 1: Triage (Executor)
- Inspect the provided stack trace, error log, or faulty output.
- Locate the precise file, line number, and variables causing the exception.
- Run an environment check if systemic errors (e.g., CUDA issues, import failures, out-of-memory errors) are suspected.

## Step 2: Root Cause Analysis (Strategist)
- Explain *why* the failure occurred, separating symptoms from the actual root cause.
- Identify if the fix requires a simple patch or an architectural adjustment to prevent recurrence.

## Step 3: Resolution & Fix (Executor)
- Present the exact file modifications needed to resolve the bug.
- Apply the changes cleanly.

## Step 4: Verification (Orchestrator)
- Execute the code, test suite, or pipeline segment to verify the error is cleared.
- Ensure no secondary regressions were introduced by the patch.