# Debugging Guide: Sealing the Double-Gauss Environment

**Goal:** Verify that the LLM-generated optimizers are evaluated in a strict Mixed-Integer Non-Linear Programming (MINLP) environment without access to continuous glass exploits or missing gradients.

## Step 1: Environment Setup in Anti-Gravity IDE
2. Navigate to your standalone visualization/testing script.
3. Ensure the objective is initialized with gradients enabled:
   `obj = DoubleGaussObjective(enable_grad=True, enable_hessian=False)`

## Step 2: Setting Breakpoints
To confirm the exploit is patched, you need to watch the exact moment the optimizer's normalized array is converted into physical lens parameters.

1. Place a breakpoint inside the `bounded_func` definition, specifically on this line:
   `x_projected = obj.project_theta(x_real, lb=lb, ub=ub)`
2. Place another breakpoint inside `bounded_grad` on:
   `xc, xi = obj.split_theta(x_projected)`

## Step 3: Inspecting the State (The "Watch" List)
Run the script in Debug mode. When execution pauses at your breakpoints, use the IDE's variable inspector/watch panel to check the following:

* **Watch `x_normalized`:** Ensure values are strictly within the $[-1, 1]$ range.
* **Watch `x_real`:** Look at indices `[18:24]`. You should see floating-point numbers (e.g., `4.732`, `12.11`). This is the optimizer *trying* to use unobtainium.
* **Watch `x_projected`:** Look at indices `[18:24]` again. These **must** be clean integers (e.g., `5.0`, `12.0`). If they are not integers, the projection logic is failing.

## Step 4: Verifying Gradient Propagation
When the debugger pauses in `bounded_grad`:
1. Step over the `obj.gradient_cont_int(xc, xi)` call.
2. Inspect the returned gradient array (`g_val`).
3. Verify that `g_val` has a shape of `(18,)` and contains non-zero floats. This confirms the optimizer is receiving the directional data it needs for the continuous geometric variables, while the discrete glass variables (`xi`) are safely excluded from the gradient calculation.