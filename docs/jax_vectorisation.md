# Implementation Strategy: Vectorizing the Objective

**1. Refactoring Objective:**
Modify `DoubleGaussObjective` to accept a batched input array of shape `(BatchSize, 24)`. 

**2. Implementing `jax.vmap`:**
Wrap the core `_f_loss` function:
`batched_loss = jax.vmap(self._f_loss, in_axes=(0, 0))`
This instructs JAX to map the function over the batch dimension (axis 0).

**3. Memory Management:**
Since memory bandwidth on HPC nodes is a bottleneck, the `BatchSize` must be tuned to fit within the CPU L3 cache (typically 32–64 candidates).