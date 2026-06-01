# Performance Analysis: M3 Pro vs. HPC Compute Cluster

**Current Bottleneck:**
The performance discrepancy (MacBook: 57s vs. HPC: 278s) is caused by the overhead of `multiprocessing.Pool` using the `spawn` method. In an HPC environment, spawning 64 fresh Python interpreters forces 64 redundant JIT compilations of the JAX objective function, causing extreme latency.

**Proposed Solution: JAX-Level Vectorization (`vmap`)**
Instead of evaluating candidates in separate Python processes, we will leverage `jax.vmap`. This allows the objective function to process a batch of candidates ($N \times 24$ dimensions) as a single tensor operation.
- **Hardware Utilization:** This utilizes the HPC's AVX-512 instruction sets (CPU) or CUDA cores (GPU) at the C++ level.
- **Zero Overhead:** No new Python processes, no redundant JIT compilation, and no `fork`/`spawn` deadlocks.