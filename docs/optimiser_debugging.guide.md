# Resolving the Evaluation Discrepancy: LLAMEA vs. DoubleGauss

To ensure the Differential Evolution (DE) optimizer scores consistently across the LLAMEA evaluation framework and the visualization test script, we must address fundamental mismatches between the optimizer's assumptions and the `DoubleGaussObjective` landscape. 

The discrepancy arises because LLAMEA tests against continuous, gradient-free black-box functions, while the visualization test involves a mixed-integer projected space with available first-order gradients.

Here is the technical breakdown of the required fixes:

## 1. Addressing the Continuous-to-Discrete Projection Trap
**The Problem:** The optimizer proposes infinitesimal continuous changes. The test script projects these into a mixed-integer space (`split_theta`, `gradient_cont_int`). This causes identical discrete states to be evaluated repeatedly, stalling the optimizer.
**The Solution:** Implement a mechanism to add a tiny jitter or force exploration when the population collapses onto the same projected discrete states.

## 2. Implementing Gradient-Assisted Local Search
**The Problem:** The `DoubleGaussObjective` explicitly provides gradients (`enable_grad=True`), but the `__call__` method ignores `grad_func`. 100,000 evaluations of pure black-box search cannot compete with a gradient-informed search on steep optical landscapes.
**The Solution:** Inject a gradient descent local search step. Periodically use the `grad_func` to take a scaled step in the direction of the negative gradient for the best individual.

## 3. Fixing the Population Starvation
**The Problem:** A fixed `pop_size = 20` for an 18+ dimensional problem guarantees premature convergence in DE.
**The Solution:** Dynamically size the population based on dimensionality. A standard heuristic is `pop_size = 10 * dim`. 

## 4. Correcting the jADE Implementation
**The Problem:** The current implementation uses a global random walk for $F$ and $CR$, which is not jADE. jADE tracks successful mutation parameters and updates their distribution means ($\mu_F$, $\mu_{CR}$).
**The Solution:** Maintain an archive of successful $F$ and $CR$ values in each generation. Update $\mu_{CR}$ using a standard arithmetic mean and $\mu_F$ using a Lehmer mean.