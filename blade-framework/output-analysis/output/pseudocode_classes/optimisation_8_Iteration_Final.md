```plaintext
WHILE evaluations < budget DO
    1. Sample categorical state from manifold probabilities.
    2. Continuous Optimization (CMA) conditioned on current manifold:
        a. Calculate weighted mean of best continuous points.
        b. If Hessian function is available and within budget, adjust sigma based on Hessian condition number.
        c. Perform CMA-ES iterations to find better continuous solutions.
    3. Dynamic Categorical Propagation (Score-Driven):
        a. Update manifold probabilities based on scores.
        b. Add exploration bias to probabilities.
    4. Trust-Region Refinement on Best Manifold:
        a. If generation is a multiple of 5 and evaluations are less than budget - 3, refine the best manifold using Hessian and gradient if available.
    INCREMENT generation counter
END WHILE

RETURN BEST FUNCTION VALUE AND POSITION
```