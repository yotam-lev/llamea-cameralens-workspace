```plaintext
WHILE evaluations < budget DO
    IF current best fitness = previous best THEN
        stagnation counter++
    
    IF hess_func available AND iteration is odd THEN
        H = compute_hessian(best solution)
        eigenvalues, eigenvectors = decompose(H)
        regularize_H(eigenvalues, eigenvectors)
        C_inv_sqrt = inverse_square_root_curvature(eigenvalues, eigenvectors)
    ELSE
        C_inv_sqrt = identity_matrix(18)

    IF stagnation counter > 15 THEN
        REGENERATE population with random solutions in range [-1, 1]
        INITIALIZE an empty list to store fitness values
        FOR each candidate solution x IN population DO
            IF evaluations exceed budget THEN BREAK the loop
            EVALUATE fitness of x using _evaluate function
            APPEND fitness value to population fitness list
        END FOR
        RESET stagnation counter to 0
        CONTINUE to next iteration

    SELECT worst individual in population to perturb

    IF both hess_func and gradient functions are available THEN
        APPLY HEAE step to generate a new candidate solution x_new
        EVALUATE fitness of the new candidate solution using _evaluate function
        REPLACE the worst solution in the population with the new candidate solution
        UPDATE the fitness value of the replaced solution in the population fitness list

    SELECT 12 best individuals based on fitness for DE
    FOR _ FROM 1 TO 25 DO
        SELECT 3 DISTINCT INDICES idx_rand FROM selected individuals
        CREATE NEW SOLUTION mu AS A COMBINATION OF THREE RANDOMLY SELECTED SOLUTIONS
            SET mu = pop[idx_rand[0]] + 0.8 * (pop[idx_rand[1]] - pop[idx_rand[2]])
        ADD NOISE TO THE FIRST 18 DIMENSIONS OF mu USING NORMAL DISTRIBUTION
            mu[:18] += np.random.normal(0, 0.1, 18)
        ROUND AND CLIP THE LAST 6 DIMENSIONS OF mu TO BE IN RANGE [0, 5]
            mu[18:24] = np.clip(np.round(np.random.uniform(0.0, 5.99, 6)), 0, 5).astype(int)
        ADD NEW SOLUTION mu TO THE OFFSPRING LIST
    END FOR

    COMBINE population and offspring
    SORT combined list by fitness values
    KEEP top 25 solutions with the lowest fitness values

END WHILE
```