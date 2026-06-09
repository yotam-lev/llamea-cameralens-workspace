```plaintext
Initialize Population:
    - Generate n_particles positions randomly within [-1, 1]
    - Initialize velocities as half of random numbers within [-1, 1]

Evaluate Initial Positions:
    - For each particle i:
        - Evaluate function value f at its position
        - Set personal best position and function value for particle i
        - Update global best if current function value is better

Main Loop (while evaluations < budget):
    - Increment iteration counter
    
    - Update Hessian Scale every 5 iterations or if not set:
        - Extract continuous variables x_c and categorical IDs cat_ids from gbest_x
        - Compute full_x by concatenating x_c and cat_ids
        - Calculate raw Hessian H_raw using hess_func(full_x)
        - Regularize Hessian to get H_scale
    
    - PSO Update:
        - Generate random vectors r1 and r2 for each particle
        - For each particle:
            - Compute standard velocity v_std using inertia weight, cognitive component, social component
            - Scale velocity by Hessian matrix H_scale
            - Update position based on new velocity
    
    - Evaluate New Positions:
        - For each particle:
            - If evaluations have reached budget, break
            - Evaluate function value f at current position
            - Update personal best if current function value is better
            - Update global best if current function value is better
                - Also update overall best if necessary
    
    - Local Search on GBest:
        - If evaluations have not reached budget and a global best solution exists:
            - Extract continuous variables x_c and categorical IDs cat_ids from gbest_x
            - Compute full_x by concatenating x_c and cat_ids
            - Regularize Hessian to get H_reg
            
            - Define obj, jac, hess functions for local search
            - Call minimize function with trust-constr method
            - Evaluate candidate solution and update global best if necessary

Return Best Function Value and Position:
    - Return overall best function value and position
```