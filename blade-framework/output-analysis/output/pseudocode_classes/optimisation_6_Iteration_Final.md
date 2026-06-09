```plaintext
function Initialize(budget, dim):
    this.budget = budget
    this.dim = dim
    this.evals = 0
    this.best_f = infinity
    this.best_x = array of zeros with length dim

function optimize(func, grad_func=None, hess_func=None):
    sigma_init = 0.35
    sigma = sigma_init
    popsize = 10
    min_local_search_freq = 2
    max_local_search_freq = 1
    n_init = min(10, budget // 15)
    population = initialize_population(n_init, dim)

    for x in population:
        if number_of_evaluations >= budget:
            break
        evaluate(x, func)

    mean = best_x[:18].copy()
    cat_state = random_categorical_state(dim=6)
    es = CMAEvolutionStrategy(mean, sigma, {'popsize': popsize, 'verbose': -1})
    generation = 0
    stagnation_counter = 0
    best_f_history = []

    while number_of_evaluations < budget:
        if generation % 5 == 0 and popsize < 20:
            popsize = min(20, popsize * 1.5)
            es.options['popsize'] = popsize

        candidates = es.ask()
        if not candidates:
            break

        fitnesses = []

        p_cat_mut = max(0.1, 0.5 - (number_of_evaluations / budget) * 0.3)

        for x_c in candidates:
            full_x = concatenate(x_c, cat_state)
            f = evaluate(full_x, func)
            fitnesses.append(f)

        es.tell(candidates, fitnesses)

        if number_of_evaluations > 10:
            best_f_history.append(best_f)
            if len(best_f_history) > 5:
                best_f_history.pop(0)
            improvement_rate = (best_f_history[-1] - best_f_history[0]) / abs(best_f_history[0])
            if improvement_rate < 1e-4:
                stagnation_counter += 1
                sigma *= 0.8
            else:
                stagnation_counter = 0
                sigma *= 1.05
                sigma = clip(sigma, min=sigma_init * 0.1, max=sigma_init * 5)

        budget_frac = number_of_evaluations / budget
        freq = max(min_local_search_freq, int(min_local_search_freq + (max_local_search_freq - min_local_search_freq) * (1 - budget_frac)**2))

        if generation % freq == 0 and grad_func is not None and hess_func is not None:
            break_if_budget_exceeded()
            gradient = grad_func(best_x[:18])
            break_if_budget_exceeded()
            hessian = hess_func(best_x[:18])
            break_if_budget_exceeded()

            eigenvalues, eigenvectors = np.linalg.eig(hessian)
            condition_number = np.max(eigenvalues) / np.min(eigenvalues)
            lambda_reg = determine_regularization_parameter(condition_number, eigenvalues)
            regularized_hessian = hessian + lambda_reg * np.eye(len(hessian))

            step_direction = solve_linear_system(regularized_hessian, gradient)

            step_size = scale_step_size(step_direction, stagnation_counter)

            new_continuous_vars = update_variables(best_x[:18], step_size, step_direction, lower_bound=-1, upper_bound=1)

            if random() < p_cat_mut:
                cat_state = mutate_categorical_state(cat_state, index=random.randint(0, 5), mutation_range=(-1, 2))

            evaluate(concatenate(new_continuous_vars, cat_state), func)

        es.options['verbose'] = -1
        generation += 1

    return best_f, best_x
```