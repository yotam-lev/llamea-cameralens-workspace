```plaintext
CLASS Optimizer:
    INITIALIZER(max_evals, num_params):
        this.max_evals = max_evals
        this.num_params = num_params
        this.evaluations = 0
        this.best_value = infinity
        this.best_solution = array of zeros with length num_params

    FUNCTION evaluate_candidate(candidate, objective_function):
        IF evaluations >= max_evals:
            RETURN infinity
        END IF

        full_x = array of zeros with length num_params
        first 18 elements of full_x are set to candidate
        round and bound the last 6 elements of full_x to valid integer range

        result = objective_function(full_x)
        increment evaluations

        IF result < best_value:
            update_best_solution(result)
            store_current_solution_as_best()
        END IF

        RETURN result

    FUNCTION __call__(objective_function, gradient_function=None, hessian_function=None):
        population_size = min(10, max(1, max_evals // 15))
        population = generate_random_population(size=population_size, bounds=(-1, 1))

        FOR each solution IN population:
            IF evaluations >= max_evals:
                BREAK
            END IF
            evaluate_candidate(solution, objective_function)
        END FOR

        WHILE evaluations < max_evals:
            candidates = ask_for_new_solutions_from_optimizer()
            IF no_more_candidates:
                BREAK
            END IF

            function_values = []
            FOR each candidate IN candidates:
                concatenated_solution = concatenate(candidate, current_categorical_values)
                function_value = evaluate_candidate(concatenated_solution, objective_function)
                APPEND function_value TO function_values
            END FOR

            tell_optimizer(function_values)

            IF iteration_number MOD 4 == 0 AND gradient_function EXISTS AND hessian_function EXISTS:
                IF evaluations >= max_evals:
                    BREAK
                END IF

                full_x = best_solution
                grad_full_x = gradient_function(full_x)
                hess_full_x = hessian_function(full_x)

                eigenvalues, eigenvectors = eigendecomposition(hess_full_x)
                regularized_eigenvalues = absolute(eigenvalues) + small_constant
                regularized_hessian = eigenvectors @ diagonal(regularized_eigenvalues) @ eigenvectors.T

                search_direction_continuous = -solve(regularized_hessian, grad_full_x)
                new_continuous_values = best_solution's_continuous_part + search_direction_continuous
                bounded_new_continuous_values = bound(new_continuous_values, min, max)

                new_categorical_values = copy(current_categorical_values)
                IF random_number < 0.3:
                    index = random_index(6)
                    new_categorical_values[index] = [OP_BOUND](current_value + small_random_integer, INTEGER_MIN, INTEGER_MAX)
                END IF

                evaluate_candidate(concatenate(bounded_new_continuous_values, new_categorical_values), objective_function)
            END IF

            IF iteration_number MOD 8 == 0 AND evaluations < max_evals - 2:
                IF evaluations >= max_evals:
                    BREAK
                END IF

                optimize_with_constraints(
                    lambda x: objective_function(concatenate(x, best_solution[18:])),
                    initial_guess = best_solution[:18],
                    gradient_function = lambda x: gradient_function(concatenate(x, best_solution[18:])) IF EXISTS,
                    hessian_function = lambda x: hessian_function(concatenate(x, best_solution[18:])) IF EXISTS,
                    variable_bounds = array of 18 tuples (-1, 1),
                    method = '[VAR_29]-[VAR_30]',
                    options = {'max_iterations': 20, '[VAR_32]': 0}
                )

                IF optimization_successful:
                    evaluate_candidate(solution x_new concatenated with best_solution[18:], objective_function)
                END IF
            END IF

            reset_optimizer_iteration_count
            increment generation_number
        END WHILE

        RETURN best_value, best_solution
```