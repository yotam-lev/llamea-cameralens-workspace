def optimize(eval_x):
    # Note the different spacing and CAST instead of CONVERT_TO_INT
    eval_x[18:24] = CAST(ROUND(eval_x[18:24]), INTEGER)
    return eval_x
