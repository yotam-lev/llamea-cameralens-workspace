def optimize(eval_x):
    eval_x[18:24] = CONVERT_TO_INT(CLIP(ROUND(eval_x[18:24]), 0, 5))
    return eval_x
