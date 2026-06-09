import re

op_map = {
    r"(?i)\b([a-zA-Z_]\w*(?:\[[^\]]+\])?)\.copy\(\)": r"copy(\1)",
    r"(?i)(clip\([^,]+),\s*-?\d+(?:\.\d+)?,\s*-?\d+(?:\.\d+)?\)": r"\1, INTEGER, INTEGER)",
}

texts = [
    "x_var.copy()",
    "np.clip(np.round(x[18:24]), 0, 5).astype(int)",
    "jnp.clip(val, -1.5, 3.14)"
]

for t in texts:
    res = t
    for p, r in op_map.items():
        res = re.sub(p, r, res)
    print(res)
