from src.canonicalizer import Canonicalizer

canon = Canonicalizer("temp", "class_1")
res = canon.canonicalize("x = y_var.copy()\nprint(x)")
print("RES1:", res)

res2 = canon.canonicalize("x = np.clip(val, -5, 5)\nprint(x)")
print("RES2:", res2)
