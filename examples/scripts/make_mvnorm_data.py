import json
import numpy as np

np.random.seed(2023)
mean = (1, 5)
cov = [[1, 0.8], [0.8, 1]]
arr = np.random.multivariate_normal(mean, cov, (2000,))

rows = [{"var1": float(v[0]), "var2": float(v[1])} for v in arr]

with open("examples/data/mvnorm_train.json", "w") as f:
    json.dump(rows, f)

print("saved examples/data/mvnorm_train.json rows=", len(rows))
