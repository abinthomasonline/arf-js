from sklearn.datasets import make_moons
import json

n_train = 2000
n_test = 1000
X, y = make_moons(n_samples=n_train + n_test, noise=0.1, random_state=2022)

train_rows = []
for i in range(n_train):
    train_rows.append({
        "dim_1": float(X[i, 0]),
        "dim_2": float(X[i, 1]),
        "target": str(int(y[i])),
    })

with open("examples/data/twomoons_train.json", "w") as f:
    json.dump(train_rows, f)

print("saved examples/data/twomoons_train.json rows=", len(train_rows))
