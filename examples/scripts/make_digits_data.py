from sklearn.datasets import load_digits
import pandas as pd
import json

# Match arfpy tutorial: only digit 0 images, pixels treated as categorical
digits = load_digits(n_class=1)
arr = digits.images.reshape((len(digits.images), -1))
df = pd.DataFrame(arr)

rows = []
for _, row in df.iterrows():
    item = {f"px_{i}": str(int(row[i])) for i in range(df.shape[1])}
    rows.append(item)

with open("examples/data/digits_zero_train.json", "w") as f:
    json.dump(rows, f)

print("saved examples/data/digits_zero_train.json rows=", len(rows))
