import json
import pandas as pd

SRC = "examples/data/titanic.csv"
OUT = "examples/data/titanic_train.json"
df = pd.read_csv(SRC)

columns = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
    "class",
    "adult_male",
    "alone",
    "survived",
]

df = df[columns].copy()
df = df.sample(n=800, random_state=2026).reset_index(drop=True)

rows = df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")
with open(OUT, "w") as f:
    json.dump(rows, f)

print("saved", OUT, "rows=", len(rows))
