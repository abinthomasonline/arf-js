import json
import pandas as pd

SRC = "examples/data/penguins.csv"
OUT = "examples/data/penguins_train.json"
df = pd.read_csv(SRC)

columns = [
    "species",
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]
df = df[columns].copy()
island_codes = {"Torgersen": 0, "Biscoe": 1, "Dream": 2}
df["island_code"] = df["island"].map(island_codes)

body_mass_median = df["body_mass_g"].median()
df["heavy"] = None
non_null = df["body_mass_g"].notnull()
df.loc[non_null, "heavy"] = (df.loc[non_null, "body_mass_g"] >= body_mass_median).astype(bool)

rows = df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")
with open(OUT, "w") as f:
    json.dump(rows, f)

print("saved", OUT, "rows=", len(rows))
