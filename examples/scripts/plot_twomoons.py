import json
import pandas as pd
import matplotlib.pyplot as plt

n_train = 2000
n_test = 1000

with open("examples/data/twomoons_train.json", "r") as f:
    train_rows = json.load(f)
with open("examples/output/twomoons_synth.json", "r") as f:
    synth_rows = json.load(f)

df = pd.DataFrame(train_rows)
df_syn = pd.DataFrame(synth_rows)
df_test = df[:n_train].sample(n=n_test, random_state=2022)

x_min = df_test["dim_1"].min()
x_max = df_test["dim_1"].max()
y_min = df_test["dim_2"].min()
y_max = df_test["dim_2"].max()

fig = plt.figure(figsize=(30, 20))

ax1 = plt.subplot(2, 2, 1)
ax1.scatter(df_test["dim_1"], df_test["dim_2"], c=df_test["target"].astype(int), alpha=0.5, cmap="coolwarm", s=10)
ax1.set_title("Original Data", fontsize=30)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.set_aspect("equal", adjustable="box")

ax2 = plt.subplot(2, 2, 2)
ax2.scatter(df_syn["dim_1"], df_syn["dim_2"], c=df_syn["target"].astype(int), alpha=0.5, cmap="coolwarm", s=10)
ax2.set_title("Synthesized Data", fontsize=30)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_aspect("equal", adjustable="box")

plt.tight_layout()
out = "examples/output/twomoons_scatter.png"
plt.savefig(out, dpi=220)
print("saved", out)
