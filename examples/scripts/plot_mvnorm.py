import json
import pandas as pd
import matplotlib.pyplot as plt

with open("examples/data/mvnorm_train.json", "r") as f:
    real_rows = json.load(f)
with open("examples/output/mvnorm_synth.json", "r") as f:
    synth_rows = json.load(f)

df = pd.DataFrame(real_rows)
df_syn = pd.DataFrame(synth_rows)

x_min = df["var1"].min()
x_max = df["var1"].max()
y_min = df["var2"].min()
y_max = df["var2"].max()

plt.figure(figsize=(30, 25))

ax1 = plt.subplot(2, 2, 1)
ax1.plot(df.to_numpy()[:, 0], df.to_numpy()[:, 1], ".", alpha=0.5)
ax1.grid(True)
ax1.set_title("Original Data", fontsize=30)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.set_aspect("equal", adjustable="box")

ax2 = plt.subplot(2, 2, 2)
ax2.plot(df_syn.to_numpy()[:, 0], df_syn.to_numpy()[:, 1], ".", alpha=0.5)
ax2.grid(True)
ax2.set_title("Synthesized Data", fontsize=30)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_aspect("equal", adjustable="box")

plt.tight_layout()
out = "examples/output/mvnorm_scatter.png"
plt.savefig(out, dpi=220)
print("saved", out)
