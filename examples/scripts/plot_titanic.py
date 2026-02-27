import json
import pandas as pd
import matplotlib.pyplot as plt

with open("examples/data/titanic_train.json", "r") as f:
    real_rows = json.load(f)
with open("examples/output/titanic_synth.json", "r") as f:
    synth_rows = json.load(f)

df_real = pd.DataFrame(real_rows)
df_syn = pd.DataFrame(synth_rows)

fig = plt.figure(figsize=(18, 12))

ax1 = plt.subplot(2, 2, 1)
real_scatter = df_real.dropna(subset=["age", "fare"])
syn_scatter = df_syn.dropna(subset=["age", "fare"])
x_min = real_scatter["age"].min()
x_max = real_scatter["age"].max()
y_min = real_scatter["fare"].min()
y_max = real_scatter["fare"].max()
ax1.scatter(
    real_scatter["age"],
    real_scatter["fare"],
    c=real_scatter["survived"].astype(int),
    cmap="coolwarm",
    alpha=0.5,
    s=20,
)
ax1.set_title("Titanic Real: age vs fare")
ax1.set_xlabel("age")
ax1.set_ylabel("fare")
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.set_aspect("equal", adjustable="box")

ax2 = plt.subplot(2, 2, 2)
ax2.scatter(
    syn_scatter["age"],
    syn_scatter["fare"],
    c=syn_scatter["survived"].astype(int),
    cmap="coolwarm",
    alpha=0.5,
    s=20,
)
ax2.set_title("Titanic Synth: age vs fare")
ax2.set_xlabel("age")
ax2.set_ylabel("fare")
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_aspect("equal", adjustable="box")

ax3 = plt.subplot(2, 2, 3)
real_sex = df_real["sex"].value_counts(dropna=False).sort_index()
syn_sex = df_syn["sex"].value_counts(dropna=False).sort_index()
cats = sorted(set(real_sex.index.astype(str)) | set(syn_sex.index.astype(str)))
real_vals = [real_sex.get(c, 0) for c in cats]
syn_vals = [syn_sex.get(c, 0) for c in cats]
x = range(len(cats))
ax3.bar([v - 0.2 for v in x], real_vals, width=0.4, label="real")
ax3.bar([v + 0.2 for v in x], syn_vals, width=0.4, label="synth")
ax3.set_xticks(list(x))
ax3.set_xticklabels(cats)
ax3.set_title("Sex Distribution")
ax3.legend()

ax4 = plt.subplot(2, 2, 4)
real_emb = df_real["embarked"].fillna("NULL").value_counts().sort_index()
syn_emb = df_syn["embarked"].fillna("NULL").value_counts().sort_index()
cats2 = sorted(set(real_emb.index.astype(str)) | set(syn_emb.index.astype(str)))
real_vals2 = [real_emb.get(c, 0) for c in cats2]
syn_vals2 = [syn_emb.get(c, 0) for c in cats2]
x2 = range(len(cats2))
ax4.bar([v - 0.2 for v in x2], real_vals2, width=0.4, label="real")
ax4.bar([v + 0.2 for v in x2], syn_vals2, width=0.4, label="synth")
ax4.set_xticks(list(x2))
ax4.set_xticklabels(cats2)
ax4.set_title("Embarked Distribution")
ax4.legend()

plt.tight_layout()
out = "examples/output/titanic_comparison.png"
plt.savefig(out, dpi=220)
print("saved", out)
