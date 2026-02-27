import json
import pandas as pd
import matplotlib.pyplot as plt

with open("examples/data/penguins_train.json", "r") as f:
    real_rows = json.load(f)
with open("examples/output/penguins_synth.json", "r") as f:
    synth_rows = json.load(f)

df_real = pd.DataFrame(real_rows)
df_syn = pd.DataFrame(synth_rows)

fig = plt.figure(figsize=(18, 12))

ax1 = plt.subplot(2, 2, 1)
real_scatter = df_real.dropna(subset=["bill_length_mm", "flipper_length_mm"])
syn_scatter = df_syn.dropna(subset=["bill_length_mm", "flipper_length_mm"])
x_min = real_scatter["bill_length_mm"].min()
x_max = real_scatter["bill_length_mm"].max()
y_min = real_scatter["flipper_length_mm"].min()
y_max = real_scatter["flipper_length_mm"].max()
species_codes = real_scatter["species"].astype("category").cat.codes
ax1.scatter(
    real_scatter["bill_length_mm"],
    real_scatter["flipper_length_mm"],
    c=species_codes,
    cmap="viridis",
    alpha=0.6,
    s=25,
)
ax1.set_title("Penguins Real: bill_length vs flipper_length")
ax1.set_xlabel("bill_length_mm")
ax1.set_ylabel("flipper_length_mm")
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.set_aspect("equal", adjustable="box")

ax2 = plt.subplot(2, 2, 2)
species_codes_syn = syn_scatter["species"].astype("category").cat.codes
ax2.scatter(
    syn_scatter["bill_length_mm"],
    syn_scatter["flipper_length_mm"],
    c=species_codes_syn,
    cmap="viridis",
    alpha=0.6,
    s=25,
)
ax2.set_title("Penguins Synth: bill_length vs flipper_length")
ax2.set_xlabel("bill_length_mm")
ax2.set_ylabel("flipper_length_mm")
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_aspect("equal", adjustable="box")

ax3 = plt.subplot(2, 2, 3)
real_species = df_real["species"].fillna("NULL").value_counts().sort_index()
syn_species = df_syn["species"].fillna("NULL").value_counts().sort_index()
cats = sorted(set(real_species.index.astype(str)) | set(syn_species.index.astype(str)))
real_vals = [real_species.get(c, 0) for c in cats]
syn_vals = [syn_species.get(c, 0) for c in cats]
x = range(len(cats))
ax3.bar([v - 0.2 for v in x], real_vals, width=0.4, label="real")
ax3.bar([v + 0.2 for v in x], syn_vals, width=0.4, label="synth")
ax3.set_xticks(list(x))
ax3.set_xticklabels(cats, rotation=15)
ax3.set_title("Species Distribution")
ax3.legend()

ax4 = plt.subplot(2, 2, 4)
real_heavy = df_real["heavy"].apply(lambda v: "NULL" if pd.isna(v) else str(v)).value_counts()
syn_heavy = df_syn["heavy"].apply(lambda v: "NULL" if pd.isna(v) else str(v)).value_counts()
cats2 = sorted(set(real_heavy.index) | set(syn_heavy.index))
real_vals2 = [real_heavy.get(c, 0) for c in cats2]
syn_vals2 = [syn_heavy.get(c, 0) for c in cats2]
x2 = range(len(cats2))
ax4.bar([v - 0.2 for v in x2], real_vals2, width=0.4, label="real")
ax4.bar([v + 0.2 for v in x2], syn_vals2, width=0.4, label="synth")
ax4.set_xticks(list(x2))
ax4.set_xticklabels(cats2)
ax4.set_title("Heavy (bool) Distribution")
ax4.legend()

plt.tight_layout()
out = "examples/output/penguins_comparison.png"
plt.savefig(out, dpi=220)
print("saved", out)
