import json
import numpy as np
import matplotlib.pyplot as plt

with open("examples/data/digits_zero_train.json", "r") as f:
    real_rows = json.load(f)
with open("examples/output/digits_synth.json", "r") as f:
    synth_rows = json.load(f)

# deterministic pick of first 4 originals, as visual reference
real_rows = real_rows[:4]


def row_to_image(row):
    vals = [int(row[f"px_{i}"]) for i in range(64)]
    return np.array(vals).reshape(8, 8)

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

for i in range(4):
    axes[0, i].set_axis_off()
    axes[0, i].imshow(row_to_image(real_rows[i]), cmap=plt.cm.gray_r, interpolation="nearest", vmin=0, vmax=16)
    axes[1, i].set_axis_off()
    axes[1, i].imshow(row_to_image(synth_rows[i]), cmap=plt.cm.gray_r, interpolation="nearest", vmin=0, vmax=16)

fig.text(0.5, 0.97, "Original zeros", ha="center", fontsize=14)
fig.text(0.5, 0.48, "Synthetic zeros", ha="center", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "examples/output/digits_grid.png"
plt.savefig(out, dpi=220)
print("saved", out)
