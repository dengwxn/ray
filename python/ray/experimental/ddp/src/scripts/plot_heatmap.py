import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..core.config import parse_args

args = parse_args()

# Read CSV files
data = []
for ls in args["layer_size_values"]:
    for nl in args["num_layers_values"]:
        filename = f"ls{ls}_nl{nl}.csv"
        filepath = os.path.join(args["output_path"], filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            row = {
                "layer_size": ls,
                "num_layers": nl,
                "relative_torch": np.log(df["torch"][0] / df["torch-ddp"][0]),
                "relative_ray": np.log(df["ray-ddp"][0] / df["torch-ddp"][0]),
            }
            data.append(row)

# Convert to DataFrame
df = pd.DataFrame(data)

# Create heatmaps
plt.figure(figsize=(12, 5))

# Table 1: relative_torch
plt.subplot(1, 2, 1)
pivot_torch = df.pivot(
    index="layer_size", columns="num_layers", values="relative_torch"
)
sns.heatmap(pivot_torch, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Log Relative Time (torch / torch-ddp)")

# Table 2: relative_ray
plt.subplot(1, 2, 2)
pivot_ray = df.pivot(index="layer_size", columns="num_layers", values="relative_ray")
sns.heatmap(pivot_ray, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Log Relative Time (ray-ddp / torch-ddp)")

plt.tight_layout()
plt.savefig(f"{args['output_path']}/heatmap.png")
