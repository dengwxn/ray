import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

pwd = "/home/wxdeng/Ray-Sea2Sky/python/ray/experimental/ddp"
version = "0110"

args: Dict[str, Any] = {}
args["layer_size_values"] = [2560, 1280, 640, 512, 320, 160, 80]
args["num_layers_values"] = [10, 40, 160, 250, 640, 2560, 10240]
args["torch_ddp"] = f"{pwd}/results/xuhui/torch_ddp/grids/{version}"
args["ray_bucketing"] = f"{pwd}/results/xuhui/ray_bucketing_unify/grids/{version}"
args["ray_bucketing_overlapping"] = (
    f"{pwd}/results/xuhui/ray_bucketing_overlapping_unify/grids/{version}"
)
args["table"] = f"{pwd}/results/xuhui/same_size/{version}/ddp.csv"
args["figure"] = f"{pwd}/results/xuhui/same_size/{version}/ddp.png"

keys = ["torch_ddp", "ray_bucketing", "ray_bucketing_overlapping"]
for key in keys:
    assert os.path.exists(args[key]), args[key]
os.makedirs(os.path.dirname(args["table"]), exist_ok=True)

# Read CSV files
data = []
for ls, nl in zip(args["layer_size_values"], args["num_layers_values"]):
    row = {
        "layer_size": ls,
        "num_layers": nl,
    }
    filename = f"ls{ls}_nl{nl}_rank0.csv"

    for key in keys:
        filepath = os.path.join(args[key], filename)
        assert os.path.exists(filepath), filepath
        df = pd.read_csv(filepath)
        total = df.iloc[0]["mean"]
        row[key] = total

    row[f"relative_{keys[1]}"] = round(row[keys[1]] / row[keys[0]], 2)
    row[f"relative_{keys[2]}"] = round(row[keys[2]] / row[keys[0]], 2)
    row[f"speedup_overlap"] = round(row[keys[1]] / row[keys[2]], 2)
    data.append(row)

# Convert to DataFrame
df = pd.DataFrame(data)

output_columns = [
    "layer_size",
    "num_layers",
    "torch_ddp",
    "relative_ray_bucketing",
    "relative_ray_bucketing_overlapping",
    "speedup_overlap",
]
df[output_columns].to_csv(args["table"], index=False)
