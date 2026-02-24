# src/dataset.py
from torch_geometric.data import Data
import torch
import pandas as pd

def load_dataset(path):
    return pd.read_parquet(path)

def convert_to_graph(row, idx):
    features = [
        float(row["poly_electronic"]),
        float(row["poly_total"]),
        float(row["band_gap"]),
        float(row["volume"]),
        float(row["log(poly_total)"])
    ]
    x = torch.tensor([features], dtype=torch.float)   # [1, num_features]

    edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor(float(row["n"]), dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, y=y)
    graph.batch = torch.tensor([0], dtype=torch.long)  # ✅ ensure batching works

    return graph, None, None
