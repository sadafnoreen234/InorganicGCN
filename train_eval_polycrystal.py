# src/train_eval_polycrystal.py
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os

# Style: Times New Roman, bold
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'

from model import DielectricGCN
from dataset import load_dataset, convert_to_graph

def polycrystal_estimate(model_class, df, graphs, crystal_col, target_col,
                         model_name="GCN", epochs=50, lr=0.001):
    # Train/test split with indices
    train_idx, test_idx = train_test_split(range(len(graphs)), test_size=0.2, random_state=42)
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    # Model setup
    model = model_class(num_node_features=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            y = batch.y.view(-1)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

    # Predictions
    preds, actuals = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            y = batch.y.view(-1)
            preds.extend(out.cpu().numpy().tolist())
            actuals.extend(y.cpu().numpy().tolist())

    # Map predictions back to crystal systems
    systems = df.iloc[test_idx][crystal_col].tolist()
    results = pd.DataFrame({"CrystalSystem": systems,
                            "Actual": actuals,
                            "Predicted": preds})

    # Aggregate polycrystalline estimates per crystal system
    poly_estimates = results.groupby("CrystalSystem")["Predicted"].agg(["mean","std"]).reset_index()
    poly_estimates.rename(columns={"mean": "PolycrystalEstimate", "std": "StdDev"}, inplace=True)

    os.makedirs("results", exist_ok=True)
    poly_estimates.to_csv(f"results/{model_name}_polycrystal_estimates.csv", index=False)

    # Plot bar chart with error bars
    plt.figure(figsize=(8,6))
    plt.bar(poly_estimates["CrystalSystem"], poly_estimates["PolycrystalEstimate"],
            yerr=poly_estimates["StdDev"], color="steelblue", alpha=0.7, capsize=5)
    plt.xlabel("Crystal System", fontsize=14, fontweight="bold")
    plt.ylabel("Estimated Dielectric Constant", fontsize=14, fontweight="bold")
    plt.title(f"{model_name} Polycrystalline Dielectric Estimates", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_polycrystal_estimates.png", dpi=1200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Load dataset
    df = load_dataset("data/train-00000-of-00001.parquet")
    graphs = [convert_to_graph(row, idx)[0] for idx, row in df.iterrows()]

    # Inspect columns to find the correct crystal system column
    print("Available columns:", df.columns.tolist())

    # Replace with the actual column name in your dataset
    crystal_col = "space_group"   # <-- change this to the correct column
    target_col = "dielectric_constant"  # <-- change if different

    polycrystal_estimate(DielectricGCN, df, graphs,
                         crystal_col=crystal_col,
                         target_col=target_col,
                         model_name="GCN",
                         epochs=50, lr=0.001)
