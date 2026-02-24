# src/train_eval_residuals.py
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

def residual_analysis(model_class, graphs, model_name="GCN", epochs=50, lr=0.001):
    # Train/test split
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
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

    residuals = np.array(preds) - np.array(actuals)

    # Metrics
    r2 = r2_score(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    print(f"{model_name} Residual Analysis: R2={r2:.4f}, RMSE={rmse:.4f}")

    os.makedirs("results", exist_ok=True)

    # Save residuals to CSV
    pd.DataFrame({"Actual": actuals, "Predicted": preds, "Residual": residuals}).to_csv(
        f"results/{model_name}_residuals.csv", index=False)

    # Residual plot (Actual vs Residuals)
    plt.figure(figsize=(8,6))
    plt.scatter(actuals, residuals, color="blue", alpha=0.4, s=100)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Actual Dielectric Constant", fontsize=14, fontweight="bold")
    plt.ylabel("Residual (Predicted - Actual)", fontsize=14, fontweight="bold")
    plt.title(f"{model_name} Residual Plot", fontsize=16, fontweight="bold")
    plt.savefig(f"results/{model_name}_residual_plot.png", dpi=1200, bbox_inches="tight")
    plt.close()

    # Histogram of residuals
    plt.figure(figsize=(8,6))
    plt.hist(residuals, bins=30, color="gray", alpha=0.7)
    plt.xlabel("Residuals", fontsize=14, fontweight="bold")
    plt.ylabel("Frequency", fontsize=14, fontweight="bold")
    plt.title(f"{model_name} Residual Distribution", fontsize=16, fontweight="bold")
    plt.savefig(f"results/{model_name}_residual_histogram.png", dpi=1200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    df = load_dataset("data/train-00000-of-00001.parquet")
    graphs = [convert_to_graph(row, idx)[0] for idx, row in df.iterrows()]
    residual_analysis(DielectricGCN, graphs, model_name="GCN", epochs=50, lr=0.001)
