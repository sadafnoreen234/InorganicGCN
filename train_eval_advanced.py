# src/train_eval_advanced.py
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'

from model import DielectricGCN
from dataset import load_dataset, convert_to_graph

def cross_validation(model_class, graphs, epochs=30, lr=0.001):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(graphs)):
        train_subset = [graphs[i] for i in train_idx]
        test_subset = [graphs[i] for i in test_idx]
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

        model = model_class(num_node_features=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch)
                y = batch.y.view(-1)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

        preds, actuals = [], []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                y = batch.y.view(-1)
                preds.extend(out.cpu().numpy().tolist())
                actuals.extend(y.cpu().numpy().tolist())

        results.append({
            "Fold": fold+1,
            "R2": r2_score(actuals, preds),
            "RMSE": np.sqrt(mean_squared_error(actuals, preds))
        })
    return pd.DataFrame(results)

def learning_curve_sizes(model_class, graphs, epochs=30, lr=0.001):
    sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = []
    for frac in sizes:
        subset_size = int(len(graphs) * frac)
        subset = graphs[:subset_size]
        train_subset, test_subset = train_test_split(subset, test_size=0.2, random_state=42)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

        model = model_class(num_node_features=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch)
                y = batch.y.view(-1)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

        preds, actuals = [], []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                y = batch.y.view(-1)
                preds.extend(out.cpu().numpy().tolist())
                actuals.extend(y.cpu().numpy().tolist())

        results.append({
            "TrainFraction": frac,
            "R2": r2_score(actuals, preds),
            "RMSE": np.sqrt(mean_squared_error(actuals, preds))
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = load_dataset("data/train-00000-of-00001.parquet")
    graphs = [convert_to_graph(row, idx)[0] for idx, row in df.iterrows()]

    os.makedirs("results", exist_ok=True)

    # Cross-validation
    cv_df = cross_validation(DielectricGCN, graphs)
    cv_df.to_csv("results/GCN_cv.csv", index=False)

    # Learning curve vs training size
    lc_df = learning_curve_sizes(DielectricGCN, graphs)
    lc_df.to_csv("results/GCN_learning_curve_sizes.csv", index=False)

    # Plot learning curve vs training size
    plt.figure(figsize=(8,6))
    plt.plot(lc_df["TrainFraction"], lc_df["R2"], marker="o", color="blue", label="R2")
    plt.plot(lc_df["TrainFraction"], lc_df["RMSE"], marker="o", color="red", label="RMSE")
    plt.xlabel("Training Data Fraction", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=14, fontweight="bold")
    plt.title("GCN Learning Curve vs Training Size", fontsize=16, fontweight="bold")
    plt.legend()
    plt.savefig("results/GCN_learning_curve_sizes.png", dpi=1200, bbox_inches="tight")
    plt.close()
