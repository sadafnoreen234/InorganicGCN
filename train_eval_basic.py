# src/train_eval_basic.py
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'

from model import DielectricGCN, DielectricSAGE, DielectricGAT
from dataset import load_dataset, convert_to_graph

def train_and_evaluate(model, graphs, model_name="GNN", epochs=50, lr=0.001):
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    train_losses, test_losses = [], []
    preds_test, actuals_test = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            y = batch.y.view(-1)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        total_test_loss = 0.0
        preds_test, actuals_test = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                y = batch.y.view(-1)
                loss = loss_fn(out, y)
                total_test_loss += loss.item()
                preds_test.extend(out.cpu().numpy().tolist())
                actuals_test.extend(y.cpu().numpy().tolist())
        test_losses.append(total_test_loss / len(test_loader))

    r2 = r2_score(actuals_test, preds_test)
    mse = mean_squared_error(actuals_test, preds_test)
    rmse = np.sqrt(mse)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame([{"Model": model_name, "R2": r2, "RMSE": rmse}]).to_csv(
        f"results/{model_name}_metrics.csv", index=False)

    # Learning curve
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, epochs+1), test_losses, label="Test Loss", color="red")
    plt.xlabel("Epochs", fontsize=14, fontweight="bold")
    plt.ylabel("Loss", fontsize=14, fontweight="bold")
    plt.title(f"{model_name} Learning Curve", fontsize=16, fontweight="bold")
    plt.legend()
    plt.savefig(f"results/{model_name}_learning_curve.png", dpi=1200, bbox_inches="tight")
    plt.close()

    # Scatter plot (blue train, red test, brittle/transparent)
    plt.figure(figsize=(8,6))
    train_preds, train_actuals = [], []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            out = model(batch)
            y = batch.y.view(-1)
            train_preds.extend(out.cpu().numpy().tolist())
            train_actuals.extend(y.cpu().numpy().tolist())

    plt.scatter(train_actuals, train_preds, color="blue", s=100, alpha=0.4, label="Train Data")
    plt.scatter(actuals_test, preds_test, color="red", s=100, alpha=0.4, label="Test Data")
    plt.xlabel("Actual Dielectric Constant", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Dielectric Constant", fontsize=14, fontweight="bold")
    plt.title(f"{model_name}: Predicted vs Actual", fontsize=16, fontweight="bold")
    plt.legend()
    plt.savefig(f"results/{model_name}_scatter.png", dpi=1200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    df = load_dataset("data/train-00000-of-00001.parquet")
    graphs = [convert_to_graph(row, idx)[0] for idx, row in df.iterrows()]
    models = {
        "GCN": DielectricGCN(num_node_features=5),
        "GraphSAGE": DielectricSAGE(num_node_features=5),
        "GAT": DielectricGAT(num_node_features=5)
    }
    for name, model in models.items():
        print(f"\nTraining {name}...")
        train_and_evaluate(model, graphs, model_name=name, epochs=50, lr=0.001)
