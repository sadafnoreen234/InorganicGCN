# src/train_eval_feature_importance.py
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

def feature_importance(model_class, df, graphs, features, model_name="GCN", epochs=30, lr=0.001):
    # Baseline performance
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

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

    baseline_r2 = r2_score(actuals, preds)

    # Permutation importance
    importances = []
    for feat in features:
        df_shuffled = df.copy()
        df_shuffled[feat] = np.random.permutation(df_shuffled[feat].values)
        graphs_shuffled = [convert_to_graph(row, idx)[0] for idx, row in df_shuffled.iterrows()]

        test_loader = DataLoader(graphs_shuffled, batch_size=32, shuffle=False)
        preds, actuals = [], []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                y = batch.y.view(-1)
                preds.extend(out.cpu().numpy().tolist())
                actuals.extend(y.cpu().numpy().tolist())

        r2_shuffled = r2_score(actuals, preds)
        importances.append(baseline_r2 - r2_shuffled)

    # Save results
    os.makedirs("results", exist_ok=True)
    fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
    fi_df.to_csv(f"results/{model_name}_feature_importance.csv", index=False)

    # Plot
    plt.figure(figsize=(8,6))
    plt.bar(features, importances, color="gray", alpha=0.7)
    plt.xlabel("Feature", fontsize=14, fontweight="bold")
    plt.ylabel("Importance (ΔR²)", fontsize=14, fontweight="bold")
    plt.title(f"{model_name} Feature Importance", fontsize=16, fontweight="bold")
    plt.savefig(f"results/{model_name}_feature_importance.png", dpi=1200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    df = load_dataset("data/train-00000-of-00001.parquet")
    graphs = [convert_to_graph(row, idx)[0] for idx, row in df.iterrows()]
    features = ["poly_electronic", "poly_total", "band_gap", "volume", "log(poly_total)"]

    feature_importance(DielectricGCN, df, graphs, features, model_name="GCN", epochs=30, lr=0.001)
