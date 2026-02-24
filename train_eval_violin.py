# src/train_eval_violin.py
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Style: Times New Roman, bold
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'

from model import DielectricGCN
from dataset import load_dataset, convert_to_graph

# Map space groups to crystal systems
def space_group_to_system(space_group_number):
    if 1 <= space_group_number <= 2:
        return "Triclinic"
    elif 3 <= space_group_number <= 15:
        return "Monoclinic"
    elif 16 <= space_group_number <= 74:
        return "Orthorhombic"
    elif 75 <= space_group_number <= 142:
        return "Tetragonal"
    elif 143 <= space_group_number <= 167:
        return "Trigonal"
    elif 168 <= space_group_number <= 194:
        return "Hexagonal"
    elif 195 <= space_group_number <= 230:
        return "Cubic"
    else:
        return "Unknown"

def violin_plot_polycrystal(model_class, df, graphs, model_name="GCN", epochs=50, lr=0.001):
    # Train/test split
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
    systems = df.iloc[test_idx]["space_group"].apply(space_group_to_system).tolist()
    results = pd.DataFrame({"CrystalSystem": systems,
                            "Actual": actuals,
                            "Predicted": preds})

    os.makedirs("results", exist_ok=True)
    results.to_csv(f"results/{model_name}_violin_data.csv", index=False)

    # Violin plot of predicted dielectric constants per crystal system
    plt.figure(figsize=(10,6))
    sns.violinplot(x="CrystalSystem", y="Predicted", data=results,
                   inner="box", palette="Set2", cut=0)
    plt.xlabel("Crystal System", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Dielectric Constant", fontsize=14, fontweight="bold")
    plt.title(f"{model_name} Polycrystalline Dielectric Distribution", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_polycrystal_violin.png", dpi=1200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Load dataset
    df = load_dataset("data/train-00000-of-00001.parquet")
    graphs = [convert_to_graph(row, idx)[0] for idx, row in df.iterrows()]

    violin_plot_polycrystal(DielectricGCN, df, graphs,
                            model_name="GCN", epochs=50, lr=0.001)
