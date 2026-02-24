# src/train_eval_bandgap_polycrystal.py
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

def bandgap_polycrystal_plot(model_class, df, graphs, model_name="GCN",
                             epochs=50, lr=0.001):
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

    # Map predictions back to band gaps and formulas
    bandgaps = df.iloc[test_idx]["band_gap"].tolist()
    formulas = df.iloc[test_idx]["formula"].tolist()

    results = pd.DataFrame({"Formula": formulas,
                            "BandGap": bandgaps,
                            "PredictedDielectric": preds,
                            "ActualDielectric": actuals})

    os.makedirs("results", exist_ok=True)
    results.to_csv(f"results/{model_name}_bandgap_polycrystal.csv", index=False)

    # Plot
    plt.figure(figsize=(10,7))

    # Scatter inorganic compounds (green)
    plt.scatter(results["BandGap"], results["PredictedDielectric"],
                color="green", alpha=0.6, s=60, label="Inorganic Compounds")

    # Example polymer data (red points, placeholder values)
    polymer_bandgaps = [5.5, 6.0, 6.5]  # Replace with actual polymer Eg values
    polymer_dielectrics = [2.2, 2.5, 2.8]  # Replace with actual polymer dielectric constants
    plt.scatter(polymer_bandgaps, polymer_dielectrics,
                color="red", marker="o", s=80, label="Polymers (Huan et al.)")

    # Figures of merit lines (example placeholders)
    x_vals = np.linspace(0, max(results["BandGap"].max(), 7), 200)
    fom1 = 10 / (x_vals + 1)   # Replace with actual formula
    fom2 = 20 / (x_vals + 1)   # Replace with actual formula
    plt.plot(x_vals, fom1, color="red", linestyle="--", label="Figure of Merit 1")
    plt.plot(x_vals, fom2, color="green", linestyle="--", label="Figure of Merit 2")

    # Highlight specific compounds
    highlight = {"HfO2": 5.8, "SiO2": 9.0, "Polyethylene": 6.5}
    highlight_dielectric = {"HfO2": 25, "SiO2": 3.9, "Polyethylene": 2.3}
    for formula, eg in highlight.items():
        plt.scatter(eg, highlight_dielectric[formula], color="black", s=120, marker="*", label=formula)
        plt.text(eg, highlight_dielectric[formula]+0.5, formula, fontsize=12, fontweight="bold")

    # Annotate promising compounds (example: top 5 by dielectric constant)
    promising = results.nlargest(5, "PredictedDielectric")
    for _, row in promising.iterrows():
        plt.text(row["BandGap"], row["PredictedDielectric"]+0.5,
                 row["Formula"], fontsize=10, fontweight="bold")

    # Labels and formatting
    plt.xlabel("DFT-GGA+U Band Gap (eV)", fontsize=14, fontweight="bold")
    plt.ylabel("Polycrystalline Dielectric Constant", fontsize=14, fontweight="bold")
    plt.title(f"{model_name} Polycrystalline Dielectric vs Band Gap", fontsize=16, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_bandgap_polycrystal.png", dpi=1200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    df = load_dataset("data/train-00000-of-00001.parquet")
    graphs = [convert_to_graph(row, idx)[0] for idx, row in df.iterrows()]

    bandgap_polycrystal_plot(DielectricGCN, df, graphs,
                             model_name="GCN", epochs=50, lr=0.001)
