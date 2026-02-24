# src/train_eval.py
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os

# Fonts and style
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'

# Direct imports
from model import DielectricGCN, DielectricSAGE, DielectricGAT
from dataset import load_dataset, convert_to_graph

def train_and_evaluate(model, graphs, df, model_name="GNN", epochs=50, lr=0.001):
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    train_losses, test_losses = [], []
    preds_test, actuals_test = [], []

    for epoch in range(epochs):
        # Training
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

        # Evaluation
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

    # Metrics
    r2 = r2_score(actuals_test, preds_test)
    mse = mean_squared_error(actuals_test, preds_test)
    rmse = mean_squared_error(actuals_test, preds_test, squared=False)

    # Binary classification metrics (threshold at median)
    threshold = np.median(actuals_test)
    actual_binary = [1 if val >= threshold else 0 for val in actuals_test]
    pred_binary = [1 if val >= threshold else 0 for val in preds_test]

    acc = accuracy_score(actual_binary, pred_binary)
    prec = precision_score(actual_binary, pred_binary)
    rec = recall_score(actual_binary, pred_binary)
    f1 = f1_score(actual_binary, pred_binary)

    print(f"{model_name} Metrics: R2={r2:.4f}, RMSE={rmse:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    os.makedirs("results", exist_ok=True)

    # Save metrics
    metrics_df = pd.DataFrame([{
        "Model": model_name,
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }])
    metrics_df.to_csv(f"results/{model_name}_metrics.csv", index=False)

    # Cross-validation with R2 and RMSE
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(graphs)):
        train_subset = [graphs[i] for i in train_idx]
        test_subset = [graphs[i] for i in test_idx]
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

        fold_model = type(model)(num_node_features=5)
        optimizer = torch.optim.Adam(fold_model.parameters(), lr=lr)

        for epoch in range(epochs):
            fold_model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                out = fold_model(batch)
                y = batch.y.view(-1)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

        preds, actuals = [], []
        fold_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                out = fold_model(batch)
                y = batch.y.view(-1)
                preds.extend(out.cpu().numpy().tolist())
                actuals.extend(y.cpu().numpy().tolist())

        r2_fold = r2_score(actuals, preds)
        rmse_fold = mean_squared_error(actuals, preds, squared=False)
        cv_results.append({"Fold": fold+1, "R2": r2_fold, "RMSE": rmse_fold})

    pd.DataFrame(cv_results).to_csv(f"results/{model_name}_cv.csv", index=False)

    # Learning curve (epochs vs loss)
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, epochs+1), test_losses, label="Test Loss", color="red")
    plt.xlabel("Epochs", fontsize=14, fontweight="bold")
    plt.ylabel("Loss", fontsize=14, fontweight="bold")
    plt.title(f"{model_name} Learning Curve", fontsize=16, fontweight="bold")
    plt.legend()
    plt.savefig(f"results/{model_name}_learning_curve.png", dpi=1200, bbox_inches="tight")
    plt.close()

    # Scatter plot (train vs test predictions)
    plt.figure(figsize=(8,6))
    train_preds, train_actuals = [], []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            out = model(batch)
            y = batch.y.view(-1)
            train_preds.extend(out.cpu().numpy().tolist())
            train_actuals.extend(y.cpu().numpy().tolist())

    plt.scatter(train_actuals, train_preds, color="blue", s=100, alpha=0.5, label="Train Data")
    plt.scatter(actuals_test, preds_test, color="red", s=100, alpha=0.5, label="Test Data")
    plt.xlabel("Actual Dielectric Constant", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Dielectric Constant", fontsize=14, fontweight="bold")
    plt.title(f"{model_name}: Predicted vs Actual", fontsize=16, fontweight="bold")
    plt.legend()
    plt.savefig(f"results/{model_name}_scatter.png", dpi=1200, bbox_inches="tight")
    plt.close()

    # Learning curves with different training sizes
    sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    lc_results = []
    for frac in sizes:
        subset_size = int(len(graphs) * frac)
        subset = graphs[:subset_size]
        train_subset, test_subset = train_test_split(subset, test_size=0.2, random_state=42)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

        lc_model = type(model)(num_node_features=5)
        optimizer = torch.optim.Adam(lc_model.parameters(), lr=lr)

        for epoch in range(epochs):
            lc_model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                out = lc_model(batch)
                y = batch.y.view(-1)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

        preds, actuals = [], []
        lc_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                out = lc_model(batch)
                y = batch.y.view(-1)
                preds.extend(out.cpu().numpy().tolist())
                actuals.extend(y.cpu().numpy().tolist())

        r2_frac = r2_score(actuals, preds)
        rmse_frac = mean_squared_error(actuals, preds, squared=False)
        lc_results.append({"TrainFraction": frac, "R2": r2_frac, "RMSE": rmse_frac})

    pd.DataFrame(lc_results).to_csv(f"results/{model_name}_learning_curve_sizes.csv", index=False)

    plt.figure(figsize=(8,6))
    plt.plot([r["TrainFraction"] for r in lc_results], [r["R2"] for r in lc_results], marker="o", label="R2")
    plt.plot([r["TrainFraction"] for r in lc_results], [r["RMSE"] for r in lc_results], marker="o", label="RMSE")
    plt.xlabel("Training Data Fraction", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=14, fontweight="bold")
    plt.title(f"{model_name}: Predicted vs Actual", fontsize=16, fontweight="bold")
    plt.legend()
    plt.savefig(f"results/{model_name}_scatter.png", dpi=1200, bbox_inches="tight")
    plt.close()