import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import r2_score, mean_squared_error
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

def evaluate(model, graphs):
    """
    Evaluate a trained model on train/test split.
    Produces scatter plot, predictions CSV, and metrics.
    """

    # Split dataset
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    model.eval()

    preds_train, actuals_train = [], []
    preds_test, actuals_test = [], []

    with torch.no_grad():
        for batch in train_loader:
            out = model(batch)
            preds_train.append(out.item())
            actuals_train.append(batch.y.item())

        for batch in test_loader:
            out = model(batch)
            preds_test.append(out.item())
            actuals_test.append(batch.y.item())

    # Scatter plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=actuals_train, y=preds_train, label="Train", color="blue")
    sns.scatterplot(x=actuals_test, y=preds_test, label="Test", color="red")
    plt.xlabel("Actual Dielectric Constant")
    plt.ylabel("Predicted Dielectric Constant")
    plt.title("Predicted vs Actual")
    plt.legend()
    plt.savefig("results/scatter_plot.png")
    print("Saved scatter plot to results/scatter_plot.png")

    # Save CSV
    df_train = pd.DataFrame({"Actual": actuals_train, "Predicted": preds_train, "Set": "Train"})
    df_test = pd.DataFrame({"Actual": actuals_test, "Predicted": preds_test, "Set": "Test"})
    df = pd.concat([df_train, df_test], ignore_index=True)
    df.to_csv("results/predictions.csv", index=False)
    print("Saved predictions to results/predictions.csv")

    # Metrics
    r2 = r2_score(actuals_test, preds_test)
    mse = mean_squared_error(actuals_test, preds_test)
    print(f"Test R2={r2:.4f}, MSE={mse:.4f}")
