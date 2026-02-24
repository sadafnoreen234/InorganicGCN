# src/train_eval_correlation.py
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os

# Style: Times New Roman, bold
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'

from dataset import load_dataset

def correlation_heatmap(df, features, target=None, model_name="GCN"):
    # If target column exists, include it
    cols = [f for f in features if f in df.columns]
    if target and target in df.columns:
        cols.append(target)

    # Compute correlation matrix
    corr_df = df[cols].corr()

    os.makedirs("results", exist_ok=True)

    # Save correlation matrix to CSV
    corr_df.to_csv(f"results/{model_name}_correlation_matrix.csv")

    # Plot heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f",
                cbar=True, square=True, annot_kws={"size":10, "weight":"bold"})
    plt.title(f"{model_name} Descriptor Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.savefig(f"results/{model_name}_correlation_heatmap.png", dpi=1200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Load dataset
    df = load_dataset("data/train-00000-of-00001.parquet")

    # Define descriptors (adjust to match your actual dataframe columns)
    features = ["poly_electronic", "poly_total", "band_gap", "volume", "log(poly_total)"]

    # If your dataset has a target column, set it here (otherwise leave None)
    target = None  # e.g., "dielectric_constant" if present

    correlation_heatmap(df, features, target, model_name="GCN")
