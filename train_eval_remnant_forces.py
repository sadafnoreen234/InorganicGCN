# src/train_eval_remnant_forces.py
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Style: Times New Roman, bold
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'

def remnant_forces_plot(data, model_name="GCN"):
    # Data format: Compound, Predicted, Experimental, ResidualForce (eV/Å)
    df = pd.DataFrame(data, columns=["Compound","Predicted","Experimental","ResidualForce"])

    # Compute deviations
    df["AbsDev"] = np.abs(df["Predicted"] - df["Experimental"])
    df["RelDev"] = df["AbsDev"] / df["Experimental"] * 100

    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/{model_name}_remnant_forces.csv", index=False)

    # Scatter plot with color by residual force
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(df["Experimental"], df["Predicted"],
                          c=df["ResidualForce"], cmap="plasma", s=100, alpha=0.8, edgecolor="k")

    max_val = max(df["Experimental"].max(), df["Predicted"].max()) + 2
    plt.plot([0, max_val], [0, max_val], color="black", linestyle="--", label="Perfect Agreement")

    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Residual Force (eV/Å)", fontsize=12, fontweight="bold")

    # Labels and formatting
    plt.xlabel("Experimental Dielectric Constant (forces <0.01 eV/Å)", fontsize=14, fontweight="bold")
    plt.ylabel("Calculated Dielectric Constant (with residual forces)", fontsize=14, fontweight="bold")
    plt.title(f"{model_name} Effect of Remnant Interatomic Forces", fontsize=16, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_remnant_forces.png", dpi=1200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Example dataset (replace with your actual values)
    data = [
        ["CompoundA", 10.5, 11.0, 0.02],
        ["CompoundB", 7.8, 8.1, 0.05],
        ["CompoundC", 12.3, 12.0, 0.01],
        ["CompoundD", 9.0, 9.5, 0.15],
        ["CompoundE", 14.2, 13.8, 0.08],
    ]

    remnant_forces_plot(data, model_name="GCN")
