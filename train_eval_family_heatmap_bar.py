# src/train_eval_family_heatmap_bar.py
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Style: Times New Roman, bold
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'

def assign_family(compound):
    """Assign chemical family based on compound formula (simplified rules)."""
    compound = compound.lower()
    if any(x in compound for x in ["f", "cl", "br", "i"]):
        return "Halide"
    elif "o" in compound:
        return "Oxide"
    elif "n" in compound:
        return "Nitride"
    elif any(x in compound for x in ["s", "se", "te"]):
        return "Chalcogenide"
    elif any(x in compound for x in ["p", "as", "sb", "bi"]):
        return "Pnictide"
    else:
        return "Other"

def family_analysis(data, model_name="GCN"):
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Compound", "MP_ID", "Predicted", "Experimental"])
    df["AbsDev"] = np.abs(df["Predicted"] - df["Experimental"])
    df["RelDev"] = df["AbsDev"] / df["Experimental"] * 100

    # Assign families
    df["Family"] = df["Compound"].apply(assign_family)

    # Compute stats per family
    family_stats = df.groupby("Family").agg(
        Count=("Compound", "count"),
        AvgRelDev=("RelDev", "mean")
    ).reset_index()

    os.makedirs("results", exist_ok=True)
    family_stats.to_csv(f"results/{model_name}_family_stats.csv", index=False)

    # Heatmap of average deviation
    plt.figure(figsize=(8,6))
    pivot = family_stats.pivot_table(values="AvgRelDev", index="Family")
    sns.heatmap(pivot, annot=True, cmap="coolwarm", fmt=".1f",
                cbar_kws={"label":"Avg Relative Deviation (%)"})
    plt.title(f"{model_name} Average Deviation by Chemical Family", fontsize=16, fontweight="bold")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_family_heatmap.png", dpi=1200, bbox_inches="tight")
    plt.close()

    # Stacked bar chart: count + deviation
    fig, ax1 = plt.subplots(figsize=(10,6))

    # Bar for counts
    ax1.bar(family_stats["Family"], family_stats["Count"], color="steelblue", alpha=0.7)
    ax1.set_xlabel("Chemical Family", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Number of Compounds", fontsize=14, fontweight="bold", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Line for average deviation
    ax2 = ax1.twinx()
    ax2.plot(family_stats["Family"], family_stats["AvgRelDev"], color="darkred", marker="o", linewidth=2)
    ax2.set_ylabel("Avg Relative Deviation (%)", fontsize=14, fontweight="bold", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred")

    plt.title(f"{model_name} Family Coverage & Accuracy", fontsize=16, fontweight="bold")
    fig.tight_layout()
    plt.savefig(f"results/{model_name}_family_bar.png", dpi=1200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Example dataset (truncated for brevity)
    data = [
        ["MnF2","mp-560902",7.12,8.1532],
        ["AgI","mp-22894",7.16,7.0033],
        ["RbBr","mp-22867",5.69,4.9034],
        ["Li3N","mp-2251",10.69,10.5035],
        ["BN","mp-984",4.68,6.3936],
        ["GaN","mp-830",10.96,9.8037],
        ["MoS2","mp-2815",9.76,12.7240],
        ["HgS","mp-9252",12.33,18.2043],
        ["ZnO","mp-1986",11.35,9.0834],
        ["BiTeI","mp-22965",23.74,14.5060],
        # ... include all compounds ...
    ]

    family_analysis(data, model_name="GCN")
