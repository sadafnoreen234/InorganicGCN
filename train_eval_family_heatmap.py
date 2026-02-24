# src/train_eval_family_heatmap.py
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
    elif any(x in compound for x in ["o"]):
        return "Oxide"
    elif "n" in compound:
        return "Nitride"
    elif any(x in compound for x in ["s", "se", "te"]):
        return "Chalcogenide"
    elif any(x in compound for x in ["p", "as", "sb", "bi"]):
        return "Pnictide"
    else:
        return "Other"

def family_heatmap(data, model_name="GCN"):
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Compound", "MP_ID", "Predicted", "Experimental"])
    df["AbsDev"] = np.abs(df["Predicted"] - df["Experimental"])
    df["RelDev"] = df["AbsDev"] / df["Experimental"] * 100

    # Assign families
    df["Family"] = df["Compound"].apply(assign_family)

    # Compute average deviation per family
    family_dev = df.groupby("Family")["RelDev"].mean().reset_index()

    os.makedirs("results", exist_ok=True)
    family_dev.to_csv(f"results/{model_name}_family_deviation.csv", index=False)

    # Heatmap
    plt.figure(figsize=(8,6))
    pivot = family_dev.pivot_table(values="RelDev", index="Family")
    sns.heatmap(pivot, annot=True, cmap="coolwarm", fmt=".1f",
                cbar_kws={"label":"Avg Relative Deviation (%)"})
    plt.title(f"{model_name} Average Deviation by Chemical Family", fontsize=16, fontweight="bold")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_family_heatmap.png", dpi=1200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Example: use your calc vs exp dataset
    data = [
        ["MnF2","mp-560902",7.12,8.1532],
        ["AgI","mp-22894",7.16,7.0033],
        ["RbBr","mp-22867",5.69,4.9034],
        ["Li3N","mp-2251",10.69,10.5035],
        ["BN","mp-984",4.68,6.3936],
        ["GaN","mp-830",10.96,9.8037],
        ["BP","mp-1479",9.27,11.0038],
        ["MoS2","mp-2815",9.76,12.7240],
        ["HgS","mp-9252",12.33,18.2043],
        ["SiC","mp-7140",10.58,9.7844],
        ["ZnTe","mp-8884",11.52,10.1043],
        ["MoSe2","mp-1634",11.73,18.0045],
        ["ZnO","mp-1986",11.35,9.0834],
        ["BiTeI","mp-22965",23.74,14.5060],
        # ... include all compounds from your dataset ...
    ]

    family_heatmap(data, model_name="GCN")
