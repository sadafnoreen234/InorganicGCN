# src/train_eval_comprehensive_analysis.py
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Style: Times New Roman, bold
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'

# Map space groups to crystal systems (if available)
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

def comprehensive_analysis(data, df=None, model_name="GCN"):
    # Convert to DataFrame
    df_data = pd.DataFrame(data, columns=["Compound", "MP_ID", "Predicted", "Experimental"])

    # Compute deviations
    df_data["AbsDev"] = np.abs(df_data["Predicted"] - df_data["Experimental"])
    df_data["RelDev"] = df_data["AbsDev"] / df_data["Experimental"] * 100

    # Metrics
    MAD = df_data["AbsDev"].mean()
    MARD = df_data["RelDev"].mean()
    RMSE = np.sqrt(((df_data["Predicted"] - df_data["Experimental"])**2).mean())
    R2 = 1 - (((df_data["Predicted"] - df_data["Experimental"])**2).sum() /
              ((df_data["Experimental"] - df_data["Experimental"].mean())**2).sum())
    Bias = (df_data["Predicted"] - df_data["Experimental"]).mean()

    os.makedirs("results", exist_ok=True)
    df_data.to_csv(f"results/{model_name}_comprehensive_data.csv", index=False)

    # 1. Parity plot with ±10% lines
    plt.figure(figsize=(8,6))
    plt.scatter(df_data["Experimental"], df_data["Predicted"], color="blue", alpha=0.6, s=80)
    max_val = max(df_data["Experimental"].max(), df_data["Predicted"].max()) + 2
    plt.plot([0, max_val], [0, max_val], color="black", linestyle="--", label="Perfect Agreement")
    plt.plot([0, max_val], [0, 1.1*max_val], color="red", linestyle=":", label="+10% Deviation")
    plt.plot([0, max_val], [0, 0.9*max_val], color="green", linestyle=":", label="-10% Deviation")
    plt.text(0.05*max_val, 0.9*max_val,
             f"MAD={MAD:.2f}, MARD={MARD:.1f}%\nRMSE={RMSE:.2f}, R²={R2:.2f}, Bias={Bias:.2f}",
             fontsize=12, fontweight="bold")
    plt.xlabel("Experimental Dielectric Constant", fontsize=14, fontweight="bold")
    plt.ylabel("Calculated Dielectric Constant", fontsize=14, fontweight="bold")
    plt.title(f"{model_name} Calculated vs Experimental", fontsize=16, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_parity_plot.png", dpi=1200, bbox_inches="tight")
    plt.close()

    # 2. Histogram of relative deviations
    plt.figure(figsize=(8,6))
    sns.histplot(df_data["RelDev"], bins=20, kde=True, color="gray")
    plt.xlabel("Relative Deviation (%)", fontsize=14, fontweight="bold")
    plt.ylabel("Frequency", fontsize=14, fontweight="bold")
    plt.title(f"{model_name} Relative Deviation Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_deviation_histogram.png", dpi=1200, bbox_inches="tight")
    plt.close()

    # 3. Boxplots per crystal system (if space_group available)
    if df is not None and "space_group" in df.columns:
        df_data["CrystalSystem"] = df["space_group"].apply(space_group_to_system).values[:len(df_data)]
        plt.figure(figsize=(10,6))
        sns.boxplot(x="CrystalSystem", y="RelDev", data=df_data, palette="Set2")
        plt.xlabel("Crystal System", fontsize=14, fontweight="bold")
        plt.ylabel("Relative Deviation (%)", fontsize=14, fontweight="bold")
        plt.title(f"{model_name} Deviation by Crystal System", fontsize=16, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"results/{model_name}_crystal_system_boxplot.png", dpi=1200, bbox_inches="tight")
        plt.close()

    # 4. Error vs. band gap scatter (if band_gap available)
    if df is not None and "band_gap" in df.columns:
        df_data["BandGap"] = df["band_gap"].values[:len(df_data)]
        plt.figure(figsize=(8,6))
        plt.scatter(df_data["BandGap"], df_data["RelDev"], color="purple", alpha=0.6, s=80)
        plt.xlabel("Band Gap (eV)", fontsize=14, fontweight="bold")
        plt.ylabel("Relative Deviation (%)", fontsize=14, fontweight="bold")
        plt.title(f"{model_name} Deviation vs Band Gap", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"results/{model_name}_error_vs_bandgap.png", dpi=1200, bbox_inches="tight")
        plt.close()

    # 5. Error vs. dielectric constant scatter
    plt.figure(figsize=(8,6))
    plt.scatter(df_data["Experimental"], df_data["RelDev"], color="orange", alpha=0.6, s=80)
    plt.xlabel("Experimental Dielectric Constant", fontsize=14, fontweight="bold")
    plt.ylabel("Relative Deviation (%)", fontsize=14, fontweight="bold")
    plt.title(f"{model_name} Deviation vs Dielectric Constant", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_error_vs_dielectric.png", dpi=1200, bbox_inches="tight")
    plt.close()

    # 6. Outlier table (top 10 largest deviations)
    outliers = df_data.nlargest(10, "RelDev")
    outliers.to_csv(f"results/{model_name}_outliers.csv", index=False)

if __name__ == "__main__":
    # Input data (Compound, MP ID, Predicted, Experimental) — truncated for brevity
    data = [
        ["MnF2","mp-560902",7.12,8.1532],
        ["AgI","mp-22894",7.16,7.0033],
        ["RbBr","mp-22867",5.69,4.9034],
        ["Li3N","mp-2251",10.69,10.5035],
        ["BN","mp-984",4.68,6.3936],
        ["GaN","mp-830",10.96,9.8037],
        # ... include all compounds from your dataset ...
        ["BiTeI","mp-22965",23.74,14.5060],
    ]

    # If you want crystal system/band gap analysis, load your full MP dataset here
    try:
        from dataset import load_dataset
        df = load_dataset("data/train-00000-of-00001.parquet")
    except:
        df = None

    comprehensive_analysis(data, df, model_name="GCN")
