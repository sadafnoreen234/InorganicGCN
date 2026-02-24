# src/train_eval_calc_vs_exp.py
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os

# Style: Times New Roman, bold
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'

def calc_vs_exp_plot(data, model_name="GCN"):
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Compound", "MP_ID", "Predicted", "Experimental"])

    # Compute deviations
    df["AbsDev"] = np.abs(df["Predicted"] - df["Experimental"])
    df["RelDev"] = np.abs(df["Predicted"] - df["Experimental"]) / df["Experimental"] * 100

    MAD = df["AbsDev"].mean()
    MARD = df["RelDev"].mean()

    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/{model_name}_calc_vs_exp.csv", index=False)

    # Scatter plot
    plt.figure(figsize=(8,6))
    plt.scatter(df["Experimental"], df["Predicted"], color="blue", alpha=0.6, s=80)

    # Perfect agreement line
    max_val = max(df["Experimental"].max(), df["Predicted"].max()) + 2
    plt.plot([0, max_val], [0, max_val], color="black", linestyle="--", label="Perfect Agreement")

    # Annotate MAD and MARD
    plt.text(0.05*max_val, 0.9*max_val,
             f"MAD = {MAD:.2f}\nMARD = {MARD:.1f}%",
             fontsize=12, fontweight="bold")

    # Labels and formatting
    plt.xlabel("Experimental Dielectric Constant", fontsize=14, fontweight="bold")
    plt.ylabel("Calculated Dielectric Constant", fontsize=14, fontweight="bold")
    plt.title(f"{model_name} Calculated vs Experimental Dielectric Constants", fontsize=16, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_calc_vs_exp.png", dpi=1200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Input data: Compound, MP ID, Predicted, Experimental
    data = [
        ["MnF2","mp-560902",7.12,8.1532],
        ["AgI","mp-22894",7.16,7.0033],
        ["RbBr","mp-22867",5.69,4.9034],
        ["Li3N","mp-2251",10.69,10.5035],
        ["BN","mp-984",4.68,6.3936],
        ["GaN","mp-830",10.96,9.8037],
        ["BP","mp-1479",9.27,11.0038],
        ["AgI","mp-22925",7.32,7.0033],
        ["NiF2","mp-559798",5.20,5.2039],
        ["RbI","mp-22903",5.69,4.9434],
        ["MoS2","mp-2815",9.76,12.7240],
        ["RbCl","mp-23295",5.65,4.9134],
        ["CaSe","mp-1415",12.47,7.8041],
        ["KN3","mp-827",6.21,6.8542],
        ["ZnO","mp-1986",11.35,9.0834],
        ["HgS","mp-9252",12.33,18.2043],
        ["SiC","mp-7140",10.58,9.7844],
        ["ZnTe","mp-8884",11.52,10.1043],
        ["MoSe2","mp-1634",11.73,18.0045],
        ["ZnF2","mp-1873",8.14,7.4039],
        ["ZnS","mp-560588",9.10,8.3246],
        ["LaCl3","mp-22896",10.22,8.5647],
        ["Ga2Se3","mp-1340",12.03,10.9548],
        ["Cr2O3","mp-19399",11.05,12.8334],
        ["AsF3","mp-28027",5.24,5.7034],
        ["SnS2","mp-9984",12.48,13.8649],
        ["PI3","mp-27529",4.11,3.6650],
        ["BaSe","mp-1253",14.13,10.7041],
        ["SrSe","mp-2758",11.89,8.5041],
        ["GaS","mp-2507",7.57,8.6351],
        ["As2Se3","mp-909",10.41,13.4052],
        ["SiC","mp-11714",10.54,9.7053],
        ["InSe","mp-22691",9.68,7.5354],
        ["HCl","mp-632326",2.77,4.0055],
        ["TlF","mp-558134",39.49,35.0056],
        ["FeS2","mp-226",28.24,24.1157],
        ["KMgF3","mp-3448",6.88,6.9758],
        ["KBrO3","mp-22958",7.12,7.3034],
        ["KMnF3","mp-555123",13.39,9.7558],
        ["KMnF3","mp-555359",9.96,9.7558],
        ["AlCuS2","mp-4979",8.68,7.7859],
        ["Cd(GaS2)2","mp-4452",9.52,11.4060],
        ["ZnSiP2","mp-4763",12.42,11.5260],
        ["AlPO4","mp-7848",3.78,6.0534],
        ["GaCuS2","mp-5238",11.55,9.5361],
        ["ZnSnP2","mp-4175",15.58,10.0060],
        ["BaSnO3","mp-3163",22.51,18.0034],
        ["Cd(GaSe2)2","mp-3772",11.06,9.2060],
        ["NaNO2","mp-2964",5.27,6.3562],
        ["GaAgS2","mp-5342",10.49,8.4160],
        ["BiTeI","mp-22965",23.74,14.5060],
    ]

    calc_vs_exp_plot(data, model_name="GCN")
