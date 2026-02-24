# Dielectric Constant & Refractive Index Analysis

This repository contains scripts for analyzing calculated vs. experimental dielectric constants, polycrystalline estimates, and related trends in inorganic compounds. Figures are generated at **1200 dpi** with **bold Times New Roman styling** for publication quality.

## 📂 Contents

- **train_eval_polycrystal.py**  
  Estimates polycrystalline dielectric constants per crystal system using GCN predictions.

- **train_eval_violin.py**  
  Violin plots of dielectric constant distributions across crystal systems.

- **train_eval_bandgap_polycrystal.py**  
  Scatter plots of dielectric constant vs. band gap, including polymers, figures of merit, and highlighted compounds.

- **train_eval_bandgap_polycrystal_labels.py**  
  Improved labeling with annotation lines for selected compounds (e.g., CoSb₃, HfO₂, SiO₂, polyethylene).

- **train_eval_refractive_bandgap.py**  
  Polycrystalline refractive index vs. band gap plots, showing inverse relationship trends across crystal systems.

- **train_eval_calc_vs_exp.py**  
  Comparison of calculated vs. experimental dielectric constants. Includes parity plots, MAD, MARD, RMSE, R², and outlier analysis.

- **train_eval_comprehensive_analysis.py**  
  Unified analysis script producing parity plots, deviation histograms, crystal system boxplots, error vs. band gap scatter, error vs. dielectric constant scatter, and outlier tables.

- **train_eval_family_heatmap_bar.py**  
  Heatmap and stacked bar chart showing average deviation and compound counts per chemical family (halides, oxides, nitrides, chalcogenides, pnictides, etc.).

- **train_eval_remnant_forces.py**  
  Scatter plots showing the effect of remnant interatomic forces on dielectric constant predictions, with color coding by residual force magnitude.

