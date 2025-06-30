import ROOT
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
pd.set_option('display.multi_sparse', False)
os.makedirs("Correlations", exist_ok=True)


def plot_corr_heatmap(
        corr_mat,
        title="",
        filename=None,
        size=(10, 8),
        dpi=300,
        cmap="vlag"
):
    # Mask only the strictly upper triangle—keep the diagonal
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)

    # Create figure & axes
    fig, ax = plt.subplots(figsize=size)

    # Draw the heatmap
    sns.heatmap(
        corr_mat,
        mask=mask,
        annot=True,
        fmt=".2f",
        linewidths=.5,
        cbar_kws={"shrink": .8},
        cmap=cmap,
        square=True,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Save or show
    if filename:
        outpath = os.path.join("Correlations", filename)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved heatmap to {filename}")
    else:
        plt.tight_layout()
        plt.show()

def rank_correlations(corr_mat, top_n=None, absolute=True):
    # Mask out the lower triangle and the diagonal
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    # Extract the upper triangle values
    pairs = (
        corr_mat.where(mask)
                 .stack()           # turns df into Series with MultiIndex (i, j)
    )
    # Sort
    if absolute:
        pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index)
    else:
        pairs = pairs.sort_values(ascending=False)

    if top_n is not None:
        return pairs.head(top_n)
    return pairs

file0 = uproot.open("Signal_features_triggered.root")
file1 = uproot.open("Spec_Bckg_features_triggered.root")
file2 = uproot.open("Comb_Bckg_features_triffered.root")
tree0 = file0["tree"]
tree1 = file1["tree"]
tree2 = file2["tree"]


branches = ["dR_TMinusKstar", "dR_TPlusKstar", "dR_TPlusTMinus", "flightLength3D", "invMassB0",
            "invMassKstarTPlus", "invMassKstarTMinus", "invMassTT", "m_kst", "pointingCos",
            "transFlightLength", "vertexChi2", "eta_B0", "pt_B0", "B0_t_xy"]

arrays0 = tree0.arrays(branches, library="np")
arrays1 = tree1.arrays(branches, library="np")
arrays2 = tree2.arrays(branches, library="np")

df0 = pd.DataFrame(arrays0)
df1 = pd.DataFrame(arrays1)
df2 = pd.DataFrame(arrays2)
df3 = pd.concat([df0, df1, df2], axis=0)
df01 = pd.concat([df0, df1], axis=0)
df02 = pd.concat([df0, df2], axis=0)

#correlation matrices
corr_matrix_0 = df0.corr()
corr_matrix_1 = df1.corr()
corr_matrix_2 = df2.corr()

for name, cm in {
    "Signal": corr_matrix_0,
    "Spec_Background": corr_matrix_1,
    "Comb_Background": corr_matrix_2
}.items():
    print(f"\n=== Top correlations in {name} ===")
    print(rank_correlations(cm, top_n=10))

plot_corr_heatmap(
    corr_matrix_0,
    title="Signal Correlation Matrix",
    filename="signal_corr.png"
)
plot_corr_heatmap(
    corr_matrix_1,
    title="Specific Background Correlation Matrix",
    filename="spec_bkg_corr.png"
)
plot_corr_heatmap(
    corr_matrix_2,
    title="Combinatorial Background Correlation Matrix",
    filename="comb_bkg_corr.png"
)

#compute mean and variance of dataframes
mean_0 = df0.mean(axis=0)
var_0 = df0.var(axis=0, ddof=1)

mean_1 = df1.mean(axis=0)
var_1 = df1.var(axis=0, ddof=1)

mean_2 = df2.mean(axis=0)
var_2 = df2.var(axis=0, ddof=1)

mean_3 = df3.mean(axis=0)
var_3 = df3.var(axis=0, ddof=1)

mean_01 = df01.mean(axis=0)
var_01 = df01.var(axis=0, ddof=1)

mean_02 = df02.mean(axis=0)
var_02 = df02.var(axis=0, ddof=1)

#fisher scores between signal and spec bckg
fisher_scores_01 = {}
for feature in branches:
    n_0 = len(df0[feature])
    n_1 = len(df1[feature])
    mu0 = mean_0[feature]
    mu1 = mean_1[feature]
    mu_total = mean_01[feature]
    v = var_01[feature]
    denom = v if v>0 else 1e-8 #protect against division by zero
    fisher_scores_01[feature] =( ( n_0*((mu0 - mu_total)**2) ) + ( n_1*((mu1 - mu_total)**2) ) )/ denom

fisher_df_01 = pd.Series(fisher_scores_01).sort_values(ascending=False) #sort fisher score from highest to lowest
print("Fisher scores between 0 and 1 (highest→lowest):")
print(fisher_df_01)

#fisher score between signal and comb bckg
fisher_scores_02 = {}
for feature in branches:
    n_0 = len(df0[feature])
    n_2 = len(df2[feature])
    mu0 = mean_0[feature]
    mu2 = mean_2[feature]
    mu_total = mean_02[feature]
    v = var_02[feature]
    denom = v if v>0 else 1e-8 #protect against division by zero
    fisher_scores_02[feature] = (( n_0*((mu0 - mu_total)**2) ) + ( n_2*((mu2 - mu_total)**2) )) / denom

fisher_df_02 = pd.Series(fisher_scores_02).sort_values(ascending=False) #sort fisher score from highest to lowest
print("Fisher scores between 0 and 2 (highest→lowest):")
print(fisher_df_02)

#fisher scores between all classes
fisher_scores_3 = {}
for feature in branches:
    n_0 = len(df0[feature])
    n_1 = len(df1[feature])
    n_2 = len(df2[feature])
    mu0 = mean_0[feature]
    mu1 = mean_1[feature]
    mu2 = mean_2[feature]
    mu_total = mean_3[feature]
    v = var_3[feature]
    denom = v if v>0 else 1e-8 #protect against division by zero
    fisher_scores_3[feature] = (( n_0*((mu0 - mu_total)**2) ) + ( n_1*((mu1 - mu_total)**2) ) + ( n_2*((mu2 - mu_total)**2) )) / denom

fisher_df_3 = pd.Series(fisher_scores_3).sort_values(ascending=False) #sort fisher score from highest to lowest
print("Fisher scores between all three classes (highest→lowest):")
print(fisher_df_3)