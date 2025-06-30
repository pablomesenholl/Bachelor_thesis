import ROOT
import uproot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
seed = 40
np.random.seed(seed) #set seed for bootstrapping

# --- Dataset Definition ---
class RootFeatureDataset(Dataset):
    """
    PyTorch Dataset for loading precomputed features from a ROOT file.
    Expects a ROOT TTree with branches for each feature and a branch 'label' with integer class labels:
    0 = signal, 1 = specific background, 2 = combinatorial background.
    """
    def __init__(self, root_file, tree_name, feature_branches, label_branch='label', transform=None):
        self.file = uproot.open(root_file)
        self.tree = self.file[tree_name]
        self.features = np.vstack([self.tree[branch].array(library='np')
                                   for branch in feature_branches]).T
        self.labels = self.tree[label_branch].array(library='np').astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return torch.from_numpy(x).float(), torch.tensor(y)

# --- Model Definition ---
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=3, dropout_prob = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def bootstrap_efficiencies_at_cut(y_true, y_score, weights, cut, n_boot=500):
    """
    Bootstrap the weighted efficiency for each class at a fixed cut.

    Parameters:
    - y_true: (N,) array of true labels (0=signal, 1=specific bkg, 2=comb bkg)
    - y_score: (N,3) array of predicted probabilities per class
    - weights: (N,) array of event weights
    - cut: float threshold on y_score[:, 0] (P(signal))
    - n_boot: number of bootstrap replicas

    Returns:
    - results: dict mapping class -> (mean_eff, err_low, err_high)
    """
    N = len(y_true)
    effs = np.zeros((n_boot, 3))

    for b in range(n_boot):
        idx = np.random.randint(0, N, size=N)
        y_b = y_true[idx]
        w_b = weights[idx]
        p0_b = y_score[idx, 0]

        for c in (0, 1, 2):
            mask_c = (y_b == c)
            mask_pass = mask_c & (p0_b > cut)
            total_w = w_b[mask_c].sum()
            pass_w = w_b[mask_pass].sum()
            effs[b, c] = pass_w / total_w if total_w > 0 else 0.0

    results = {}
    for c in (0, 1, 2):
        arr = effs[:, c]
        mu = arr.mean()
        std = arr.std()
        results[c] = (mu, std)

    return results

# --- Main Evaluation and ROC Plotting ---
def main():
    # Parameters
    root_file = 'merged_triggered.root'
    tree_name = 'myTree'
    feature_branches = [
        'dR_TPlusKstar', 'dR_TMinusKstar', 'dR_TPlusTMinus', 'm_kst', 'invMassB0', 'invMassTT', 'invMassKstarTPlus', 'invMassKstarTMinus', 'pt_B0', 'pointingCos', 'transFlightLength', 'vertexChi2', 'eta_B0'
    ] # 'vertexChi2'
    batch_size = 128
    model_path = 'mlp_classifier.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and DataLoader
    dataset = RootFeatureDataset(root_file, tree_name, feature_branches)
    loader = DataLoader(dataset, batch_size=batch_size)

    # Load model with safe unpickling
    model = MLPClassifier(input_dim=len(feature_branches)).to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Gather true labels and scores
    y_true = []
    y_score = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            y_true.extend(y.numpy())
            y_score.extend(probs.cpu().numpy())
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # plot the predicted probability distribution for signal vs other
    n_classes = 3
    class_names = ['Signal', 'Specific Bkg', 'Comb. Bkg']

    # compute actual weights
    PS = 35 #prescale factor
    L = 40e3  # luminosity in inverse pico barn
    sigma_bb = 4.4670e08  # bb cross section in pb
    sigma_cc = 8.0040e09  # cc cross section in pb
    f = 0.404  # hadronization fraction of bb into B^0
    eff_trigger_signal = 2.8e-3  # efficiencies trigger
    eff_trigger_spec = 5.5e-4
    eff_trigger_comb = 1.512e-3
    eff_gen_signal = 10000 / 125885  # efficiencies generator
    eff_gen_spec = 20000 / 61142
    eff_gen_comb = 6615 / 1200000
    BR_signal = 1.0e-07  # BR of signal
    BR_tau_hadr = 0.1825  # BR of TAUOLA 5
    BR_tau_muonic = 0.1739  # BR of TAUOLA 2
    BR_Kstar = 2.0 / 3.0  # BR of Kstar to K^+ pi^-
    BR_D = 9.8e-3  # BR of B^0 to D^-
    BR_Dstar = 0.0148  # BR of B^0 to D^*-
    BR_D_semilept = 0.0527  # BR of D^- to Kstar and muon
    BR_Dstar_hadr = 0.307  # BR of D^*- to D^- and pi^0
    BR_tau_3prong = 0.0931  # BR of tau into three charged pions
    BR_eff_sig = 2 * BR_signal * BR_tau_hadr * BR_tau_muonic * BR_Kstar  # effective BR of signal
    BR_eff_spec = (BR_D * BR_D_semilept * BR_tau_3prong * BR_Kstar) + (
            BR_Dstar * BR_Dstar_hadr * BR_D_semilept * BR_tau_3prong * BR_Kstar)
    N_sig = L * sigma_bb * BR_eff_sig * eff_gen_signal * eff_trigger_signal
    N_spec = L * sigma_bb * BR_eff_spec * eff_gen_spec * eff_trigger_spec
    N_comb = L * (sigma_bb + sigma_cc) * eff_gen_comb * eff_trigger_comb
    N_tot = N_sig + N_spec + N_comb
    counts = np.array([N_sig, N_spec, N_comb])
    true_priors = counts / counts.sum()
    test_counts = np.bincount(y_true, minlength=3)
    test_priors = test_counts / test_counts.sum()
    weights = true_priors[y_true] / test_priors[y_true] #per events weights

    #compute efficiency
    signal_mask = (y_true == 0)
    bckg_mask = (y_true == 1) | (y_true == 2)
    n_sig_total = np.sum(signal_mask)
    n_bckg_total = np.sum(bckg_mask)

    cut_value_array = np.linspace(0, 1, 10000)
    def efficiency(cuts):
        eff_sig = []
        eff_bckg = []
        for cut in cuts:
            eff_sig.append(np.sum(y_score[signal_mask, 0] > cut) / n_sig_total)
            eff_bckg.append(np.sum(y_score[bckg_mask, 0] > cut) / n_bckg_total)
        return np.array(eff_sig), np.array(eff_bckg)

    eff, eff_bckg = efficiency(cut_value_array)

    def weighted_efficiencies(cuts):
        eff_sig = []
        eff_bkg = []
        for c in cuts:
            sig_sel = (y_true == 0) & (y_score[:, 0] > c)
            bkg_sel = (y_true != 0) & (y_score[:, 0] > c)

            # weighted sums
            w_sig_pass = weights[sig_sel].sum()
            w_sig_tot = weights[y_true == 0].sum()
            w_bkg_pass = weights[bkg_sel].sum()
            w_bkg_tot = weights[y_true != 0].sum()

            eff_sig.append(w_sig_pass / w_sig_tot)
            eff_bkg.append(w_bkg_pass / w_bkg_tot)

        return np.array(eff_sig), np.array(eff_bkg)

    eff_sig_w, eff_bkg_w = weighted_efficiencies(cut_value_array)
    N_bkg_true = N_spec + N_comb

    B_MIN = 1.0
    punzi_a = 3.0


    # Expected event yields at each cut
    s_exp = eff_sig_w * N_sig  # weighted signal yield
    b_exp = eff_bkg_w * N_bkg_true  # weighted background yield

    # first try punzi
    punzi_vals = np.full_like(b_exp, np.nan)
    valid = b_exp >= B_MIN
    punzi_vals[valid] = eff_sig_w[valid] / (punzi_a / 2.0 + np.sqrt(b_exp[valid]))
    idx_opt_punzi = np.nanargmax(punzi_vals)
    opt_cut_punzi = cut_value_array[idx_opt_punzi]
    opt_Punzi = punzi_vals[idx_opt_punzi]
    print(f"Punzi-optimal cut = {opt_cut_punzi}")
    print(f"  Punzi  = {opt_Punzi}")
    print(f"  ε_sig(w) = {eff_sig_w[idx_opt_punzi]:.4f}")
    print(f"  ε_bkg(w) = {eff_bkg_w[idx_opt_punzi]:.4f}  "
          f"(→ B_exp = {b_exp[idx_opt_punzi]:.2f} events)")

    def asimov_significance(s, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            term = (s + b) * np.log1p(s / b) - s
        term = np.where(term > 0, term, 0.)
        return np.sqrt(2 * term)
    Z_asimov = np.full_like(b_exp, np.nan)
    valid = b_exp >= B_MIN
    Z_asimov[valid] = asimov_significance(s_exp[valid], b_exp[valid])
    idx_opt_asimov = np.nanargmax(Z_asimov)
    opt_cut_asimov = cut_value_array[idx_opt_asimov]
    opt_ZA = Z_asimov[idx_opt_asimov]
    print(f"Weighted-Asimov optimum cut = {opt_cut_asimov}, Asimov = {opt_ZA}")
    print(f"Selection efficiency of weighted signal vs. background at P(signal)>{opt_cut_asimov:.3f}: efficiency signal = ", eff_sig_w[idx_opt_asimov], "efficiency background = ", eff_bkg_w[idx_opt_asimov])
    print(f"Background events that survive asimov opt cut: B_exp = {b_exp[idx_opt_asimov]:.2f} events")

    results = bootstrap_efficiencies_at_cut(y_true, y_score, weights, opt_cut_asimov, n_boot=1000)
    for cls, (mu, std) in results.items():
        print(f"Efficiency of classifier for Class {cls}: ε = {mu:.5f} +-{std:.5f}")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(cut_value_array, punzi_vals,
             color='darkorange', lw=1.2)
    ax1.set_xlabel(r'cut on $p_\mathrm{sig}$', fontsize=12)
    ax1.set_ylabel('Punzi significance $s$', color='darkorange', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='darkorange')
    ax1.axvline(opt_cut_punzi, color='black', ls='--', lw=0.8, label='Optimal cut at $p_\mathrm{sig}$ = 0.965')
    ax2 = ax1.twinx()
    ax2.plot(cut_value_array, Z_asimov,
             color='navy', lw=1.2, label='Asimov $Z$')
    ax2.set_ylabel('Asimov significance $Z$', color='navy', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='navy')
    ax2.axvline(opt_cut_asimov, color='navy', ls='--', lw=0.8)
    ax2.set_ylim(0.0003, 0.0017)
    ax1.grid(alpha=0.3)
    ax1.legend()
    fig.tight_layout()
    fig.savefig("Result_plots/Punzi_vs_Asimov.png")

    #Plot optimal cut in efficiency
    plt.figure()
    # plt.plot(cut_value_array, eff, label="Signal Efficiency")
    # plt.plot(cut_value_array, eff_bckg, label="Background Efficiency")
    # plt.scatter(opt_cut_punzi, eff_sig_w[idx_opt_punzi], color='red', label=f"punzi-opt cut={opt_cut_punzi:.3f}")
    plt.axvline(opt_cut_asimov, color='black', ls=':', label='0.965 cut')
    plt.plot(cut_value_array, eff_sig_w, label="Weighted Signal Efficiency", color='steelblue')
    plt.plot(cut_value_array, eff_bkg_w, label="Weighted Background Efficiency", color='green')
    # plt.scatter(opt_cut_asimov, eff_sig_w[idx_opt_asimov], color='blue', label=f"asimov-opt cut={opt_cut_asimov:.3f}")
    plt.ylabel("Efficiency")
    plt.xlabel("Cut in $p_0$")
    plt.legend()
    plt.title("Efficiency as a function of cut in $p_0$")
    plt.grid(True)
    plt.savefig("Result_plots/efficiency_plot.png")

    print("Selection efficiency of signal vs. background at P(signal)>0.925: efficiency signal = ", eff_sig_w[idx_opt_punzi], "efficiency background = ", eff_bkg_w[idx_opt_punzi])

    #Plot probability to be signal
    plt.figure()
    for true_class in range(n_classes):
        mask = (y_true == true_class)
        plt.hist(
            y_score[mask, 0],  # predicted P(signal) for events of this true class
            bins=50,
            alpha=0.5,
            density=True,
            label=f'True = {class_names[true_class]}'
        )
    plt.xlabel('Predicted Probability to be Signal $p_0$')
    plt.ylabel('Arbitrary Units')
    plt.title('Distribution of $p_0$ for Signal and Background')
    plt.xlim(0, 1)
    plt.yscale('log')
    plt.axvline(0.965, label='0.965 cut')
    plt.legend(loc='upper center')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Result_plots/prob_dist_signal.png')

    # Binarize labels for one-vs-rest ROC
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average
    fpr['micro'], tpr['micro'], _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Compute macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'], tpr['macro'] = all_fpr, mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    # Plotting ROC curve
    plt.figure(figsize=(8, 6))
    colors = ['navy', 'darkorange', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})", linewidth=2)
    # plt.plot(fpr['micro'], tpr['micro'], label=f"Micro-avg (AUC = {roc_auc['micro']:.2f})", linestyle=':')
    # plt.plot(fpr['macro'], tpr['macro'], label=f"Macro-avg (AUC = {roc_auc['macro']:.2f})", linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 3-Class MLP')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Result_plots/ROC_curves.png')

    #classifier efficiencies at the optimal cut:
    mask_sig = (y_true == 0)
    mask_spec = (y_true == 1)
    mask_comb = (y_true == 2)
    mask_bkg = (y_true != 0)

    eff_cls_sig = weights[mask_sig & (y_score[:, 0] > opt_cut_asimov)].sum() / weights[mask_sig].sum()
    eff_cls_spec = weights[mask_spec & (y_score[:, 0] > opt_cut_asimov)].sum() / weights[mask_spec].sum()
    eff_cls_comb = weights[mask_comb & (y_score[:, 0] > opt_cut_asimov)].sum() / weights[mask_comb].sum()
    eff_cls_bkg = weights[mask_bkg & (y_score[:, 0] > opt_cut_asimov)].sum() / weights[mask_bkg].sum()

    # eff_cls_sig = np.sum(y_score[mask_sig, 0] > opt_cut) / np.sum(mask_sig)
    # eff_cls_spec = np.sum(y_score[mask_spec, 0] > opt_cut) / np.sum(mask_spec)
    # eff_cls_comb = np.sum(y_score[mask_comb, 0] > opt_cut) / np.sum(mask_comb)
    print(f"Classifier efficiencies after cut for signal, spec-, comb. background and total background: ", eff_cls_sig, eff_cls_spec, eff_cls_comb, eff_cls_bkg)

    #calculate master formula
    PS = 35 #prescale factor
    L = 40e3 #luminosity in inverse pico barn
    sigma_bb = 4.4670e08 #bb cross section in pb
    sigma_cc = 8.0040e09 #cc cross section in pb
    sigma_tot = sigma_bb + sigma_cc #total cross section in pb for comb. bckg
    f = 0.404 #hadronization fraction of bb into B^0
    BR_signal = 1.0e-07 #BR of signal
    BR_tau_hadr = 0.1825 #BR of TAUOLA 5
    BR_tau_muonic = 0.1739 #BR of TAUOLA 2
    BR_Kstar = 2.0/3.0 #BR of Kstar to K^+ pi^-
    eff_trigger_signal = 2.8e-3 #efficiencies trigger
    eff_trigger_spec = 5.5e-4
    eff_trigger_comb = 1.512e-3
    eff_gen_signal = 10000/125885 #efficiencies generator
    eff_gen_spec = 20000/61142
    eff_gen_comb = 6615/1200000
    BR_eff_sig = 2 * BR_signal * BR_tau_hadr * BR_tau_muonic * BR_Kstar #effective BR of signal
    print("Effective Branching ratio signal: ", BR_eff_sig)


    N_signal = L * sigma_bb * BR_eff_sig * eff_cls_sig * eff_trigger_signal * eff_gen_signal
    print("N_signal = ", N_signal)

    # N_background calculation
    BR_D = 9.8e-3 #BR of B^0 to D^-
    BR_Dstar = 0.0148 #BR of B^0 to D^*-
    BR_D_semilept = 0.0527 #BR of D^- to Kstar and muon
    BR_Dstar_hadr = 0.307 #BR of D^*- to D^- and pi^0
    BR_tau_3prong = 0.0931 #BR of tau into three charged pions
    BR_eff_spec = (BR_D*BR_D_semilept*BR_tau_3prong*BR_Kstar) + (BR_Dstar*BR_Dstar_hadr*BR_D_semilept*BR_tau_3prong*BR_Kstar)
    print("Effective Branching ratio specific bckg: ", BR_eff_spec)

    N_spec_bckg = L * sigma_bb * BR_eff_spec * eff_trigger_spec * eff_gen_spec * eff_cls_spec
    N_comb_bckg = L * sigma_tot * eff_trigger_comb * eff_gen_comb * eff_cls_comb
    N_bckg = N_spec_bckg + N_comb_bckg
    print("N_spec = ", N_spec_bckg)
    print("N_comb = ", N_comb_bckg)
    print("N_bckg  = ", N_bckg)


    #simplyfied determination of limit BR
    sigma = 5
    BR = (sigma * np.sqrt(N_bckg))/(L * sigma_bb * eff_cls_sig * eff_trigger_signal * eff_gen_signal)
    print("simplyfied BR limit = ", BR)

    #Z-value for SM BR
    def Z(BR_z):
        z = (L * sigma_bb * BR_z * eff_cls_sig * eff_trigger_signal * eff_gen_signal)/(np.sqrt(N_bckg))
        return z
    brs = np.logspace(-9, -1, 1000)
    z_values = []
    for br in brs:
        z = Z(br)
        z_values.append(z)
    z_values = np.array(z_values)
    idx_z = np.argmin(np.abs(z_values - 5))
    print("BR limit for z = 5: ", brs[idx_z])

    #asimov significance
    def asimov(s, b):
        z = np.sqrt(2 * ( (s + b) * np.log(1 + (s/b)) - s ))
        return z

    brs = np.logspace(-9, -1, 1000)
    asimov_values = []
    for br in brs:
        s = L * sigma_bb * (br/1.0e-7) * eff_cls_sig * eff_trigger_signal * eff_gen_signal
        asim = asimov(s, N_bckg)
        asimov_values.append(asim)
    asimov_values = np.array(asimov_values)
    idx_asimov = np.argmin(np.abs(asimov_values - 5))
    print("BR limit for asimov = 5: ", brs[idx_asimov])

    #try CLs method
    from scipy.stats import poisson
    def compute_cls(N_obs, s, b):
        p_sb = poisson.cdf(N_obs, s + b)
        p_b = poisson.cdf(N_obs, b)
        if p_b == 0:
            return 1.0
        return p_sb / p_b

    brs = np.logspace(-7, -4, 10000)
    cls_values = []
    for br in brs:
        s = L * sigma_bb * (br/1.0e-7) * eff_cls_sig * eff_trigger_signal * eff_gen_signal
        cls = compute_cls(N_bckg, s, N_bckg)
        cls_values.append(cls)

    for br, cls in zip(brs, cls_values):
        if cls < 0.05:
            print(f"95% CL sensitivity reach: BR > {br:.2e}")
            break

    plt.figure()
    plt.plot(brs, cls_values, label='CL$_s$(BR)')
    plt.axhline(0.05, color='red', linestyle='--', label='95% CL threshold')
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Branching Ratio (BR)', fontsize=14)
    plt.ylabel('CL$_s$', fontsize=14)
    plt.title('Sensitivity to $B^0 \\to K^* \\tau^+ \\tau^-$ with CMS', fontsize=15)
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Result_plots/CLs vs. BR.png")




if __name__ == '__main__':
    main()
