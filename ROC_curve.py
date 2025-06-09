import uproot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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

def bayes_reweight(p_flat, true_priors):
    """
    p_flat: Tensor [batch,3] from softmax(logits)
    true_priors: list or 1D-tensor [P_sig, P_spec, P_comb]
    returns: Tensor [batch,3] p_true(c|x)
    """
    # make priors a tensor on the right device
    prior = torch.tensor(true_priors, device=p_flat.device).unsqueeze(0)  # shape [1,3]
    unnorm = p_flat * prior            # broadcast → [batch,3]
    return unnorm / unnorm.sum(dim=1, keepdim=True)  # normalize rows

# --- Main Evaluation and ROC Plotting ---
def main():
    # Parameters
    root_file = 'merged.root'
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

    # true prior proportions between classes, need to be adjusted
    true_priors = [1e-7, 0.1, 0.9]

    # Gather true labels and scores
    y_true = []
    y_score = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            p_true = bayes_reweight(probs, true_priors)
            y_true.extend(y.numpy())
            y_score.extend(probs.cpu().numpy())
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    #build weights based on the actual proportions of signal and background classes
    W_sig, W_spec, W_comb = true_priors
    weights = np.where(y_true == 0, W_sig, np.where(y_true == 1, W_spec, W_comb))

    # plot the predicted probability distribution for signal vs other
    n_classes = 3
    class_names = ['Signal', 'Specific Bkg', 'Comb. Bkg']

    #compute efficiency
    signal_mask = (y_true == 0)
    bckg_mask = (y_true == 1) | (y_true == 2)
    cut_value = 0.997
    n_sig_total = np.sum(signal_mask)
    n_sig_pass = np.sum(y_score[signal_mask, 0] > cut_value)
    epsilon_sig = n_sig_pass / n_sig_total
    n_bckg_total = np.sum(bckg_mask)
    n_bckg_pass = np.sum(y_score[bckg_mask, 0] > cut_value)
    epsilon_bckg = n_bckg_pass / n_bckg_total

    cut_value_array = np.linspace(0, 1, 1000)
    def efficiency(cuts):
        eff_sig = []
        eff_bckg = []
        for cut in cuts:
            eff_sig.append(np.sum(y_score[signal_mask, 0] > cut) / n_sig_total)
            eff_bckg.append(np.sum(y_score[bckg_mask, 0] > cut) / n_bckg_total)
        return np.array(eff_sig), np.array(eff_bckg)

    eff, eff_bckg = efficiency(cut_value_array)

    #compute significance array
    signif = []
    for cut in cut_value_array:
        # boolean masks
        pass_sig  = (y_score[:,0] > cut) & (y_true == 0)
        pass_bkg  = (y_score[:,0] > cut) & (y_true > 0)
        # unweighted sums
        S = np.sum(pass_sig)
        B = np.sum(pass_bkg)
        signif.append(S/np.sqrt(B) if (B>0) else 0.0) #compute significance, protect against div by 0
    signif = np.array(signif)

    #find maximum significance cut
    idx_opt = np.argmax(signif)
    opt_cut = cut_value_array[idx_opt]
    pass_sig_opt = (y_score[:, 0] > opt_cut) & (y_true == 0)
    pass_bkg_opt = (y_score[:, 0] > opt_cut) & (y_true > 0)
    # S_opt = np.sum(weights[pass_sig_opt])
    # B_opt = np.sum(weights[pass_bkg_opt])
    print(f"Optimal cut by S/√B = {opt_cut:.3f}")
    # print(f"  -> S = {S_opt}, B = {B_opt}, S/√B = {signif[idx_opt]:.3f}")

    plt.figure()
    plt.plot(cut_value_array, eff, label="Signal Efficiency")
    plt.plot(cut_value_array, eff_bckg, label="Background Efficiency")
    plt.scatter(opt_cut, eff[idx_opt], color='red', label=f"S/√B-opt cut={opt_cut:.3f}")
    plt.ylabel("Efficiency")
    plt.xlabel("Cut value")
    plt.legend()
    plt.grid(True)
    plt.savefig("efficiency_plot.png")

    print("Selection efficiency of signal vs. background at P(signal)>0.997: efficiency signal = ", epsilon_sig, "efficiency background = ", epsilon_bckg)

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
    plt.xlabel('Predicted Probability to be Signal')
    plt.ylabel('Arbitrary Units')
    plt.title('Distribution of P(signal) for Signal and Background')
    plt.xlim(0, 1)
    plt.yscale('log')
    plt.legend(loc='upper center')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('prob_dist_signal.png')

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

    # Plotting
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
    plt.savefig('ROC_curves.png')

if __name__ == '__main__':
    main()
