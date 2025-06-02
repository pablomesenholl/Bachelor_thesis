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
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Main Evaluation and ROC Plotting ---
def main():
    # Parameters
    root_file = 'merged.root'
    tree_name = 'myTree'
    feature_branches = [
        'dR_TPlusTMinus', 'eta_B0',
        'flightLength3D', 'invMassB0',
        'pointingCos', 'vertexChi2'
    ] # 'vertexChi2'
    batch_size = 256
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
    plt.plot(fpr['micro'], tpr['micro'], label=f"Micro-avg (AUC = {roc_auc['micro']:.2f})", linestyle=':')
    plt.plot(fpr['macro'], tpr['macro'], label=f"Macro-avg (AUC = {roc_auc['macro']:.2f})", linestyle='--')
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
