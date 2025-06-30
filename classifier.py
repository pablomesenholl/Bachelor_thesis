import uproot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

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


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=3, dropout_prob=0.2):
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


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, num_classes=3):
    model.eval()
    total_loss = 0
    correct = 0
    class_correct = Counter()
    class_counts = Counter()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            #per class
            for true, pred in zip(y, preds):
                class_counts[true.item()] += 1
                if true == pred:
                    class_correct[true.item()] += 1
    accuracy = correct / len(loader.dataset)
    val_loss = total_loss / len(loader.dataset)
    #per class accuracy
    per_class_acc = {
        c: class_correct[c] / class_counts[c]
        for c in range(num_classes)
    }
    return val_loss, accuracy, per_class_acc


def main():
    # --- User parameters ---
    root_file = 'merged_triggered.root'  # path to your ROOT file
    tree_name = 'myTree'
    feature_branches = ['dR_TPlusKstar', 'dR_TMinusKstar', 'dR_TPlusTMinus', 'm_kst', 'invMassB0', 'invMassTT', 'invMassKstarTPlus', 'invMassKstarTMinus', 'pt_B0', 'pointingCos', 'transFlightLength', 'vertexChi2', 'eta_B0']  # replace with your feature names 'vertexChi2'
    batch_size = 128
    epochs = 30
    lr = 1e-3

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = RootFeatureDataset(root_file, tree_name, feature_branches)

    # --- Print raw class counts ---
    from collections import Counter
    counts = Counter(dataset.labels.tolist())
    print(f"Total events: {len(dataset)}")
    print(f"  signal (0):                {counts.get(0,0)}")
    print(f"  specific background (1):   {counts.get(1,0)}")
    print(f"  combinatorial background (2): {counts.get(2,0)}")

    #introduce weights to make all classes equally large
    flat_weights = torch.tensor([
        1.0 / counts[0],
        1.0 / counts[1],
        1.0 / counts[2]
    ], dtype=torch.float32)
    flat_weights /= flat_weights.sum() #normalization

    #compute actual weights
    L = 40e3  # luminosity in inverse pico barn
    sigma_bb = 4.4610e08  # bb cross section in pb
    sigma_cc = 7.9070e09  # cc cross section in pb
    f = 0.404  # hadronization fraction of bb into B^0
    eff_trigger_signal = 2.8e-3 #efficiencies trigger
    eff_trigger_spec = 5.5e-4
    eff_trigger_comb = 1.512e-3
    eff_gen_signal = 10000/125885 #efficiencies generator
    eff_gen_spec = 20000/61142
    eff_gen_comb = 6615/1200000
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
    N_sig = L * sigma_bb * f * BR_eff_sig * eff_gen_signal * eff_trigger_signal
    N_spec = L * sigma_bb * f * BR_eff_spec * eff_gen_spec * eff_trigger_spec
    N_comb = L * (sigma_bb + sigma_cc) * eff_gen_comb * eff_trigger_comb
    N_tot = N_sig + N_spec + N_comb
    true_weights = torch.tensor([
        1.0 / (N_sig / N_tot),
        1.0 / (N_spec / N_tot),
        1.0 / (N_comb / N_tot)
    ], dtype=torch.float32)
    true_weights /= true_weights.sum() #normalization

    # Split train / val
    n = len(dataset)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = MLPClassifier(input_dim=len(feature_branches)).to(device)
    criterion = nn.CrossEntropyLoss(weight=flat_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    #initialize lists
    train_losses = []
    val_losses = []
    val_accs = []

    # --- Training loop ---
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, per_class_acc = evaluate(model, val_loader, criterion, device)

        # store for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d}: Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "f"Per‑class Acc={per_class_acc}")

    # --- Plot training progress ---
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy vs. Epoch')

    plt.tight_layout()
    plt.savefig("Result_plots/Loss_vs_epochs.png")

    # --- Save trained model ---
    torch.save(model.state_dict(), 'mlp_classifier.pt')
    print("Training complete. Model saved to 'mlp_classifier.pt'.")



if __name__ == '__main__':
    main()
