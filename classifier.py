import uproot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict

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
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], output_dim=3):
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
    root_file = 'merged.root'  # path to your ROOT file
    tree_name = 'myTree'
    feature_branches = ['dR_TPlusKstar', 'eta_B0', 'invMassB0', 'invMassTT', 'invMassKstarTPlus', 'invMassKstarTMinus', 'pt_B0', 'pointingCos', 'transFlightLength', 'vertexChi2']  # replace with your feature names 'vertexChi2'
    batch_size = 256
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Training loop ---
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, per_class_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}: Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "f"Per‑class Acc={per_class_acc}")

    # --- Save trained model ---
    torch.save(model.state_dict(), 'mlp_classifier.pt')
    print("Training complete. Model saved to 'mlp_classifier.pt'.")



if __name__ == '__main__':
    main()
