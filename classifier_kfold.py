import uproot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from collections import Counter

class RootFeatureDataset(Dataset):
    """
    PyTorch Dataset for loading precomputed features from a ROOT file.

    Expects a ROOT TTree with branches for each feature and a branch 'label' with integer class labels:
    0 = signal, 1 = specific background, 2 = combinatorial background.
    """
    def __init__(self, root_file, tree_name, feature_branches, label_branch='label', transform=None):
        self.file = uproot.open(root_file)
        self.tree = self.file[tree_name]
        self.features = np.vstack([
            self.tree[branch].array(library='np')
            for branch in feature_branches
        ]).T
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
    def __init__(self, input_dim, hidden_dims, output_dim=3, dropout_prob=0.5):
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

# Training and evaluation functions

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device, num_classes=3):
    model.eval()
    total_loss, correct = 0.0, 0
    class_correct = Counter()
    class_counts = Counter()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * X.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            for t, p in zip(y, preds):
                class_counts[t.item()] += 1
                if t == p:
                    class_correct[t.item()] += 1
    acc = correct / len(loader.dataset)
    loss = total_loss / len(loader.dataset)
    per_class = {c: class_correct[c]/class_counts[c] for c in class_counts}
    return loss, acc, per_class


def cross_validate(dataset, hyperparams, n_splits=5):
    X, y = dataset.features, dataset.labels
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold_accs = []

    for train_idx, val_idx in skf.split(X, y):
        train_ds = Subset(dataset, train_idx)
        val_ds   = Subset(dataset, val_idx)
        train_loader = DataLoader(train_ds, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=hyperparams['batch_size'])

        model = MLPClassifier(
            input_dim=X.shape[1],
            hidden_dims=hyperparams['hidden_dims'],
            output_dim=len(np.unique(y)),
            dropout_prob=hyperparams['dropout_prob']
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay']
        )

        # Training loop
        for _ in range(hyperparams['epochs']):
            train(model, train_loader, criterion, optimizer, device)

        # Validation
        _, val_acc, _ = evaluate(model, val_loader, criterion, device)
        fold_accs.append(val_acc)

    return np.mean(fold_accs)


def grid_search(root_file, tree_name, feature_branches):
    dataset = RootFeatureDataset(root_file, tree_name, feature_branches)

    # Define hyperparameter grid
    param_grid = {
        'hidden_dims': [[32,16], [64,32,16], [128,64,32]],
        'dropout_prob': [0.2, 0.5],
        'lr': [1e-3, 1e-4],
        'weight_decay': [1e-4, 1e-5],
        'batch_size': [128, 256],
        'epochs': [20, 30]
    }

    best_score = 0.0
    best_params = None
    # Iterate combinations
    for params in ParameterGrid(param_grid):
        print(f"Testing params: {params}")
        acc = cross_validate(dataset, params, n_splits=5)
        print(f"Mean CV accuracy: {acc:.4f}\n")
        if acc > best_score:
            best_score = acc
            best_params = params

    print(f"Best CV accuracy {best_score:.4f} with params: {best_params}")
    return best_params


def train_final(root_file, tree_name, feature_branches, best_params):
    # Retrain on full dataset with best hyperparameters
    dataset = RootFeatureDataset(root_file, tree_name, feature_branches)
    loader = DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLPClassifier(
        input_dim=dataset.features.shape[1],
        hidden_dims=best_params['hidden_dims'],
        output_dim=len(np.unique(dataset.labels)),
        dropout_prob=best_params['dropout_prob']
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']
    )

    for epoch in range(1, best_params['epochs']+1):
        loss, acc = train(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")

    path = 'mlp_best.pt'
    torch.save(model.state_dict(), path)
    print(f"Saved final model with best hyperparameters to '{path}'")

if __name__ == '__main__':
    root_file = 'merged.root'
    tree_name  = 'myTree'
    feature_branches = [
        'dR_TPlusKstar','dR_TMinusKstar','dR_TPlusTMinus',
        'm_kst','invMassB0','invMassTT',
        'invMassKstarTPlus','invMassKstarTMinus',
        'pt_B0','pointingCos','transFlightLength',
        'vertexChi2','eta_B0'
    ]

    best_params = grid_search(root_file, tree_name, feature_branches)
    train_final(root_file, tree_name, feature_branches, best_params)
