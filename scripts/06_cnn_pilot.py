"""ExoSAN CLEAN: tiny dual-branch 1D CNN, 5-fold pilot (not final test)."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('planet', 'false_positive', 'eclipsing_binary',
           'stellar_variability', 'quiet')


def load():
    with (ROOT / 'data/catalogs/pilot50_v1.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 50 or any(sum(r['class_label'] == label for r in rows) != 10
                              for label in CLASSES):
        raise ValueError('Expected 10 stars per class in frozen pilot')
    gs, ls, fs, labels = [], [], [], []
    for row in rows:
        kic = row['kepid']
        with np.load(ROOT / f'data/features/bls_v2/{kic}.npz') as z:
            g, l = np.asarray(z['global_views'], 'f4'), np.asarray(z['local_views'], 'f4')
        if g.shape != (3, 2, 401) or l.shape != (3, 2, 161):
            raise ValueError(f'KIC {kic}: wrong BLS view shapes')
        for view in (g, l):
            view[:, 0, :] = np.clip(view[:, 0, :] * 1000, -20, 20)
            view[:, 1, :] = (view[:, 1, :] > 0).astype('f4')
        gs.append(g.reshape(6, 401)); ls.append(l.reshape(6, 161))
        with np.load(ROOT / f'data/features/fourier_v3/{kic}.npz') as z:
            f = np.asarray(z['feature_vector'], 'f4')
        if f.shape != (48,) or not np.isfinite(f).all():
            raise ValueError(f'KIC {kic}: invalid Fourier features')
        fs.append(f); labels.append(CLASSES.index(row['class_label']))
    return rows, np.stack(gs), np.stack(ls), np.stack(fs), np.array(labels)


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(6, 16, 7, padding=3), nn.ReLU(),
                                    nn.MaxPool1d(4), nn.Conv1d(16, 32, 5, padding=2),
                                    nn.ReLU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class ExoCNN(nn.Module):
    def __init__(self, fusion):
        super().__init__()
        self.g, self.l, self.fusion = Branch(), Branch(), fusion
        self.fft = nn.Sequential(nn.Linear(48, 16), nn.ReLU()) if fusion else None
        self.head = nn.Sequential(nn.Linear(64 + 16 * fusion, 48), nn.ReLU(),
                                  nn.Dropout(.2), nn.Linear(48, 5))

    def forward(self, g, l, f):
        chunks = [self.g(g), self.l(l)]
        if self.fusion:
            chunks.append(self.fft(f))
        return self.head(torch.cat(chunks, dim=1))


def cross_validate(rows, g, l, f, y, mode, epochs):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    guesses = np.full(len(y), -1, dtype=int)
    torch.set_num_threads(2)
    for fold, (tr, va) in enumerate(cv.split(g, y), 1):
        torch.manual_seed(42 + fold)
        scaler = StandardScaler().fit(f[tr])  # no validation leakage
        ft = torch.from_numpy(np.clip(scaler.transform(f), -5, 5).astype('f4'))
        gt, lt = torch.from_numpy(g), torch.from_numpy(l)
        yt = torch.from_numpy(y).long()
        model = ExoCNN(mode == 'fusion')
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        train_ids = torch.as_tensor(tr)
        for epoch in range(epochs):
            model.train()
            order = train_ids[torch.randperm(len(train_ids))]
            for group in order.split(8):
                loss = nn.functional.cross_entropy(model(gt[group], lt[group], ft[group]),
                                                   yt[group])
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            guesses[va] = model(gt[va], lt[va], ft[va]).argmax(dim=1).numpy()
        print(f'{mode:6s} fold {fold}: macro-F1={f1_score(y[va], guesses[va], average="macro", zero_division=0):.3f}')
    print(f'{mode}: OOF balanced accuracy={balanced_accuracy_score(y, guesses):.3f}; '
          f'macro-F1={f1_score(y, guesses, average="macro", zero_division=0):.3f}')
    return [(mode, r['kepid'], r['class_label'], CLASSES[p])
            for r, p in zip(rows, guesses)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--mode', choices=('both', 'bls', 'fusion'), default='both')
    args = parser.parse_args()
    rows, g, l, f, y = load()
    all_preds = []
    modes = ('bls', 'fusion') if args.mode == 'both' else (args.mode,)
    for mode in modes:
        all_preds.extend(cross_validate(rows, g, l, f, y, mode, args.epochs))
    report = ROOT / 'reports/pilot50_cnn_predictions.csv'
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(('mode', 'kepid', 'true_label', 'predicted_label'))
        writer.writerows(all_preds)
    print('Saved:', report)


if __name__ == '__main__':
    main()
