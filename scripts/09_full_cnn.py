"""ExoSAN CNN V3: 1024/256-bin views, shape inspection, best validation checkpoint."""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from config import OLD_ROOT
CLASSES = ('planet', 'false_positive', 'eclipsing_binary', 'stellar_variability', 'quiet')
CATALOGS = (ROOT / 'data/catalogs/ml_split_v1.csv', OLD_ROOT / 'data/ml/ml_split_v1.csv',
            OLD_ROOT / 'data/catalogs/processed/ml_split_v1.csv')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPORTS, MODELS = ROOT / 'reports', ROOT / 'models'; G_N, L_N = 1024, 256

def load(split):
    path = next((p for p in CATALOGS if p.exists()), None)
    if path is None: raise FileNotFoundError('Frozen ml_split_v1.csv missing')
    with path.open(encoding='utf-8-sig', newline='') as file:
        rows = [r for r in csv.DictReader(file) if r['split'] == split]
    if len(rows) != {'train': 4200, 'val': 900, 'test': 900}[split]: raise ValueError('Wrong split')
    gs, ls, fs, labels, errors = [], [], [], [], []
    for row in rows:
        k = row['kepid']
        try:
            with np.load(ROOT / f'data/features/views_v3/{k}.npz') as z:
                g, l = z['global_views'].copy(), z['local_views'].copy()
                if str(z['version']) != 'views-v3': raise ValueError('Wrong views version')
            with np.load(ROOT / f'data/features/fourier_v3/{k}.npz') as z:
                f = np.asarray(z['feature_vector'], 'f4')
            if g.shape != (3, 2, G_N) or l.shape != (3, 2, L_N) or f.shape != (48,):
                raise ValueError('Wrong dimensions')
            for v in (g, l):
                v[:, 0, :] = np.arcsinh(v[:, 0, :] * 1000)  # flux in ppt, NO hard clipping
                v[:, 1, :] = (v[:, 1, :] > 0).astype('f4')  # distinguish empty bins
            if any(not np.isfinite(v).all() for v in (g, l, f)): raise ValueError('Nonfinite')
            gs.append(g.reshape(6, G_N)); ls.append(l.reshape(6, L_N))
            fs.append(f); labels.append(CLASSES.index(row['class_label']))
        except (OSError, KeyError, ValueError) as exc: errors.append((k, str(exc)))
    if errors: raise RuntimeError(f'{split}: {len(errors)} invalid stars: {errors[:5]}')
    return rows, np.stack(gs), np.stack(ls), np.stack(fs), np.array(labels)

class Branch(nn.Module):
    """[B,6,L] -> Conv16 -> Pool/2 -> Conv32 -> Avg32+Max32 = [B,64]."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(6, 16, 7, padding=3), nn.ReLU(),
                                  nn.MaxPool1d(2), nn.Conv1d(16, 32, 5, padding=2),
                                  nn.ReLU())

    def forward(self, x):
        z = self.conv(x)
        return torch.cat([z.mean(dim=-1), z.amax(dim=-1)], dim=1)

class ExoCNN(nn.Module):
    """BLS: 64+64 -> 48 -> Dropout20% -> 5; Fusion adds Fourier 48->16."""
    def __init__(self, fusion):
        super().__init__()
        self.global_branch, self.local_branch = Branch(), Branch()
        self.fourier_branch = nn.Sequential(nn.Linear(48, 16), nn.ReLU()) if fusion else None
        self.head = nn.Sequential(nn.Linear(128 + 16 * fusion, 48), nn.ReLU(),
                                  nn.Dropout(.20), nn.Linear(48, 5))

    def forward(self, g, l, f):
        parts = [self.global_branch(g), self.local_branch(l)]
        if self.fourier_branch is not None: parts.append(self.fourier_branch(f))
        return self.head(torch.cat(parts, dim=1))

def show_architecture():
    for fusion in (False, True):
        m = ExoCNN(fusion).eval()
        g, l, f = torch.zeros(2, 6, G_N), torch.zeros(2, 6, L_N), torch.zeros(2, 48)
        with torch.no_grad():
            print('\nMODE:', 'CNN-Fusion' if fusion else 'CNN-BLS')
            print('Global/Local:', tuple(g.shape), tuple(l.shape), '->',
                  tuple(m.global_branch(g).shape), tuple(m.local_branch(l).shape))
            print('Fourier:', '(2,48) -> (2,16)' if fusion else 'unused',
                  '| output:', tuple(m(g, l, f).shape), '| params:', sum(x.numel() for x in m.parameters()))
            print('Convolutions:', m.global_branch.conv, '\nHead:', m.head)
def tensors(g, l, f, scaler):
    return (torch.as_tensor(g, device=DEVICE), torch.as_tensor(l, device=DEVICE),
            torch.as_tensor(np.clip(scaler.transform(f), -5, 5).astype('f4'), device=DEVICE))
def evaluate(m, data, labels, batch=256):
    m.eval(); pred, loss = [], 0.
    with torch.no_grad():
        for start in range(0, len(labels), batch):
            ids = slice(start, start + batch)
            logits = m(*(x[ids] for x in data))
            yy = torch.as_tensor(labels[ids], device=DEVICE, dtype=torch.long)
            loss += nn.functional.cross_entropy(logits, yy, reduction='sum').item()
            pred.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(pred), loss / len(labels)
def report(mode, split, rows, y, pred):
    f1 = f1_score(y, pred, average='macro', zero_division=0); bal = balanced_accuracy_score(y, pred)
    print(f'{mode} {split}: balanced accuracy={bal:.4f}; macro-F1={f1:.4f}')
    with (REPORTS / f'full6000_cnn_{mode}_{split}_predictions.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(('kepid', 'true', 'predicted'))
        w.writerows((r['kepid'], r['class_label'], CLASSES[p]) for r, p in zip(rows, pred))
    cm = confusion_matrix(y, pred, labels=range(5))
    with (REPORTS / f'full6000_cnn_{mode}_{split}_confusion.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(('true/pred', *CLASSES))
        w.writerows((name, *cm[i]) for i, name in enumerate(CLASSES))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--test', choices=('bls', 'fusion')); ap.add_argument('--architecture', action='store_true')
    args = ap.parse_args()
    if args.architecture: show_architecture(); return
    torch.manual_seed(42); np.random.seed(42)
    if DEVICE.type == 'cpu': torch.set_num_threads(4)
    REPORTS.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    if args.test:
        rows, g, l, f, y = load('test')
        ckpt = torch.load(MODELS / f'full6000_cnn_{args.test}.pt', map_location=DEVICE, weights_only=False)
        scaler = StandardScaler(); scaler.mean_, scaler.scale_ = ckpt['mean'], ckpt['scale']
        model = ExoCNN(args.test == 'fusion').to(DEVICE)
        model.load_state_dict(ckpt['weights'])
        report(args.test, 'test', rows, y, evaluate(model, tensors(g, l, f, scaler), y)[0]); return
    _, gt, lt, ft, yt = load('train')
    valrows, gv, lv, fv, yv = load('val')
    scaler = StandardScaler().fit(ft)  # training statistics ONLY
    tr, va = tensors(gt, lt, ft, scaler), tensors(gv, lv, fv, scaler)
    targets = torch.as_tensor(yt, device=DEVICE, dtype=torch.long)
    for name in ('bls', 'fusion'):
        torch.manual_seed(42)
        model = ExoCNN(name == 'fusion').to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01); best, history = -1., []
        for epoch in range(1, args.epochs + 1):
            model.train(); total = 0.
            for ids in torch.randperm(len(yt), device=DEVICE).split(64):
                loss = nn.functional.cross_entropy(model(*(v[ids] for v in tr)), targets[ids])
                opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * len(ids)
            pred, val_loss = evaluate(model, va, yv)
            f1 = f1_score(yv, pred, average='macro', zero_division=0)
            history.append((epoch, total / len(yt), val_loss, f1))
            if f1 > best:
                best = f1
                torch.save(dict(weights=model.state_dict(), mean=scaler.mean_, scale=scaler.scale_,
                                epoch=epoch, view_version='views-v3'), MODELS / f'full6000_cnn_{name}.pt')
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f'{name} epoch {epoch}: train={total/len(yt):.4f} val_loss={val_loss:.4f} val_F1={f1:.4f}', flush=True)
        torch.save(dict(weights=model.state_dict(), mean=scaler.mean_, scale=scaler.scale_), MODELS / f'full6000_cnn_{name}_last.pt')
        with (REPORTS / f'full6000_cnn_{name}_history.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(('epoch','train_loss','val_loss','val_macro_f1')); w.writerows(history)
        ckpt = torch.load(MODELS / f'full6000_cnn_{name}.pt', map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['weights']); report(name, 'val', valrows, yv, evaluate(model, va, yv)[0])
    print('Validation done; test stays locked until model selection.')

if __name__ == '__main__':
    main()
