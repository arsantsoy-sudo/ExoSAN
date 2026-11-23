"""Full ExoSAN RF: frozen train/val; test only after model selection."""
import argparse
import csv
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
from sys import path as syspath
syspath.insert(0, str(ROOT))
from config import OLD_ROOT
CLASSES = ('planet', 'false_positive', 'eclipsing_binary',
           'stellar_variability', 'quiet')
CATALOGS = (ROOT / 'data/catalogs/ml_split_v1.csv',
            OLD_ROOT / 'data/ml/ml_split_v1.csv',
            OLD_ROOT / 'data/catalogs/processed/ml_split_v1.csv')
REPORTS, MODELS = ROOT / 'reports', ROOT / 'models'


def load_rows():
    path = next((p for p in CATALOGS if p.exists()), None)
    if path is None:
        raise FileNotFoundError('Copy the frozen ml_split_v1.csv to data/catalogs')
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    expected = {'train': 4200, 'val': 900, 'test': 900}
    if len(rows) != 6000 or {s: sum(r['split'] == s for r in rows)
                              for s in expected} != expected:
        raise ValueError('Wrong frozen train/val/test counts')
    if len({int(r['kepid']) for r in rows}) != 6000:
        raise ValueError('Duplicate KIC in frozen split')
    return rows


def features(rows, split):
    xb, xf, labels, subset, missing = [], [], [], [], []
    for row in rows:
        if row['split'] != split:
            continue
        kic = row['kepid']
        try:
            with np.load(ROOT / f'data/features/bls_v2/{kic}.npz') as z:
                data = []
                for name in ('top_snr', 'top_periods', 'top_duration',
                             'top_depth', 'top_events'):
                    values = np.asarray(z[name], float)
                    if values.shape != (3,):
                        raise ValueError(f'Invalid BLS {name}')
                    scale = 1e6 if name == 'top_depth' else 1
                    data.append(np.log1p(np.maximum(values * scale, 0)))
                b = np.concatenate(data)
            with np.load(ROOT / f'data/features/fourier_v3/{kic}.npz') as z:
                f = np.asarray(z['feature_vector'], float)
            if f.shape != (48,) or not np.isfinite(f).all():
                raise ValueError('Invalid Fourier features')
            xb.append(b); xf.append(f)
            labels.append(CLASSES.index(row['class_label'])); subset.append(row)
        except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
            missing.append((kic, str(exc)))
    if missing:
        print(f'{split}: {len(missing)} incomplete stars (first 8): {missing[:8]}')
        raise RuntimeError('Do not train/evaluate on a silently reduced dataset')
    return subset, np.asarray(labels), np.asarray(xb), np.asarray(xf)


def write_csv(filename, header, rows):
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / filename).open('w', encoding='utf-8', newline='') as file:
        w = csv.writer(file); w.writerow(header); w.writerows(rows)


def evaluate(model, name, mode, subset, y, x):
    pred = model.predict(x)
    bal = balanced_accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='macro', zero_division=0)
    print(f'{mode} {name:8s} | balanced accuracy={bal:.4f} | macro-F1={f1:.4f}')
    write_csv(f'full6000_rf_{mode}_{name}_predictions.csv',
              ('kepid', 'true', 'predicted'),
              [(r['kepid'], r['class_label'], CLASSES[p]) for r, p in zip(subset, pred)])
    matrix = confusion_matrix(y, pred, labels=np.arange(5))
    write_csv(f'full6000_rf_{mode}_{name}_confusion.csv',
              ('true/pred', *CLASSES),
              [(label, *matrix[i].tolist()) for i, label in enumerate(CLASSES)])
    return name, mode, round(bal, 5), round(f1, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', choices=('bls', 'fourier', 'fusion'),
                    help='Evaluate ONE already frozen model on test; no retraining')
    args = ap.parse_args()
    rows = load_rows()
    if args.test:
        subset, y, xb, xf = features(rows, 'test')
        inputs = {'bls': xb, 'fourier': xf, 'fusion': np.hstack([xb, xf])}
        target = MODELS / f'full6000_rf_{args.test}.joblib'
        if not target.exists():
            raise FileNotFoundError(target)
        model = joblib.load(target)
        evaluate(model, args.test, 'test', subset, y, inputs[args.test])
        return
    _, yt, bt, ft = features(rows, 'train')
    valrows, yv, bv, fv = features(rows, 'val')
    MODELS.mkdir(parents=True, exist_ok=True)
    experiments = {'bls': (bt, bv), 'fourier': (ft, fv),
                   'fusion': (np.hstack([bt, ft]), np.hstack([bv, fv]))}
    metrics = []
    for name, (xtrain, xval) in experiments.items():
        model = make_pipeline(SimpleImputer(strategy='median'),
                              RandomForestClassifier(n_estimators=200,
                              min_samples_leaf=2, random_state=42,
                              class_weight='balanced', n_jobs=-1))
        model.fit(xtrain, yt)
        joblib.dump(model, MODELS / f'full6000_rf_{name}.joblib')
        metrics.append(evaluate(model, name, 'val', valrows, yv, xval))
    write_csv('full6000_rf_val_metrics.csv',
              ('model', 'split', 'balanced_accuracy', 'macro_f1'), metrics)
    print('VAL finished; compare with the OLD validation results, not Pilot-50 CV.')
    print('Do not use --test until you have selected the final model.')


if __name__ == '__main__':
    main()
