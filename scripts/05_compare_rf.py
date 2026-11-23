"""ExoSAN CLEAN: BLS vs Fourier V3 vs fusion, training-only 5-fold CV."""
import csv
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/catalogs/pilot50_v1.csv'
BLS = ROOT / 'data/features/bls_v2'
FFT = ROOT / 'data/features/fourier_v3'
REPORTS = ROOT / 'reports'
CLASSES = ('planet', 'false_positive', 'eclipsing_binary',
           'stellar_variability', 'quiet')


def build():
    with MANIFEST.open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 50 or any(sum(r['class_label'] == label for r in rows) != 10
                              for label in CLASSES):
        raise ValueError('Expected frozen pilot of 10 stars per class')
    b_features, f_features, labels = [], [], []
    for row in rows:
        kic = row['kepid']
        with np.load(BLS / f'{kic}.npz', allow_pickle=False) as b:
            fields = ('top_snr', 'top_periods', 'top_duration',
                      'top_depth', 'top_events')
            arrays = []
            for name in fields:
                a = np.asarray(b[name], dtype=float)
                if a.shape != (3,):
                    raise ValueError(f'KIC {kic}: incorrect {name} shape')
                arrays.append(np.log1p(np.maximum(a * (1e6 if name ==
                                                'top_depth' else 1), 0)))
            b_features.append(np.concatenate(arrays))
        with np.load(FFT / f'{kic}.npz', allow_pickle=False) as f:
            a = np.asarray(f['feature_vector'], dtype=float)
            if a.shape != (48,):
                raise ValueError(f'KIC {kic}: expected 48 FFT features')
            f_features.append(a)
        labels.append(CLASSES.index(row['class_label']))
    return rows, np.array(labels), np.asarray(b_features), np.asarray(f_features)


def write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file); writer.writerow(columns); writer.writerows(rows)


def main():
    rows, y, xb, xf = build()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    experiments = {'bls': xb, 'fourier': xf, 'fusion': np.hstack([xb, xf])}
    metrics, predictions = [], []
    REPORTS.mkdir(parents=True, exist_ok=True)
    for name, x in experiments.items():
        pred = np.full(len(y), -1, dtype=int)
        for fold, (tr, va) in enumerate(cv.split(x, y), 1):
            model = make_pipeline(SimpleImputer(strategy='median'),
                                  RandomForestClassifier(n_estimators=200,
                                  min_samples_leaf=2, random_state=42,
                                  class_weight='balanced', n_jobs=-1))
            model.fit(x[tr], y[tr]); pred[va] = model.predict(x[va])
            metrics.append((name, fold,
                            round(balanced_accuracy_score(y[va], pred[va]), 4),
                            round(f1_score(y[va], pred[va], average='macro'), 4)))
        metrics.append((name, 'OOF_total',
                        round(balanced_accuracy_score(y, pred), 4),
                        round(f1_score(y, pred, average='macro'), 4)))
        cm = confusion_matrix(y, pred, labels=np.arange(5))
        write_csv(REPORTS / f'pilot50_{name}_confusion.csv',
                  ('true/pred', *CLASSES),
                  [(label, *cm[i].tolist()) for i, label in enumerate(CLASSES)])
        predictions += [(name, r['kepid'], r['class_label'], CLASSES[p])
                        for r, p in zip(rows, pred)]
        print(f'{name:8s} | OOF balanced accuracy=' 
              f'{balanced_accuracy_score(y, pred):.3f} | '
              f'macro-F1={f1_score(y, pred, average="macro"):.3f}')
    write_csv(REPORTS / 'pilot50_rf_metrics.csv',
              ('features', 'fold', 'balanced_accuracy', 'macro_f1'), metrics)
    write_csv(REPORTS / 'pilot50_rf_predictions.csv',
              ('features', 'kepid', 'true_label', 'predicted_label'), predictions)
    print('Saved RF reports:', REPORTS)
    print('CAUTION: 50 stars; CV is exploratory, not a final test.')


if __name__ == '__main__':
    main()
