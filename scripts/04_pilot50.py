"""ExoSAN CLEAN: frozen 10/class training-only pilot, then existing pipeline."""
import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import OLD_ROOT, RAW_DIR  # noqa: E402

CLASSES = ('planet', 'false_positive', 'eclipsing_binary',
           'stellar_variability', 'quiet')
MANIFEST = ROOT / 'data/catalogs/pilot50_v1.csv'
SPLITS = (OLD_ROOT / 'data/ml/ml_split_v1.csv',
          ROOT / 'data/catalogs/ml_split_v1.csv',
          OLD_ROOT / 'data/catalogs/processed/ml_split_v1.csv')
STAGES = {'preprocessing': ('01_preprocessing', 'data/processed'),
          'bls': ('02_bls', 'data/features/bls_v2'),
          'fourier': ('03_fourier', 'data/features/fourier_v3')}
EXCLUDE = {757450, 3219995, 2442448}  # earlier hand-picked examples


def select(min_fits):
    if MANIFEST.exists():
        print('Frozen manifest exists:', MANIFEST)
        return
    path = next((p for p in SPLITS if p.exists()), None)
    if path is None:
        raise FileNotFoundError('Place ml_split_v1.csv in CLEAN/data/catalogs/')
    with path.open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    eligible = [r for r in rows if r['split'] == 'train'
                and r['class_label'] in CLASSES and int(r['kepid']) not in EXCLUDE]
    train = {int(r['kepid']): r for r in eligible}
    if len(train) != len(eligible):
        raise ValueError('Duplicate KIC in training split')
    available = {}
    for folder in RAW_DIR.iterdir():
        if not folder.is_dir() or not folder.name.startswith('kplr'):
            continue
        part = folder.name.split('_')[0][4:]
        if part.isdigit() and int(part) in train:
            kic = int(part)
            n = sum(1 for _ in folder.rglob('*.fits'))
            available[kic] = available.get(kic, 0) + n
    chosen = []
    for i, label in enumerate(CLASSES):
        pool = [k for k, r in train.items() if r['class_label'] == label
                and available.get(k, 0) >= min_fits]
        random.Random(42 + i).shuffle(pool)
        if len(pool) < 10:
            raise ValueError(f'{label}: only {len(pool)} stars have >= {min_fits} FITS')
        chosen += [{'kepid': k, 'class_label': label, 'class_id': i,
                    'split': 'train', 'n_fits': available[k]} for k in pool[:10]]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=chosen[0].keys())
        writer.writeheader(); writer.writerows(chosen)
    print('Saved 50 frozen, training-only stars:', MANIFEST)
    for label in CLASSES:
        print(f'  {label}: 10')


def process(stage, per_class):
    with MANIFEST.open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 50 or any(sum(r['class_label'] == label for r in rows) != 10
                              for label in CLASSES):
        raise ValueError('Manifest must contain exactly 10 stars per class')
    if per_class:
        rows = [r for label in CLASSES
                for r in [x for x in rows if x['class_label'] == label][:per_class]]
    names = list(STAGES) if stage == 'all' else [stage]
    errors = []
    for name in names:
        script, outdir = STAGES[name]
        for number, row in enumerate(rows, 1):
            kic = row['kepid']
            output = ROOT / outdir / f'{kic}.npz'
            if output.exists():
                print(f'[{name} {number}/{len(rows)}] {kic}: already exists')
                continue
            print(f'[{name} {number}/{len(rows)}] KIC {kic} ({row["class_label"]})', flush=True)
            command = [sys.executable, '-m', f'scripts.{script}', '--kepid', kic]
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode or not output.exists():
                errors.append((name, kic, row['class_label'], result.returncode))
                print(f'FAILED: {name} KIC {kic}', flush=True)
    if errors:
        logfile = ROOT / 'reports/pilot50_failures.csv'
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file); writer.writerow(('stage', 'kepid', 'class', 'code'))
            writer.writerows(errors)
        print(f'{len(errors)} failed runs, see {logfile}')
    else:
        print('Requested stages finished without missing outputs.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('select', *STAGES, 'all'), default='select')
    parser.add_argument('--per-class', type=int, default=0,
                        help='0=all ten; 1=quick check of one per class')
    parser.add_argument('--min-fits', type=int, default=8)
    args = parser.parse_args()
    if not MANIFEST.exists():
        select(args.min_fits)
    if args.stage != 'select':
        process(args.stage, args.per_class)


if __name__ == '__main__':
    main()
