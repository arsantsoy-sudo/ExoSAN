
"""ExoSAN CLEAN: resumable 6000-star pipeline with per-stage timing."""
import argparse
import csv
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import OLD_ROOT
CATALOGS = (ROOT / 'data/catalogs/ml_split_v1.csv',
            OLD_ROOT / 'data/ml/ml_split_v1.csv',
            OLD_ROOT / 'data/catalogs/processed/ml_split_v1.csv')
CLASSES = {'planet': 0, 'false_positive': 1, 'eclipsing_binary': 2,
           'stellar_variability': 3, 'quiet': 4}
# Order is essential: later stages consume earlier NPZ outputs.
STAGES = {'preprocessing': ('01_preprocessing', 'data/processed'),
          'bls': ('02_bls', 'data/features/bls_v2'),
          'views': ('02_views', 'data/features/views_v3'),
          'fourier': ('03_fourier', 'data/features/fourier_v3')}
REQUIRED = {'preprocessing': ('time', 'flux', 'flux_raw', 'flux_err'),
            'bls': ('top_periods', 'global_views', 'local_views'),
            'views': ('global_views', 'local_views', 'version'),
            'fourier': ('feature_vector',)}
REPORT = ROOT / 'reports'
def catalog():
    path = next((p for p in CATALOGS if p.exists()), None)
    if path is None:
        raise FileNotFoundError('ml_split_v1.csv not found in CLEAN or OLD')
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    expected = {('train', c): 840 for c in CLASSES}
    expected.update({(s, c): 180 for s in ('val', 'test') for c in CLASSES})
    counts = Counter((r['split'], r['class_label']) for r in rows)
    ids = [int(r['kepid']) for r in rows]
    if len(rows) != 6000 or counts != expected or len(set(ids)) != 6000:
        raise ValueError('Unexpected catalog split/class counts or duplicate KIC')
    if any(int(r['class_id']) != CLASSES[r['class_label']] for r in rows):
        raise ValueError('Incorrect class_id in catalog')
    print(f'Frozen catalog: {path} | train 4200 / val 900 / test 900')
    return rows
def valid_output(stage, kic):
    path = ROOT / STAGES[stage][1] / f'{kic}.npz'
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            if any(key not in z for key in REQUIRED[stage]):
                return False
            if stage == 'bls':
                return (z['global_views'].shape == (3, 2, 401) and
                        z['local_views'].shape == (3, 2, 161))
            if stage == 'views':
                return (z['global_views'].shape == (3, 2, 1024) and
                        z['local_views'].shape == (3, 2, 256) and
                        str(z['version']) == 'views-v3' and
                        np.isfinite(z['global_views']).all() and
                        np.isfinite(z['local_views']).all())
            if stage == 'fourier':
                v = z['feature_vector']
                return v.shape == (48,) and np.isfinite(v).all()
            return z['time'].size >= 100
    except (OSError, ValueError, KeyError, EOFError):
        return False
def write_errors(errors):
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / 'full6000_failures.csv').open('w', newline='', encoding='utf-8') as f:
        out = csv.writer(f)
        out.writerow(('kepid', 'split', 'class_label', 'stage', 'exit_code'))
        out.writerows(errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('train', 'val', 'test', 'all'), default='all')
    ap.add_argument('--stage', choices=('all', *STAGES), default='all')
    ap.add_argument('--limit', type=int, default=0, help='First N catalog stars; 0=all')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--keep-plots', action='store_true')
    args = ap.parse_args()
    rows = catalog()
    if args.split != 'all':
        rows = [r for r in rows if r['split'] == args.split]
    if args.limit:
        rows = rows[:args.limit]
    stages = list(STAGES) if args.stage == 'all' else [args.stage]
    if args.check_only:
        for stage in stages:
            for split in ('train', 'val', 'test'):
                part = [r for r in rows if r['split'] == split]
                if part:
                    done = sum(valid_output(stage, r['kepid']) for r in part)
                    print(f'{stage:14s} {split:5s}: {done}/{len(part)} ready')
        return
    REPORT.mkdir(parents=True, exist_ok=True)
    log = REPORT / 'full6000_processing.log'
    times = REPORT / 'full6000_stage_times.csv'
    failed, completed, skipped = [], 0, 0
    new_csv = not times.exists() or times.stat().st_size == 0
    try:
        with log.open('a', encoding='utf-8') as output, \
             times.open('a', newline='', encoding='utf-8') as tf:
            table = csv.writer(tf)
            if new_csv:
                table.writerow(('kepid', 'split', 'stage', 'seconds', 'exit_code', 'valid'))
            for i, row in enumerate(rows, 1):
                kic = row['kepid']
                for stage in stages:
                    if valid_output(stage, kic):
                        skipped += 1
                        continue
                    script = STAGES[stage][0]
                    print(f'[{i}/{len(rows)}] {row["split"]} KIC {kic}: {stage}', flush=True)
                    output.write(f'\n### {kic} {row["split"]} {stage}\n'); output.flush()
                    cmd = [sys.executable, '-m', f'scripts.{script}', '--kepid', kic]
                    started = time.perf_counter()
                    with subprocess.Popen(cmd, cwd=ROOT, stdout=output,
                                          stderr=subprocess.STDOUT) as proc:
                        code = proc.wait()
                    seconds = time.perf_counter() - started
                    valid = code == 0 and valid_output(stage, kic)
                    table.writerow((kic, row['split'], stage, f'{seconds:.2f}', code, int(valid)))
                    tf.flush()
                    print(f'  TIME {stage}: {seconds:.1f} s | {"OK" if valid else "FAILED"}', flush=True)
                    if not valid:
                        failed.append((kic, row['split'], row['class_label'], stage, code))
                        write_errors(failed)
                        print(f'FAILED: details in {log}', flush=True)
                        break
                    completed += 1
                    if not args.keep_plots:
                        pic = {'preprocessing': f'results/preprocessing_v2/{kic}_qc.png',
                               'bls': f'results/bls_v2/{kic}_diagnostic.png',
                               'fourier': f'results/fourier_v3/{kic}_diagnostic.png',
                               'views': None}[stage]
                        if pic:
                            (ROOT / pic).unlink(missing_ok=True)
                if i % 50 == 0:
                    print(f'Progress {i}/{len(rows)} | new={completed}, '
                          f'reused={skipped}, failed={len(failed)}', flush=True)
    finally:
        write_errors(failed)
    print(f'Done: new={completed}, reused={skipped}, failed={len(failed)}')
    print(f'Timings: {times} | check completeness using --check-only')


if __name__ == '__main__':
    main()
