"""ExoSAN CLEAN — preprocessing V2 (one Kepler star at a time).
Run from CLEAN: python -m scripts.01_preprocessing --kepid 757450
Reads FITS from OLD; writes NPZ and a small QC plot to CLEAN only.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import RAW_DIR, PROCESSED_DIR, RESULTS_DIR

QUALITY_MASK = 1130799  # Kepler standard/default mask (bitwise)
WINDOW_DAYS = 3.0
GAP_DAYS = 0.5


def load_file(path):
    with fits.open(path, memmap=True) as hdul:
        table = hdul[1].data
        t = np.asarray(table['TIME'], dtype=np.float64)
        f = np.asarray(table['PDCSAP_FLUX'], dtype=np.float64)
        e = np.asarray(table['PDCSAP_FLUX_ERR'], dtype=np.float64)
        quality = np.asarray(table['SAP_QUALITY'], dtype=np.int64)
        quarter = int(hdul[0].header.get('QUARTER', -1))
    good = (np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
            & (f > 0) & (e > 0) & ((quality & QUALITY_MASK) == 0))
    if good.sum() < 20:
        return None
    t, f, e = t[good], f[good], e[good]
    median = np.median(f)
    order = np.argsort(t)
    return (t[order], f[order] / median - 1.0,
            e[order] / median, np.full(len(t), quarter, np.int16),
            int(len(quality)), int(good.sum()))


def detrend(t, flux, quarter):
    """Median subtraction per quarter and continuous observing segment."""
    flat = np.full(len(t), np.nan, dtype=np.float64)
    boundaries = np.r_[0, np.where((np.diff(t) > GAP_DAYS) |
                                    (np.diff(quarter) != 0))[0] + 1, len(t)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        length = stop - start
        if length < 10:
            continue
        dt = np.diff(t[start:stop])
        dt = dt[dt > 0]
        if not dt.size:
            continue
        width = max(5, int(round(WINDOW_DAYS / np.median(dt))) | 1)
        # The median filter can accept a window larger than the segment,
        # but we cap it to avoid an excessively edge-dominated baseline.
        largest_odd = length if length % 2 else length - 1
        width = min(width, largest_odd)
        baseline = median_filter(flux[start:stop], size=width, mode='nearest')
        flat[start:stop] = flux[start:stop] - baseline
    return flat


def run(kepid, force=False):
    folder_prefix = f'kplr{kepid:09d}'
    folders = sorted(p for p in RAW_DIR.glob(folder_prefix + '*') if p.is_dir())
    paths = sorted(file for folder in folders for file in folder.rglob('*.fits'))
    if not paths:
        raise FileNotFoundError(f'FITS files not found for KIC {kepid} in {RAW_DIR}')

    output = PROCESSED_DIR / f'{kepid}.npz'
    if output.exists() and not force:
        print(f'Already exists: {output} (use --force to overwrite)')
        return

    pieces = []
    total_rows = 0
    for path in paths:
        try:
            piece = load_file(path)
            if piece is not None:
                pieces.append(piece[:4])
                total_rows += piece[4]
        except (OSError, KeyError, ValueError, IndexError) as exc:
            print(f'SKIP {path.name}: {exc}')
    if not pieces:
        raise RuntimeError('No usable FITS data for this star')

    t, f, e, q = [np.concatenate([p[k] for p in pieces]) for k in range(4)]
    idx = np.argsort(t, kind='stable')
    t, f, e, q = t[idx], f[idx], e[idx], q[idx]
    # Remove duplicate timestamps if overlapping files were downloaded.
    unique = np.r_[True, np.diff(t) > 1e-8]
    t, f, e, q = t[unique], f[unique], e[unique], q[unique]
    flat = detrend(t, f, q)
    valid = np.isfinite(flat)
    if valid.sum() < 100:
        raise RuntimeError('Too few samples left after detrending')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, time=t[valid], flux_raw=f[valid].astype('f4'),
                        flux=flat[valid].astype('f4'), flux_err=e[valid].astype('f4'),
                        quarter=q[valid], kepid=np.int64(kepid),
                        window_days=np.float32(WINDOW_DAYS),
                        quality_mask=np.int64(QUALITY_MASK), version='2.0-pilot')

    # Plot only 30 days: complete arrays remain in the NPZ.
    view = valid & (t <= t[0] + 30)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t[view], f[view] * 1e6, '.', ms=2)
    axes[1].plot(t[view], flat[view] * 1e6, '.', ms=2)
    axes[0].set_title('Quality-filtered, normalized (before detrending)')
    axes[1].set_title('After 3-day segmented median detrending')
    for ax in axes:
        ax.set_ylabel('Relative flux [ppm]')
        ax.grid(alpha=.2)
    axes[1].set_xlabel('Kepler time [BKJD]')
    fig.suptitle(f'KIC {kepid} | {len(np.unique(q[valid]))} quarters | pilot V2')
    fig.tight_layout()
    plot = RESULTS_DIR / 'preprocessing_v2' / f'{kepid}_qc.png'
    plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot, dpi=140)
    plt.close(fig)
    print(f'FITS: {len(paths)} | usable: {len(pieces)} | quarters: {len(np.unique(q[valid]))}')
    print(f'Rows before filters: {total_rows:,} | saved cadences: {valid.sum():,}')
    print(f'NPZ: {output}\nQC: {plot}\nOLD data were not modified.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kepid", type=int, nargs="+",
        default=[757450, 3219995, 2442448]
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for kepid in args.kepid:
        print(f"\n=== KIC {kepid} ===")
        try:
            run(kepid, args.force)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"Ошибка KIC {kepid}: {error}")
