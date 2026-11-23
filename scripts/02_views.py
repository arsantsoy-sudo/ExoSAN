"""ExoSAN CLEAN | V3 views from EXISTING blind BLS peaks (no period search)."""
import argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/processed'
BLS = ROOT / 'data/features/bls_v2'
OUT = ROOT / 'data/features/views_v3'
PLOTS = ROOT / 'results/views_v3'
GLOBAL_N, LOCAL_N = 1024, 256


def median_bins(x, flux, left, right, n):
    """Robust median + occupancy mask; missing bins are NOT interpolated."""
    idx = np.floor((x - left) * n / (right - left)).astype(int)
    use = (idx >= 0) & (idx < n) & np.isfinite(flux)
    if not np.any(use):
        return np.zeros((2, n), dtype='f4')
    ids, values = idx[use], flux[use]
    order = np.lexsort((values, ids))  # first by bin, then by flux
    values, ids = values[order], ids[order]
    counts = np.bincount(ids, minlength=n)
    starts = np.cumsum(counts) - counts
    good = counts > 0
    med = np.zeros(n, dtype='f4')
    low = starts[good] + (counts[good] - 1) // 2
    high = starts[good] + counts[good] // 2
    med[good] = (values[low] + values[high]) / 2
    return np.stack([med, good.astype('f4')])  # flux, observed-bin mask


def make_views(t, flux, periods, durations, t0):
    globals_, locals_ = [], []
    for p, d, epoch in zip(periods, durations, t0):
        if not (np.isfinite(p) and np.isfinite(d) and np.isfinite(epoch)
                and 0 < d < p):
            raise ValueError('Invalid BLS period/duration/transit_time')
        # Full orbit: [-P/2, P/2), always 1024 phase bins.
        delta = (t - epoch + p / 2) % p - p / 2
        globals_.append(median_bins(delta / p, flux, -.5, .5, GLOBAL_N))
        # Transit-centered: [-4D, 4D); 32 bins within duration D.
        locals_.append(median_bins(delta / d, flux, -4, 4, LOCAL_N))
    return np.asarray(globals_, 'f4'), np.asarray(locals_, 'f4')


def plot(kic, g, l, periods, durations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout='constrained')
    for i in range(3):
        for j, (data, left, right) in enumerate(((g[i], -.5, .5), (l[i], -4, 4))):
            ax = axes[i, j]
            x = left + (np.arange(data.shape[-1]) + .5) * (right - left) / data.shape[-1]
            ax.plot(x, np.where(data[1] > 0, data[0] * 1e6, np.nan), '.-', ms=1.5, lw=.6)
            if j == 1:
                ax.axvspan(-.5, .5, alpha=.12, color='tab:orange')
            ax.set(xlabel='Orbital phase' if j == 0 else 'Time / BLS duration',
                   ylabel='Flux (ppm)', title=f'#{i+1} P={periods[i]:.4f} d, D={durations[i]:.3f} d')
            ax.text(.02, .98, f'Bins with data: {data[1].mean():.0%}', transform=ax.transAxes,
                    va='top', fontsize=9)
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f'KIC {kic} | V3 views | periods unchanged')
    fig.savefig(PLOTS / f'{kic}_views.png', dpi=140)
    plt.close(fig)


def run(kic, force=False, show_plot=False):
    target = OUT / f'{kic}.npz'
    if target.exists() and not force:
        try:
            with np.load(target, allow_pickle=False) as z:
                if (z['global_views'].shape == (3, 2, GLOBAL_N)
                        and z['local_views'].shape == (3, 2, LOCAL_N)
                        and str(z['version']) == 'views-v3'):
                    print(f'KIC {kic}: views already ready')
                    return
        except (OSError, ValueError, KeyError):
            pass
    with np.load(RAW / f'{kic}.npz', allow_pickle=False) as z:
        t, flux = (np.asarray(z[key], 'f8') for key in ('time', 'flux'))
    with np.load(BLS / f'{kic}.npz', allow_pickle=False) as z:
        p, d, epoch = (np.asarray(z[key], 'f8') for key in
                       ('top_periods', 'top_duration', 'top_t0'))
    good = np.isfinite(t) & np.isfinite(flux)
    g, l = make_views(t[good], flux[good], p, d, epoch)
    if g.shape != (3, 2, GLOBAL_N) or l.shape != (3, 2, LOCAL_N):
        raise ValueError('Unexpected view shapes')
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, global_views=g, local_views=l, top_periods=p,
                        top_duration=d, top_t0=epoch, version='views-v3')
    print(f'KIC {kic}: global {GLOBAL_N}, local {LOCAL_N}; '
          f'coverage {g[:, 1].mean():.0%}/{l[:, 1].mean():.0%}')
    if show_plot:
        plot(kic, g, l, p, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kepid', nargs='+', type=int, required=True)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()
    for kic in args.kepid:
        run(kic, args.force, args.plot)


if __name__ == '__main__':
    main()
