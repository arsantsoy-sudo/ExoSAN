
"""ExoSAN CLEAN: blind BLS; original period search, clearer phase plots."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed'
OUT = ROOT / 'data/features/bls_v2'
PLOTS = ROOT / 'results/bls_v2'
BANDS = [(1, 5, 20000, [.05, .10, .18]),
         (5, 12, 18000, [.08, .16, .30]),
         (12, 40, 14000, [.12, .25, .50]),
         (40, 150, 10000, [.20, .40, .80])]
def load(kic):
    with np.load(DATA / f'{kic}.npz', allow_pickle=False) as d:
        t, y, e = (np.asarray(d[k], float) for k in ('time', 'flux', 'flux_err'))
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    t, y, e = t[ok], y[ok], e[ok]
    order = np.argsort(t)
    t, y, e = t[order], y[order], e[order]
    if t.size < 1000 or np.ptp(t) < 150:
        raise ValueError('Недостаточно наблюдений или короткая временная база')
    adjacent = np.diff(y)[(np.diff(t) > 0) & (np.diff(t) < .05)]
    noise = (1.4826 * np.median(np.abs(adjacent - np.median(adjacent)))
             / np.sqrt(2)) if adjacent.size else np.std(y)
    return t, y, np.maximum(e, max(noise, 1e-7))
def distinct(periods, values, n=8):
    chosen = []
    for i in np.argsort(values)[::-1]:
        p = float(periods[i])
        if np.isfinite(values[i]) and all(abs(p / q - 1) > .01 for q in chosen):
            chosen.append(p)
        if len(chosen) >= n:
            break
    return chosen
def count_events(t, p, d, t0):
    phase = (t - t0 + p / 2) % p - p / 2
    cycle = np.rint((t[np.abs(phase) < d / 2] - t0) / p).astype(int)
    return int(np.sum(np.unique(cycle, return_counts=True)[1] >= 2))
def search(t, y, dy):
    model = BoxLeastSquares(t, y, dy=dy)
    periods, powers, candidates = [], [], []
    for low, high, n, durations in BANDS:
        f = np.linspace(1 / high, 1 / low, n)
        coarse = model.power(1 / f, durations, objective='snr', method='fast')
        periods.append(np.asarray(coarse.period))
        powers.append(np.asarray(coarse.power))
        for p0 in distinct(coarse.period, coarse.power):
            f0, df = 1 / p0, f[1] - f[0]
            fine_f = np.linspace(max(1 / high, f0 - 8 * df),
                                 min(1 / low, f0 + 8 * df), 241)
            fine = model.power(1 / fine_f, durations, objective='snr',
                               method='fast', oversample=10)
            k = int(np.nanargmax(fine.power))
            p, d, t0 = (float(fine.period[k]), float(fine.duration[k]),
                        float(fine.transit_time[k]))
            depth, snr = float(fine.depth[k]), float(fine.depth_snr[k])
            events = count_events(t, p, d, t0)
            if depth > 0 and events >= 3 and np.isfinite(snr):
                candidates.append((snr, p, d, t0, depth, events))
        print(f'  Поиск {low}–{high} суток завершён', flush=True)
    candidates.sort(reverse=True)
    top = []
    for c in candidates:
        if all(abs(c[1] / other[1] - 1) > .01 for other in top):
            top.append(c)
        if len(top) == 3:
            break
    if len(top) < 3:
        raise RuntimeError('Получено меньше трёх допустимых BLS-пиков')
    p, s = np.concatenate(periods), np.concatenate(powers)
    order = np.argsort(p)
    return p[order], s[order], top
def bin_view(x, y, lo, hi, n):
    edges = np.linspace(lo, hi, n + 1)
    ids = np.searchsorted(edges, x, side='right') - 1
    good = (ids >= 0) & (ids < n)
    count = np.bincount(ids[good], minlength=n)
    signal = np.zeros(n, dtype='f4')
    for i in np.flatnonzero(count):
        signal[i] = np.median(y[good & (ids == i)])
    return np.stack((signal, (count > 0).astype('f4')))
def save(kic, t, y, pgrid, power, top, write=True):
    glob, local = [], []
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout='constrained')
    axes[0, 0].plot(pgrid, power, lw=.5)
    for c in top:
        axes[0, 0].axvline(c[1], ls='--', alpha=.6)
    axes[0, 0].set(xscale='log', xlabel='Период (сутки)',
                    ylabel='BLS SNR', title='Слепой BLS, без каталога')
    axes[0, 1].axis('off'); axes[0, 2].axis('off')
    for i, (snr, p, d, t0, depth, events) in enumerate(top):
        dt = (t - t0 + p / 2) % p - p / 2
        g = bin_view(dt / p, y, -.5, .5, 401)
        l = bin_view(dt / d, y, -4, 4, 161)
        glob.append(g); local.append(l)
        for row, lo, hi, n in ((1, -.5, .5, 301), (2, -4, 4, 61)):
            ax = axes[row, i]
            phase = dt / p if row == 1 else dt / d
            viz = bin_view(phase, y, lo, hi, n)  # только для графика
            x = np.linspace(lo, hi, n, endpoint=False) + (hi - lo) / (2 * n)
            if row == 2:
                near = (phase >= lo) & (phase <= hi)
                ax.scatter(phase[near], y[near] * 1e6, s=.6, color='gray', alpha=.15)
                ax.axvspan(-.5, .5, color='orange', alpha=.12)
            curve = np.where(viz[1] > 0, viz[0] * 1e6, np.nan)
            ax.plot(x, curve, '.-', ms=2, lw=.7, color='tab:blue')
            ax.axvline(0, color='tomato', ls='--', alpha=.7)
            ax.set(title=f'#{i + 1}: P={p:.5f} d, SNR={snr:.1f}',
                   xlabel='Фаза' if row == 1 else 'Время / длительность',
                   ylabel='Поток, ppm')
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f'KIC {kic} | BLS V2 | Global / Local')
    fig.savefig(PLOTS / f'{kic}_diagnostic.png', dpi=140)
    plt.close(fig)
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        c = np.asarray(top)
        np.savez_compressed(OUT / f'{kic}.npz', period_grid=pgrid.astype('f4'),
            bls_power=power.astype('f4'), top_snr=c[:, 0], top_periods=c[:, 1],
            top_duration=c[:, 2], top_t0=c[:, 3], top_depth=c[:, 4],
            top_events=c[:, 5], global_views=np.asarray(glob),
            local_views=np.asarray(local), version='bls-v2-compact-blind')
    for i, (snr, p, d, t0, depth, events) in enumerate(top, 1):
        print(f'  #{i}: P={p:.6f} d; SNR={snr:.1f}; depth={depth*1e6:.0f} ppm; events={int(events)}')
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kepid', nargs='+', type=int, default=[757450, 3219995, 2442448])
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()
    for kic in args.kepid:
        output = OUT / f'{kic}.npz'
        if output.exists() and not args.force and not args.plot_only:
            print(f'KIC {kic}: уже готов (для пересчёта --force)')
            continue
        t, y, dy = load(kic)
        if args.plot_only:
            with np.load(output, allow_pickle=False) as a:
                p, power = a['period_grid'], a['bls_power']
                names = ('top_snr', 'top_periods', 'top_duration', 'top_t0', 'top_depth', 'top_events')
                top = list(zip(*(a[name] for name in names)))
        else:
            print(f'KIC {kic}: начинаем BLS', flush=True)
            p, power, top = search(t, y, dy)
        save(kic, t, y, p, power, top, write=not args.plot_only)
if __name__ == '__main__': main()
