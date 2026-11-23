

"""ExoSAN CLEAN | Fourier V3: two-channel spectral features, not BLS replacement."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, next_fast_len
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/processed'
OUT = ROOT / 'data/features/fourier_v3'
PLOTS = ROOT / 'results/fourier_v3'
EDGES = np.geomspace(0.5, 150.0, 25)
def load(kic):
    with np.load(INPUT / f'{kic}.npz', allow_pickle=False) as z:
        t, raw, flat, q = (np.asarray(z[k], float) for k in
                           ('time', 'flux_raw', 'flux', 'quarter'))
    good = np.isfinite(t) & np.isfinite(raw) & np.isfinite(flat) & np.isfinite(q)
    t, raw, flat, q = (a[good] for a in (t, raw, flat, q))
    ix = np.argsort(t)
    t, raw, flat, q = (a[ix] for a in (t, raw, flat, q))
    if len(t) < 1000 or np.ptp(t) < 150:
        raise ValueError('Недостаточно наблюдений')
    for quarter in np.unique(q):
        mask = q == quarter
        raw[mask] -= np.median(raw[mask])
        flat[mask] -= np.median(flat[mask])
    return t, raw, flat
def transforms(t, raw, flat):
    diff = np.diff(t)
    nearby = diff[(diff > .015) & (diff < .03)]
    if nearby.size < 100:
        raise ValueError('Не удалось определить Kepler long cadence')
    dt = float(np.median(nearby))
    n = int(np.ceil(np.ptp(t) / dt)) + 2
    index = np.rint((t - t[0]) / dt).astype(int)
    ok = (index >= 0) & (index < n)
    count = np.bincount(index[ok], minlength=n)
    mask = count > 0
    taper = np.hanning(n)
    nf = next_fast_len(4 * n)
    freq = rfftfreq(nf, dt)
    # Keep gaps missing; never interpolate across quarters.
    outputs = []
    for y in (raw, flat):
        grid = np.bincount(index[ok], weights=y[ok], minlength=n)
        grid[mask] /= count[mask]
        grid[mask] -= np.average(grid[mask], weights=count[mask])
        outputs.append(2 * np.abs(rfft(grid * taper, n=nf)) / taper[mask].sum())
    window = np.abs(rfft(mask.astype(float) * taper, n=nf)) / taper[mask].sum()
    return freq, outputs[0], outputs[1], window, dt, mask.mean()
def features(freq, raw, flat):
    period = np.divide(1, freq, out=np.full_like(freq, np.inf), where=freq > 0)
    vectors, curves = [], []
    for amp in (raw, flat):
        energy = np.array([np.sqrt(np.mean(amp[(period >= a) & (period < b)] ** 2))
                           if np.any((period >= a) & (period < b)) else 0.
                           for a, b in zip(EDGES[:-1], EDGES[1:])])
        vectors.extend(np.log1p(energy * 1e6))  # robust, fixed units: ppm
        curves.append(energy * 1e6)
    return np.array(vectors, dtype='f4'), np.asarray(curves)

def strongest(freq, amp, window):
    use = (freq >= 1 / 150) & (freq <= 1 / .5)
    ix = np.flatnonzero(use)
    bg = median_filter(amp, size=401, mode='nearest')
    excess = amp / np.maximum(bg, np.median(bg[use]) * .2 + 1e-12)
    peaks, _ = find_peaks(excess[ix], distance=16)
    order = ix[peaks[np.argsort(excess[ix][peaks])[::-1]]]
    chosen = []
    for i in order:
        p = 1 / freq[i]
        if all(abs(p / (1 / freq[j]) - 1) > .02 for j in chosen):
            chosen.append(int(i))
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        raise ValueError('Меньше трёх спектральных пиков')
    return np.array(chosen), excess[np.array(chosen)], window[np.array(chosen)]

def plot(kic, freq, raw, flat, window, curves, ix):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    valid = (freq >= 1 / 150) & (freq <= 1 / .5)
    periods = 1 / freq[valid]
    ax[0, 0].semilogx(periods, flat[valid] * 1e6, lw=.6)
    ax[0, 1].semilogx(periods, raw[valid] * 1e6, lw=.6)
    for j, pos in enumerate(ix, 1):
        ax[0, 0].axvline(1 / freq[pos], ls='--', alpha=.7,
                         label=f'FFT #{j}: {1 / freq[pos]:.3f} d')
    bls = ROOT / 'data/features/bls_v2' / f'{kic}.npz'
    if bls.exists():
        with np.load(bls, allow_pickle=False) as z:
            bp = float(z['top_periods'][0])
        ax[0, 0].axvline(bp, c='tab:red', ls=':', label=f'BLS: {bp:.3f} d')
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set(title='FFT: после detrending (транзиты)', ylabel='Амплитуда, ppm')
    ax[0, 1].set(title='FFT: до detrending (переменность)', ylabel='Амплитуда, ppm')
    ax[1, 0].semilogx(periods, window[valid], lw=.7)
    ax[1, 0].set(title='Спектральное окно: эффект пропусков', ylabel='Относит. амплитуда')
    centers = np.sqrt(EDGES[:-1] * EDGES[1:])
    ax[1, 1].semilogx(centers, curves[0], '.-', label='До detrending')
    ax[1, 1].semilogx(centers, curves[1], '.-', label='После detrending')
    ax[1, 1].set(title='Спектральные признаки для ML', ylabel='RMS амплитуда, ppm')
    ax[1, 1].legend(fontsize=8)
    for a in ax.flat:
        a.set_xlabel('Период, сутки'); a.grid(alpha=.2)
    fig.suptitle(f'KIC {kic} | Fourier V3 | спектральные признаки')
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / f'{kic}_diagnostic.png', dpi=140)
    plt.close(fig)

def run(kic, force):
    target = OUT / f'{kic}.npz'
    if target.exists() and not force:
        print(f'KIC {kic}: готов (пересчёт: --force)')
        return
    t, raw, flat = load(kic)
    f, a, b, w, dt, coverage = transforms(t, raw, flat)
    vector, curves = features(f, a, b)
    ix, score, alias = strongest(f, b, w)
    plot(kic, f, a, b, w, curves, ix)
    OUT.mkdir(parents=True, exist_ok=True)
    names = [f'{channel}_pbin_{j:02d}' for channel in ('raw', 'flat')
             for j in range(len(EDGES) - 1)]
    np.savez_compressed(target, feature_vector=vector, feature_names=np.array(names),
                        period_bin_edges=EDGES.astype('f4'), frequency=f.astype('f4'),
                        amplitude_raw=a.astype('f4'), amplitude_flat=b.astype('f4'),
                        spectral_window=w.astype('f4'),
                        top_spectral_periods=(1 / f[ix]).astype('f4'),
                        top_peak_ratio=score.astype('f4'), top_window_level=alias.astype('f4'),
                        cadence=np.float32(dt), coverage=np.float32(coverage),
                        version='fourier-v3-features')
    print(f'KIC {kic}: 48 FFT-признаков; пики: ' +
          ', '.join(f'{p:.3f} d' for p in 1 / f[ix]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kepid', nargs='+', type=int, default=[757450, 3219995, 2442448])
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    for kic in args.kepid:
        try: run(kic, args.force)
        except (OSError, ValueError, KeyError) as exc: print(f'KIC {kic}: {exc}')


if __name__ == '__main__':
    main()
