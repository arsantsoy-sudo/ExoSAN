
"""ExoSAN CLEAN — phase folding before/after detrending."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]


def fold(time, period):
    return ((time - time.min()) / period + 0.5) % 1 - 0.5


def bin_median(phase, flux, n=401, limits=(-0.5, 0.5)):
    edges = np.linspace(limits[0], limits[1], n + 1)
    centers = (edges[1:] + edges[:-1]) / 2
    index = np.searchsorted(edges, phase, side="right") - 1
    values = np.full(n, np.nan)

    for i in range(n):
        points = flux[index == i]
        if len(points) >= 3:
            values[i] = np.median(points)

    return centers, values


def transit_depth(phase, flux):
    """Approximate transit depth in ppm."""
    x, y = bin_median(phase, flux)
    inside = np.abs(x) < 0.012
    outside = (np.abs(x) > 0.07) & (np.abs(x) < 0.4)

    if not (
        np.isfinite(y[inside]).any()
        and np.isfinite(y[outside]).any()
    ):
        return np.nan

    return (
        np.nanmedian(y[outside]) - np.nanmedian(y[inside])
    ) * 1e6


def main(kepid, period):
    source = ROOT / "data" / "processed" / f"{kepid}.npz"

    if not source.is_file():
        raise FileNotFoundError(
            f"NPZ не найден: {source}\n"
            "Сначала запусти 01_preprocessing.py"
        )

    with np.load(source, allow_pickle=False) as data:
        required = {"time", "flux_raw", "flux", "quarter"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"В NPZ нет полей: {sorted(missing)}")

        t = np.asarray(data["time"], float)
        raw = np.asarray(data["flux_raw"], float)
        clean = np.asarray(data["flux"], float)
        quarters = np.asarray(data["quarter"])

    valid = np.isfinite(t) & np.isfinite(raw) & np.isfinite(clean)
    t = t[valid]
    raw = raw[valid]
    clean = clean[valid]
    quarters = quarters[valid]

    if len(t) < 100 or period <= 0:
        raise ValueError("Недостаточно измерений или неверный период")

    # Каталожный период, начало фазы определяем по данным.
    initial_phase = fold(t, period)
    x, y = bin_median(initial_phase, clean)
    filled = np.where(np.isfinite(y), y, np.nanmedian(y))
    smooth = median_filter(filled, size=3, mode="wrap")

    phase_center = x[np.argmin(smooth)]
    phase = (initial_phase - phase_center + 0.5) % 1 - 0.5

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8), layout="constrained"
    )

    for col, (flux, name) in enumerate([
        (raw, "До detrending"),
        (clean, "После detrending (3 суток)")
    ]):
        for row, (bounds, bins) in enumerate([
            ((-0.5, 0.5), 401),
            ((-0.08, 0.08), 161)
        ]):
            xx, yy = bin_median(phase, flux, bins, bounds)
            ax = axes[row, col]

            ax.plot(
                xx, yy * 1e6,
                ".-", markersize=3, linewidth=0.8
            )
            ax.axvline(0, color="tomato", linestyle="--")
            ax.axhline(0, color="gray", linewidth=0.6)
            ax.set_title(
                f"{name} — "
                f"{'вся орбита' if row == 0 else 'область транзита'}"
            )
            ax.set_xlabel("Орбитальная фаза")
            ax.set_ylabel("Относительная яркость [ppm]")
            ax.grid(alpha=0.2)

    # Одинаковая шкала на двух увеличенных графиках.
    levels = np.concatenate([
        ax.lines[0].get_ydata() for ax in axes[1]
    ])
    levels = levels[np.isfinite(levels)]

    if len(levels):
        lo, hi = np.percentile(levels, [1, 99])
        pad = max((hi - lo) * 0.1, 100)
        for ax in axes[1]:
            ax.set_ylim(lo - pad, hi + pad)

    fig.suptitle(
        f"KIC {kepid} | {len(np.unique(quarters))} кварталов "
        f"| P = {period:.7f} суток"
    )

    output = (
        ROOT / "results" / "phase_fold_check"
        / f"{kepid}_before_after.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    print(f"Измерений: {len(t):,}")
    print(f"Кварталов: {len(np.unique(quarters))}")
    print(f"Глубина до: ~{transit_depth(phase, raw):.0f} ppm")
    print(f"Глубина после: ~{transit_depth(phase, clean):.0f} ppm")
    print("Оценки глубины предварительные.")
    print("График:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kepid", type=int, default=757450)
    parser.add_argument("--period", type=float, default=8.8849116)
    args = parser.parse_args()

    main(args.kepid, args.period)
