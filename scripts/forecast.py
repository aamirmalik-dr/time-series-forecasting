"""End-to-end demo: decompose, fit SARIMAX, backtest, and plot.

Runs entirely offline on the Mauna Loa CO2 monthly series bundled with
statsmodels. Writes three figures and prints backtest metrics.

Usage:
    python scripts/forecast.py [--results-dir results] [--horizon 12]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tsforecast import (
    SarimaxForecaster,
    decompose,
    load_co2_monthly,
    walk_forward_backtest,
)

ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 12)


def plot_decomposition(series, out_path: Path) -> None:
    """Plot STL decomposition components and save to file.

    Args:
        series: Monthly series to decompose.
        out_path: Where to save the PNG.
    """
    result = decompose(series, period=12, method="stl")
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for ax, component, label in zip(
        axes,
        (result.observed, result.trend, result.seasonal, result.resid),
        ("Observed", "Trend", "Seasonal", "Residual"),
        strict=True,
    ):
        ax.plot(component.index, component.values, linewidth=0.9)
        ax.set_ylabel(label)
    axes[0].set_title(f"STL decomposition (seasonal strength {result.seasonal_strength():.3f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path} (seasonal strength {result.seasonal_strength():.3f})")


def plot_forecast(series, horizon: int, out_path: Path) -> None:
    """Hold out the last ``horizon`` points, forecast them, and plot.

    Args:
        series: Monthly series.
        horizon: Number of trailing points to hold out and forecast.
        out_path: Where to save the PNG.
    """
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    model = SarimaxForecaster(order=ORDER, seasonal_order=SEASONAL_ORDER).fit(train)
    forecast = model.forecast(steps=horizon)
    fig, ax = plt.subplots(figsize=(10, 5))
    tail = series.iloc[-horizon * 6 :]
    ax.plot(tail.index, tail.values, label="observed", linewidth=1.0)
    ax.plot(test.index, forecast.mean.values, label="forecast", linewidth=1.4)
    ax.fill_between(
        test.index,
        forecast.lower.values,
        forecast.upper.values,
        alpha=0.25,
        label="95% interval",
    )
    ax.set_title(f"SARIMAX{ORDER}x{SEASONAL_ORDER} forecast, last {horizon} months held out")
    ax.set_ylabel("CO2 (ppm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_residual_diagnostics(series, out_path: Path) -> None:
    """Fit on the full series and save statsmodels residual diagnostics.

    Args:
        series: Monthly series.
        out_path: Where to save the PNG.
    """
    model = SarimaxForecaster(order=ORDER, seasonal_order=SEASONAL_ORDER).fit(series)
    fig = model.results.plot_diagnostics(figsize=(11, 8))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path} (AIC {model.aic():.1f})")


def main() -> None:
    """Run decomposition, forecast plot, diagnostics, and walk-forward backtest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--horizon", type=int, default=12, help="backtest horizon per fold")
    parser.add_argument(
        "--holdout-months",
        type=int,
        default=60,
        help="total months reserved for the walk-forward backtest",
    )
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    series = load_co2_monthly()
    print(
        f"loaded co2 monthly series: {len(series)} observations, "
        f"{series.index[0]:%Y-%m} to {series.index[-1]:%Y-%m}"
    )

    plot_decomposition(series, args.results_dir / "decomposition.png")
    plot_forecast(series, args.horizon, args.results_dir / "forecast_vs_actual.png")
    plot_residual_diagnostics(series, args.results_dir / "residual_diagnostics.png")

    initial_train = len(series) - args.holdout_months
    print(
        f"walk-forward backtest: initial train {initial_train} months, "
        f"{args.holdout_months} held-out months, horizon {args.horizon}"
    )
    result = walk_forward_backtest(
        series,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        initial_train_size=initial_train,
        horizon=args.horizon,
    )
    print(result.summary())


if __name__ == "__main__":
    main()
