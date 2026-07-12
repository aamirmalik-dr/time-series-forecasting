"""tsforecast command-line interface.

A small forecasting application over the bundled monthly CO2 sample. It runs
fully offline: the sample series ships in the repository at
``data/co2_sample.csv``. Everything is driven by subcommands with explicit
flags.

Subcommands:
    forecast     Fit SARIMAX, forecast a held-out window with a 95% interval,
                 and write results/forecast.csv plus the hero plot results/forecast.png.
    backtest     Run a walk-forward (expanding-window) backtest and write
                 results/metrics.json.
    figures      Regenerate all figures: decomposition, forecast, residual diagnostics.
    diagnostics  Fit on the full series and print residual white-noise diagnostics.

Examples:
    python scripts/forecast.py forecast --horizon 24
    python scripts/forecast.py backtest --horizon 12 --holdout 60
    python scripts/forecast.py figures --results-dir results
    python scripts/forecast.py diagnostics --lags 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tsforecast import (
    SarimaxForecaster,
    decompose,
    load_co2_sample,
    load_fred_csv,
    walk_forward_backtest,
)

DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, 12)


def _parse_order(text: str) -> tuple[int, ...]:
    """Parse a comma-separated order string like ``1,1,1`` into a tuple of ints.

    Args:
        text: Comma-separated integers.

    Returns:
        Tuple of parsed integers.
    """
    return tuple(int(part) for part in text.split(","))


def _load_series(args: argparse.Namespace) -> pd.Series:
    """Load the working series from the committed sample or a FRED CSV.

    Args:
        args: Parsed CLI arguments. Uses ``args.data`` and ``args.column``.

    Returns:
        The series to model.
    """
    if args.data is not None:
        series = load_fred_csv(args.data, column=args.column)
        return series.asfreq(pd.infer_freq(series.index) or "MS").dropna()
    return load_co2_sample()


def _plot_forecast(
    series: pd.Series,
    horizon: int,
    order: tuple[int, ...],
    seasonal_order: tuple[int, ...],
    out_path: Path,
) -> tuple[pd.DataFrame, SarimaxForecaster]:
    """Fit on all but the last ``horizon`` points, forecast them, and plot the hero figure.

    Args:
        series: Monthly series.
        horizon: Number of trailing points held out and forecast.
        order: SARIMAX non-seasonal order.
        seasonal_order: SARIMAX seasonal order.
        out_path: Where to save the hero PNG.

    Returns:
        A tuple of (forecast table, fitted forecaster). The table has columns
        date, forecast, lower_95, upper_95, actual, abs_error.
    """
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    model = SarimaxForecaster(order=order, seasonal_order=seasonal_order).fit(train)
    fc = model.forecast(steps=horizon)

    table = pd.DataFrame(
        {
            "forecast": fc.mean.to_numpy(),
            "lower_95": fc.lower.to_numpy(),
            "upper_95": fc.upper.to_numpy(),
            "actual": test.to_numpy(),
        },
        index=test.index,
    )
    table["abs_error"] = (table["actual"] - table["forecast"]).abs()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    history = series.iloc[-horizon * 6 :]
    ax.plot(history.index, history.to_numpy(), color="#4b5563", linewidth=1.1, label="observed")
    ax.plot(
        test.index,
        fc.mean.to_numpy(),
        color="#c026d3",
        linewidth=1.8,
        label="forecast",
    )
    ax.fill_between(
        test.index,
        fc.lower.to_numpy(),
        fc.upper.to_numpy(),
        color="#c026d3",
        alpha=0.18,
        label="95% prediction interval",
    )
    ax.scatter(
        test.index,
        test.to_numpy(),
        s=18,
        color="#111827",
        zorder=5,
        label="held-out actual",
    )
    ax.set_title(
        f"SARIMAX{tuple(order)}x{tuple(seasonal_order)} forecast of the last "
        f"{horizon} months (95% prediction interval)"
    )
    ax.set_ylabel("CO2 (ppm)")
    ax.set_xlabel("date")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return table, model


def _plot_decomposition(series: pd.Series, out_path: Path) -> float:
    """Plot the STL decomposition and return the seasonal strength.

    Args:
        series: Monthly series to decompose.
        out_path: Where to save the PNG.

    Returns:
        Seasonal strength on a 0 to 1 scale.
    """
    result = decompose(series, period=12, method="stl")
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for ax, component, label in zip(
        axes,
        (result.observed, result.trend, result.seasonal, result.resid),
        ("Observed", "Trend", "Seasonal", "Residual"),
        strict=True,
    ):
        ax.plot(component.index, component.to_numpy(), linewidth=0.9)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)
    strength = result.seasonal_strength()
    axes[0].set_title(f"STL decomposition (seasonal strength {strength:.3f})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return strength


def _plot_diagnostics(model: SarimaxForecaster, out_path: Path) -> None:
    """Save the statsmodels residual diagnostics panel for a fitted model.

    Args:
        model: A fitted forecaster.
        out_path: Where to save the PNG.
    """
    fig = model.results.plot_diagnostics(figsize=(11, 8))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def cmd_forecast(args: argparse.Namespace) -> None:
    """Run the forecast subcommand: fit, forecast, save CSV and hero plot."""
    series = _load_series(args)
    order = _parse_order(args.order)
    seasonal_order = _parse_order(args.seasonal_order)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.plot or args.results_dir / "forecast.png"
    csv_path = args.output or args.results_dir / "forecast.csv"

    table, _ = _plot_forecast(series, args.horizon, order, seasonal_order, plot_path)
    table.to_csv(csv_path, index_label="date", float_format="%.4f")
    print(f"loaded {len(series)} obs, {series.index[0]:%Y-%m} to {series.index[-1]:%Y-%m}")
    print(f"wrote {plot_path}")
    print(f"wrote {csv_path}")
    print(
        f"held-out {args.horizon} months: mean abs error "
        f"{table['abs_error'].mean():.3f} ppm, max {table['abs_error'].max():.3f} ppm"
    )


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run the backtest subcommand: walk-forward backtest, save metrics.json."""
    series = _load_series(args)
    order = _parse_order(args.order)
    seasonal_order = _parse_order(args.seasonal_order)
    initial_train = len(series) - args.holdout
    result = walk_forward_backtest(
        series,
        order=order,
        seasonal_order=seasonal_order,
        initial_train_size=initial_train,
        horizon=args.horizon,
        step=args.step,
    )
    print(
        f"walk-forward backtest: initial train {initial_train} months, "
        f"{args.holdout} held-out months, horizon {args.horizon}"
    )
    print(result.summary())

    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics or args.results_dir / "metrics.json"
    payload = {
        "model": f"SARIMAX{tuple(order)}x{tuple(seasonal_order)}",
        "series": "co2_monthly_sample",
        "n_observations": int(len(series)),
        "initial_train_size": int(initial_train),
        "holdout_months": int(args.holdout),
        "horizon": int(args.horizon),
        "step": int(args.step) if args.step is not None else int(args.horizon),
        "n_folds": int(result.n_folds),
        "n_scored_points": int(len(result.actuals)),
        "rmse": round(result.rmse, 4),
        "mae": round(result.mae, 4),
        "mape_percent": round(result.mape, 4),
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {metrics_path}")


def cmd_figures(args: argparse.Namespace) -> None:
    """Run the figures subcommand: regenerate all three figures."""
    series = _load_series(args)
    order = _parse_order(args.order)
    seasonal_order = _parse_order(args.seasonal_order)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    strength = _plot_decomposition(series, args.results_dir / "decomposition.png")
    print(f"wrote {args.results_dir / 'decomposition.png'} (seasonal strength {strength:.3f})")

    _, model = _plot_forecast(
        series, args.horizon, order, seasonal_order, args.results_dir / "forecast.png"
    )
    print(f"wrote {args.results_dir / 'forecast.png'}")

    full = SarimaxForecaster(order=order, seasonal_order=seasonal_order).fit(series)
    _plot_diagnostics(full, args.results_dir / "residual_diagnostics.png")
    print(f"wrote {args.results_dir / 'residual_diagnostics.png'} (AIC {full.aic():.1f})")


def cmd_diagnostics(args: argparse.Namespace) -> None:
    """Run the diagnostics subcommand: print residual white-noise statistics."""
    series = _load_series(args)
    order = _parse_order(args.order)
    seasonal_order = _parse_order(args.seasonal_order)
    model = SarimaxForecaster(order=order, seasonal_order=seasonal_order).fit(series)
    diag = model.residual_diagnostics(lags=args.lags)
    print(f"model SARIMAX{tuple(order)}x{tuple(seasonal_order)} on {len(series)} obs")
    print(f"AIC {model.aic():.1f}")
    print(f"residual mean {diag.mean:.4f}, std {diag.std:.4f}")
    print(
        f"Ljung-Box(lags={diag.lags}) stat {diag.ljung_box_stat:.2f}, "
        f"p-value {diag.ljung_box_pvalue:.4f} -> "
        f"{'white noise' if diag.is_white_noise() else 'autocorrelation remains'}"
    )


def _add_common(sub: argparse.ArgumentParser) -> None:
    """Attach flags shared by every subcommand."""
    sub.add_argument("--results-dir", type=Path, default=Path("results"))
    sub.add_argument("--order", default="1,1,1", help="non-seasonal order p,d,q")
    sub.add_argument("--seasonal-order", default="1,1,1,12", help="seasonal order P,D,Q,s")
    sub.add_argument("--data", type=Path, default=None, help="optional FRED CSV instead of sample")
    sub.add_argument("--column", default=None, help="value column when using --data")


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all subcommands.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="forecast.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_fc = subparsers.add_parser("forecast", help="fit, forecast a held-out window, save outputs")
    _add_common(p_fc)
    p_fc.add_argument("--horizon", type=int, default=24, help="months to hold out and forecast")
    p_fc.add_argument("--output", type=Path, default=None, help="forecast CSV path")
    p_fc.add_argument("--plot", type=Path, default=None, help="hero PNG path")
    p_fc.set_defaults(func=cmd_forecast)

    p_bt = subparsers.add_parser("backtest", help="walk-forward backtest, save metrics.json")
    _add_common(p_bt)
    p_bt.add_argument("--horizon", type=int, default=12, help="forecast horizon per fold")
    p_bt.add_argument("--holdout", type=int, default=60, help="total months reserved for backtest")
    p_bt.add_argument("--step", type=int, default=None, help="train-window growth between folds")
    p_bt.add_argument("--metrics", type=Path, default=None, help="metrics JSON path")
    p_bt.set_defaults(func=cmd_backtest)

    p_fig = subparsers.add_parser("figures", help="regenerate all figures")
    _add_common(p_fig)
    p_fig.add_argument("--horizon", type=int, default=24, help="held-out horizon for forecast plot")
    p_fig.set_defaults(func=cmd_figures)

    p_diag = subparsers.add_parser("diagnostics", help="print residual white-noise diagnostics")
    _add_common(p_diag)
    p_diag.add_argument("--lags", type=int, default=12, help="Ljung-Box lags")
    p_diag.set_defaults(func=cmd_diagnostics)

    return parser


def main() -> None:
    """Parse arguments and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
