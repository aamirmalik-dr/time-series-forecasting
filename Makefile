# tsforecast task runner
# Requires the package installed in the active environment: pip install -e ".[dev]"

PYTHON ?= python

.PHONY: help install forecast backtest figures diagnostics sample test lint format all clean

help:
	@echo "Targets:"
	@echo "  install      pip install -e '.[dev]'"
	@echo "  forecast     fit and forecast a held-out window -> results/forecast.{png,csv}"
	@echo "  backtest     walk-forward backtest -> results/metrics.json"
	@echo "  figures      regenerate decomposition, forecast, residual diagnostics figures"
	@echo "  diagnostics  print residual white-noise diagnostics"
	@echo "  sample       regenerate data/co2_sample.csv from statsmodels"
	@echo "  test         run the pytest suite"
	@echo "  lint         ruff check src tests scripts"
	@echo "  format       black + ruff --fix"
	@echo "  all          forecast + backtest + figures"

install:
	$(PYTHON) -m pip install -e ".[dev]"

forecast:
	$(PYTHON) scripts/forecast.py forecast --horizon 24

backtest:
	$(PYTHON) scripts/forecast.py backtest --horizon 12 --holdout 60

figures:
	$(PYTHON) scripts/forecast.py figures --horizon 24

diagnostics:
	$(PYTHON) scripts/forecast.py diagnostics --lags 12

sample:
	$(PYTHON) scripts/make_sample.py

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests scripts

format:
	black src tests scripts
	ruff check --fix src tests scripts

all: forecast backtest figures

clean:
	rm -rf results/*.png results/*.csv results/*.json
