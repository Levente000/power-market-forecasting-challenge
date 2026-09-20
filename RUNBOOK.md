# Run Instructions

## Prerequisites

Python 3.10 or later is required. Internet access is needed for the first weather data fetch.

## Setup

Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows use these commands instead.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

Train and tune the model. The first run downloads weather data and caches it locally.

```bash
python -m src.scripts.tune_model --trials 50
```

## Backtesting

Evaluate the model on the holdout period.

```bash
python -m src.scripts.backtest
```

## Forecast

Create the forecast output for the latest available date.

```bash
python -m src.scripts.make_forecast
```

You can also specify a date.

```bash
python -m src.scripts.make_forecast --target-date 2024-07-05
```

## Configuration

Settings live in config/settings.json.

Important settings are include_weather, holdout_days, objective, quantile_tau, and target_mode.

Set objective to quantile and adjust quantile_tau if you want a more conservative or more aggressive forecast.