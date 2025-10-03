## Crypto ML Pipeline with Airflow

This project is an end-to-end MLOps lab that orchestrates cryptocurrency data pipelines and ML workflows using Apache Airflow. It ingests market data, engineers features, trains and evaluates multiple models, performs hourly inference, runs A/B tests between model variants, and generates a live monitoring dashboard. All services run locally via Docker Compose.

### Key Features
- **Data collection** from CoinPaprika, CryptoCompare, and Fear & Greed Index APIs
- **Feature engineering** with rich technical indicators and targets
- **Model training** (Logistic Regression, Random Forest, Gradient Boosting, Ridge Regression) with metrics persisted to Postgres
- **Hourly inference** with scaled features and ensemble signals
- **A/B testing** framework comparing two model versions with persisted outcomes
- **Monitoring dashboard** (HTML) showing accuracy, data quality, model performance, and retraining events
- **Auto-retrain trigger** based on accuracy, drift, or model staleness

### Repository Structure
```text
airflow_lab/
  dags/
    data_collection.py          # Collects market data, OHLCV, sentiment; validates quality (*/15 mins)
    feature_engineering.py      # Builds features/targets and stores in Postgres (hourly)
    model_training.py           # Trains 4 models per symbol; saves metrics and artifacts (daily)
    prediction_pipeline.py      # Loads models and predicts; saves predictions (hourly)
    ab_testing.py               # Compares two model versions; saves results (every 2 hours)
    monitoring_dashboard.py     # Generates dashboard and stores monitoring metrics (*/30 mins)
    auto_retrain.py             # Triggers retraining when thresholds exceed (every 6 hours)
  models/                       # Persisted trained model artifacts (.pkl) and scaling JSON
  dashboard/                    # Generated HTML dashboard and A/B test JSON
  logs/                         # Airflow logs
  plugins/                      # (empty placeholder)
  data/               
    historical_data.py          # Standalone backfill script for extended OHLCV history
  docker-compose.yml            # Airflow Webserver, Scheduler, Postgres
  dockerfile                    # Airflow 2.7 base + Python deps
  requirements.txt              # Python dependencies for the Airflow image
  README.md
```

## Architecture and Data Flow

```text
APIs (CoinPaprika, CryptoCompare, Fear & Greed)
        │
        ▼
crypto_data_collection (*/15) ──► Postgres tables: market_data, ohlcv_data, sentiment_data
        │                                  │
        └──────────────► data quality checks│
                                           ▼
crypto_feature_engineering (hourly) ──► Postgres table: features
                                           │
                                           ▼
crypto_model_training (daily) ──► models/*.pkl, *_scaling_params.json + model_metrics
                                           │
                                           ▼
crypto_prediction_inference (hourly) ──► predictions table + reports
                                           │
                                           ├────────► crypto_monitoring_dashboard (*/30) ──► dashboard.html
                                           │
                                           └────────► crypto_ab_testing (*/120) ──► ab_test_results + ab_test_report.json

crypto_auto_retrain_trigger (*/6) monitors accuracy, drift, staleness → triggers crypto_model_training
```

## Airflow DAGs and Schedules

- **crypto_data_collection** (*/15 minutes)
  - Creates tables if absent; fetches per-symbol market data and hourly OHLCV; ingests Fear & Greed index
  - Persists to `market_data`, `ohlcv_data`, `sentiment_data`; runs data-quality checks (freshness, nulls, outliers)

- **crypto_feature_engineering** (hourly)
  - Builds indicators: SMA/EMA, RSI, MACD, Bollinger Bands, volatility, momentum, lags, rolling stats, time and sentiment features
  - Creates targets: future returns + up/down labels; persists to `features`

- **crypto_model_training** (daily @ 00:00)
  - Prepares data for each symbol (30 days), splits train/val/test, scales features
  - Trains: Logistic Regression, Random Forest, Gradient Boosting, Ridge Regression
  - Saves models to `models/` and scaling params JSON; writes `model_metrics`; logs a training report

- **crypto_prediction_inference** (hourly)
  - Verifies models and scaling params exist; loads latest features matching training schema
  - Scales features, produces classification probabilities and regression return; ensemble voting for signal (BUY/SELL/HOLD)
  - Persists to `predictions` and computes rolling accuracy

- **crypto_ab_testing** (every 2 hours)
  - Compares two model variants (default: Random Forest vs Gradient Boosting) with traffic split
  - Stores `ab_test_results` and generates `dashboard/ab_test_report.json`

- **crypto_monitoring_dashboard** (every 30 minutes)
  - Aggregates accuracy, data quality, model performance trends, A/B testing, and retraining events
  - Generates `dashboard/dashboard.html` and stores time-series metrics in `monitoring_metrics`

- **crypto_auto_retrain_trigger** (every 6 hours)
  - Triggers retraining when: accuracy < 48%, accuracy drops >10% vs baseline, drift > 0.3, or last training ≥ 7 days
  - Records decisions in `retrain_decisions`

## Models

- Classification: Logistic Regression, Random Forest, Gradient Boosting
- Regression: Ridge Regression (predicts next-hour return and price)
- Artifacts saved per symbol in `models/`:
  - `{symbol}_random_forest.pkl`, `{symbol}_gradient_boosting.pkl`, `{symbol}_logistic_regression.pkl`, `{symbol}_ridge_regression.pkl`
  - `{symbol}_scaling_params.json` (feature names, means, stds)

## Getting Started

### Prerequisites
- Docker and Docker Compose installed

### Environment Variables
Define these before starting (e.g., in a `.env` next to `docker-compose.yml`):

```env
POSTGRES_USER
POSTGRES_PASSWORD
POSTGRES_DB

AIRFLOW_ADMIN_USER
AIRFLOW_ADMIN_PASSWORD
AIRFLOW_ADMIN_FNAME
AIRFLOW_ADMIN_LNAME
AIRFLOW_ADMIN_ROLE
AIRFLOW_ADMIN_EMAIL
```

### Build and Run

```bash
docker compose up -d --build
```

- Airflow Web UI: `http://localhost:8080` (login with the admin creds above)

### Configure Airflow Connection
Create the Postgres connection `crypto_db` used by DAGs:
- Conn Id: `crypto_db`
- Conn Type: Postgres
- Host: `postgres` (Docker service name)
- Schema: `airflow`
- Login: `${POSTGRES_USER}`
- Password: `${POSTGRES_PASSWORD}`
- Port: `5432`

You can add this via Airflow UI (Admin → Connections) or CLI inside the webserver container.

### Initial Historical Backfill (optional but recommended)
Populate `ohlcv_data` with extended history to improve features/training:

```bash
# Execute inside the webserver or scheduler container
docker compose exec airflow-webserver bash -lc "python /opt/airflow/data/historical_data.py"
```

This script fetches up to ~1 year of hourly OHLCV per symbol and writes to Postgres.

### Enable and Run DAGs
In Airflow UI:
1. Unpause `crypto_data_collection` and wait for the first successful run
2. Unpause `crypto_feature_engineering`
3. Unpause `crypto_model_training` (or trigger manually once)
4. Unpause `crypto_prediction_inference` for hourly predictions
5. Unpause `crypto_ab_testing`, `crypto_monitoring_dashboard`, and `crypto_auto_retrain_trigger`

### Outputs
- Models: `models/`
- Predictions and metrics: Postgres tables (`predictions`, `model_metrics`, `monitoring_metrics`, etc.)
- Dashboard: `dashboard/dashboard.html` (regenerated every 30 minutes)
- A/B test report JSON: `dashboard/ab_test_report.json`

## Data Schema (high level)

- `market_data(symbol, timestamp, price_usd, …)`
- `ohlcv_data(symbol, timestamp, open, high, low, close, volume, trades)`
- `sentiment_data(timestamp, fear_greed_value, fear_greed_classification)`
- `features(symbol, timestamp, … engineered features …, targets)`
- `model_metrics(symbol, model_name, metric_name, metric_value, training_timestamp)`
- `predictions(symbol, prediction_timestamp, current_price, predicted_direction, confidence, …)`
- `ab_test_results(test_name, symbol, model_version, prediction_timestamp, predicted_direction, confidence, …)`
- `retrain_decisions(decision_time, should_retrain, reasons, details)`
- `monitoring_metrics(timestamp, metric_type, metric_name, metric_value, metadata)`

## Troubleshooting

- **DAG import errors**: Confirm the Airflow image built successfully and volumes are mounted; check `logs/`.
- **Connection `crypto_db` not found**: Create the Postgres connection as described above.
- **API rate limits**: The collectors include retries; if failures persist, increase intervals or add API keys if available.
- **Models not found at inference**: Ensure `crypto_model_training` ran and artifacts exist in `models/`.
- **Dashboard not updating**: Ensure `crypto_monitoring_dashboard` is unpaused and has access to tables; check Airflow task logs.

## Requirements
See `requirements.txt`. The Docker image installs these into the Airflow environment:

```text
apache-airflow==2.7.0
apache-airflow-providers-postgres==5.6.0
requests==2.31.0
pandas==2.0.3
psycopg2-binary==2.9.7
numpy==1.24.3
scikit-learn==1.3.0
```

## Notes
- All schedules are set with `catchup=False` for simplicity.
- Model artifacts and dashboard files are volume-mounted so they persist across container restarts.


