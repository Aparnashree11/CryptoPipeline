"""
Model Monitoring Dashboard Pipeline
Tracks model performance, data quality, and prediction accuracy over time
Generates HTML dashboard and updates metrics
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import pandas as pd
import logging
import json
import ast
import html as html_module
import os

DASHBOARD_FOLDER = '/opt/airflow/dashboard'
AB_TEST_JSON = os.path.join(DASHBOARD_FOLDER, 'ab_test_report.json')

default_args = {
    'owner': 'ml-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

DASHBOARD_PATH = '/opt/airflow/dashboard/dashboard.html'


def calculate_prediction_accuracy(**context):
    """
    Calculate rolling accuracy of predictions by comparing with actual outcomes
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        # Compare predictions with actual outcomes (1 hour later)
        query = """
            WITH prediction_outcomes AS (
                SELECT 
                    p.symbol,
                    p.prediction_timestamp,
                    p.current_price,
                    p.predicted_direction,
                    p.confidence,
                    p.signal,
                    -- Get actual price 1 hour later from features
                    (
                        SELECT f.close
                        FROM features f
                        WHERE f.symbol = p.symbol
                        AND f.timestamp >= p.prediction_timestamp + INTERVAL '50 minutes'
                        AND f.timestamp <= p.prediction_timestamp + INTERVAL '70 minutes'
                        ORDER BY ABS(EXTRACT(EPOCH FROM (f.timestamp - (p.prediction_timestamp + INTERVAL '1 hour'))))
                        LIMIT 1
                    ) as actual_price_1h_later
                FROM predictions p
                WHERE p.prediction_timestamp >= NOW() - INTERVAL '7 days'
            )
            SELECT 
                symbol,
                prediction_timestamp,
                predicted_direction,
                signal,
                confidence,
                current_price,
                actual_price_1h_later,
                CASE 
                    WHEN actual_price_1h_later > current_price THEN 1
                    ELSE 0
                END as actual_direction,
                (actual_price_1h_later - current_price) / current_price as actual_return
            FROM prediction_outcomes
            WHERE actual_price_1h_later IS NOT NULL;
        """
        
        df = hook.get_pandas_df(query)
        
        if df.empty:
            logging.warning("No predictions with outcomes yet")
            return {
                'overall_accuracy': 0,
                'total_predictions': 0,
                'by_symbol': {},
                'by_signal': {}
            }
        
        # Calculate accuracy
        df['correct'] = (df['predicted_direction'] == df['actual_direction']).astype(int)
        
        overall_accuracy = df['correct'].mean()
        
        # Per-symbol accuracy
        symbol_accuracy = df.groupby('symbol').agg({
            'correct': 'mean',
            'predicted_direction': 'count'
        }).rename(columns={'predicted_direction': 'count'}).to_dict('index')
        
        # Per-signal accuracy
        signal_accuracy = df.groupby('signal').agg({
            'correct': 'mean',
            'predicted_direction': 'count'
        }).rename(columns={'predicted_direction': 'count'}).to_dict('index')
        
        # Confidence vs accuracy
        df['confidence_bucket'] = pd.cut(df['confidence'], bins=[0, 0.55, 0.6, 0.65, 1.0], 
                                         labels=['50-55%', '55-60%', '60-65%', '65%+'])
        confidence_accuracy = df.groupby('confidence_bucket')['correct'].mean().to_dict()
        
        metrics = {
            'overall_accuracy': float(overall_accuracy),
            'total_predictions': len(df),
            'by_symbol': {k: {'accuracy': float(v['correct']), 'count': int(v['count'])} 
                         for k, v in symbol_accuracy.items()},
            'by_signal': {k: {'accuracy': float(v['correct']), 'count': int(v['count'])} 
                         for k, v in signal_accuracy.items()},
            'by_confidence': {str(k): float(v) for k, v in confidence_accuracy.items() if pd.notna(v)},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logging.info(f"Overall Prediction Accuracy: {overall_accuracy:.2%} ({len(df)} predictions)")
        
        context['ti'].xcom_push(key='accuracy_metrics', value=metrics)
        
        return metrics
        
    except Exception as e:
        logging.error(f"Failed to calculate accuracy: {str(e)}")
        raise

def calculate_ab_testing_metrics(**context):
    """Fetch A/B testing results from ab_test_results table"""
    try:
        if not os.path.exists(AB_TEST_JSON):
            logging.warning(f"A/B test JSON file not found: {AB_TEST_JSON}")
            return []

        with open(AB_TEST_JSON, 'r') as f:
            data = json.load(f)

        # Transform JSON to match the structure used in HTML
        ab_results = [{
            'model_a': data.get('model_a', 'N/A'),
            'model_b': data.get('model_b', 'N/A'),
            'metric_name': 'accuracy',  # your dashboard expects this key
            'metric_value': data.get('model_b_metrics', {}).get('accuracy', 0.0)
        }]

        return ab_results

    except Exception as e:
        logging.error(f"Failed to load A/B testing metrics from JSON: {str(e)}")
        raise

def calculate_retraining_events():
    """Fetch retraining decisions from retrain_decisions table"""
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        sql = """
            SELECT
                should_retrain AS decision,
                details,
                created_at::text AS created_at
            FROM retrain_decisions
            WHERE created_at >= NOW() - INTERVAL '30 days'
            ORDER BY created_at DESC
            LIMIT 20;
        """
        df = hook.get_pandas_df(sql)
        df["details"] = df["details"].astype(str)  # ensure JSON serializes
        return df.to_dict(orient="records")
    except Exception as e:
        logging.error(f"Failed to calculate retraining events: {str(e)}")
        raise


def calculate_data_quality_metrics(**context):
    """
    Monitor data quality - completeness, freshness, outliers
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        metrics = {}
        
        # Data freshness
        freshness_query = """
            SELECT 
                'ohlcv_data' as table_name,
                MAX(timestamp) as last_update,
                EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))/60 as minutes_old
            FROM ohlcv_data
            UNION ALL
            SELECT 
                'features' as table_name,
                MAX(timestamp) as last_update,
                EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))/60 as minutes_old
            FROM features;
        """
        
        freshness_df = hook.get_pandas_df(freshness_query)
        # Convert timestamp columns to strings
        if 'last_update' in freshness_df.columns:
            freshness_df['last_update'] = freshness_df['last_update'].astype(str)
        metrics['data_freshness'] = freshness_df.to_dict('records')
        
        # Data completeness (NULL values)
        completeness_query = """
            SELECT 
                symbol,
                COUNT(*) as total_records,
                SUM(CASE WHEN rsi_14 IS NULL THEN 1 ELSE 0 END) as null_rsi,
                SUM(CASE WHEN macd IS NULL THEN 1 ELSE 0 END) as null_macd,
                SUM(CASE WHEN fear_greed_value IS NULL THEN 1 ELSE 0 END) as null_sentiment
            FROM features
            WHERE timestamp >= NOW() - INTERVAL '7 days'
            GROUP BY symbol;
        """
        
        completeness_df = hook.get_pandas_df(completeness_query)
        completeness_df['completeness_pct'] = (
            1 - (completeness_df[['null_rsi', 'null_macd', 'null_sentiment']].sum(axis=1) / 
                 (completeness_df['total_records'] * 3))
        ) * 100
        
        metrics['data_completeness'] = completeness_df.to_dict('records')
        
        # Outlier detection (price movements > 10% in 1 hour)
        outliers_query = """
            WITH price_changes AS (
                SELECT 
                    symbol,
                    timestamp,
                    close,
                    LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_close,
                    ABS((close - LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp)) / 
                        NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp), 0)) as pct_change
                FROM features
                WHERE timestamp >= NOW() - INTERVAL '7 days'
            )
            SELECT 
                symbol,
                COUNT(*) as outlier_count
            FROM price_changes
            WHERE pct_change > 0.10
            GROUP BY symbol;
        """
        
        outliers_df = hook.get_pandas_df(outliers_query)
        metrics['outliers'] = outliers_df.to_dict('records') if not outliers_df.empty else []
        
        # Record counts
        counts_query = """
            SELECT 
                (SELECT COUNT(*) FROM ohlcv_data WHERE timestamp >= NOW() - INTERVAL '7 days') as ohlcv_count,
                (SELECT COUNT(*) FROM features WHERE timestamp >= NOW() - INTERVAL '7 days') as features_count,
                (SELECT COUNT(*) FROM predictions WHERE prediction_timestamp >= NOW() - INTERVAL '7 days') as predictions_count;
        """
        
        counts = hook.get_first(counts_query)
        metrics['record_counts'] = {
            'ohlcv': counts[0],
            'features': counts[1],
            'predictions': counts[2]
        }
        
        metrics['timestamp'] = datetime.utcnow().isoformat()
        
        context['ti'].xcom_push(key='quality_metrics', value=metrics)
        
        return metrics
        
    except Exception as e:
        logging.error(f"Failed to calculate data quality: {str(e)}")
        raise


def calculate_model_performance_trends(**context):
    """
    Track model performance over time
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        # Model metrics from training runs
        metrics_query = """
            SELECT 
                symbol,
                model_name,
                metric_name,
                metric_value,
                training_timestamp,
                DATE(training_timestamp) as training_date
            FROM model_metrics
            WHERE training_timestamp >= NOW() - INTERVAL '30 days'
            ORDER BY training_timestamp DESC;
        """
        
        df = hook.get_pandas_df(metrics_query)
        
        if df.empty:
            logging.warning("No model metrics found")
            return {}
        
        # Get latest metrics per model
        latest_metrics = df.groupby(['symbol', 'model_name', 'metric_name']).first().reset_index()
        
        # Pivot for easier viewing
        pivot = latest_metrics.pivot_table(
            index=['symbol', 'model_name'],
            columns='metric_name',
            values='metric_value'
        ).reset_index()
        
        # Convert timestamps to strings
        if not df.empty:
            df['training_timestamp'] = df['training_timestamp'].astype(str)
            if 'training_date' in df.columns:
                df['training_date'] = df['training_date'].astype(str)

        metrics = {
            'latest_training': str(df['training_timestamp'].max()) if not df.empty else None,
            'model_performance': pivot.to_dict('records'),
            'training_history': df.to_dict('records')[-50:]
        }
        
        context['ti'].xcom_push(key='performance_metrics', value=metrics)
        
        return metrics
        
    except Exception as e:
        logging.error(f"Failed to calculate performance trends: {str(e)}")
        raise

def format_retrain_details(details_str):
    # Safely convert string back to dict
    try:
        details = ast.literal_eval(details_str)
    except Exception:
        return "Invalid details format"

    reasons = details.get('reasons', [])
    summary = []

    if 'data_drift' in reasons:
        triggers = details.get('drift_check', {}).get('triggers', [])
        symbols = [t.get('symbol', '') for t in triggers]
        if symbols:
            summary.append(f"Data Drift: {', '.join(symbols)}")

    if 'accuracy_degradation' in reasons:
        triggers = details.get('accuracy_check', {}).get('triggers', [])
        symbols = [t.get('symbol', '') for t in triggers]
        if symbols:
            summary.append(f"Accuracy Degradation: {', '.join(symbols)}")

    if 'training_staleness' in reasons:
        days = details.get('staleness_check', {}).get('days_since_training', 0)
        summary.append(f"Staleness: {days:.1f} days")

    return "; ".join(summary) if summary else "No issues"


def generate_html_dashboard(**context):
    """
    Generate HTML dashboard with all metrics
    """
    try:
        # Get metrics from XCom
        accuracy_metrics = context['ti'].xcom_pull(key='accuracy_metrics') or {}
        quality_metrics = context['ti'].xcom_pull(key='quality_metrics') or {}
        performance_metrics = context['ti'].xcom_pull(key='performance_metrics') or {}
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Crypto ML Pipeline - Monitoring Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            margin-top: 10px;
            opacity: 0.9;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 20px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .status-good {{ color: #10b981; }}
        .status-warning {{ color: #f59e0b; }}
        .status-bad {{ color: #ef4444; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background-color: #f9fafb;
            font-weight: 600;
            color: #374151;
        }}
        tr:hover {{
            background-color: #f9fafb;
        }}
        .timestamp {{
            text-align: right;
            color: #666;
            font-size: 0.9em;
            margin-top: 20px;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-success {{ background: #d1fae5; color: #065f46; }}
        .badge-warning {{ background: #fef3c7; color: #92400e; }}
        .badge-danger {{ background: #fee2e2; color: #991b1b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Crypto ML Pipeline Dashboard</h1>
            <div class="subtitle">Real-time monitoring of model performance, data quality, and predictions</div>
            <div class="subtitle">Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
        </div>

        <!-- Key Metrics -->
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Prediction Accuracy</h3>
                <div class="metric-value {'status-good' if accuracy_metrics.get('overall_accuracy', 0) > 0.55 else 'status-warning'}">
                    {accuracy_metrics.get('overall_accuracy', 0):.1%}
                </div>
                <div class="metric-label">
                    Based on {accuracy_metrics.get('total_predictions', 0)} predictions
                </div>
            </div>

            <div class="metric-card">
                <h3>Data Freshness</h3>
                <div class="metric-value {'status-good' if quality_metrics.get('data_freshness', [{}])[0].get('minutes_old', 999) < 120 else 'status-warning'}">
                    {quality_metrics.get('data_freshness', [{}])[0].get('minutes_old', 0):.0f} min
                </div>
                <div class="metric-label">Since last data update</div>
            </div>

            <div class="metric-card">
                <h3>Total Predictions</h3>
                <div class="metric-value">
                    {quality_metrics.get('record_counts', {}).get('predictions', 0)}
                </div>
                <div class="metric-label">Last 7 days</div>
            </div>

            <div class="metric-card">
                <h3>Feature Records</h3>
                <div class="metric-value">
                    {quality_metrics.get('record_counts', {}).get('features', 0):,}
                </div>
                <div class="metric-label">Last 7 days</div>
            </div>
        </div>

        <!-- Accuracy by Symbol -->
        <div class="section">
            <h3>Prediction Accuracy by Symbol</h3>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Accuracy</th>
                        <th>Predictions</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add accuracy by symbol
        for symbol, data in accuracy_metrics.get('by_symbol', {}).items():
            accuracy = data['accuracy']
            count = data['count']
            status = 'success' if accuracy > 0.55 else 'warning' if accuracy > 0.50 else 'danger'
            html += f"""
                    <tr>
                        <td>{symbol.upper()}</td>
                        <td>{accuracy:.1%}</td>
                        <td>{count}</td>
                        <td><span class="badge badge-{status}">{accuracy:.1%}</span></td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>

        <!-- Accuracy by Signal -->
        <div class="section">
            <h3>Accuracy by Trading Signal</h3>
            <table>
                <thead>
                    <tr>
                        <th>Signal</th>
                        <th>Accuracy</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add accuracy by signal
        for signal, data in accuracy_metrics.get('by_signal', {}).items():
            html += f"""
                    <tr>
                        <td><strong>{signal}</strong></td>
                        <td>{data['accuracy']:.1%}</td>
                        <td>{data['count']}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>

        <!-- Data Quality -->
        <div class="section">
            <h3>Data Quality Metrics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Total Records</th>
                        <th>Completeness</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add data quality
        for item in quality_metrics.get('data_completeness', []):
            completeness = item.get('completeness_pct', 0)
            status = 'success' if completeness > 95 else 'warning' if completeness > 90 else 'danger'
            html += f"""
                    <tr>
                        <td>{item['symbol'].upper()}</td>
                        <td>{item['total_records']:,}</td>
                        <td>{completeness:.1f}%</td>
                        <td><span class="badge badge-{status}">{completeness:.1f}%</span></td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>

        <!-- Model Performance -->
        <div class="section">
            <h3>Latest Model Performance</h3>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>F1 Score</th>
                        <th>Precision</th>
                        <th>Recall</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add model performance
        for model in performance_metrics.get('model_performance', []):
            if model.get('model_name') in ['random_forest', 'gradient_boosting']:
                html += f"""
                    <tr>
                        <td>{model.get('symbol', 'N/A').upper()}</td>
                        <td>{model.get('model_name', 'N/A').replace('_', ' ').title()}</td>
                        <td>{model.get('accuracy', 0):.3f}</td>
                        <td>{model.get('f1', 0):.3f}</td>
                        <td>{model.get('precision', 0):.3f}</td>
                        <td>{model.get('recall', 0):.3f}</td>
                    </tr>
"""
        
        html += f"""
                </tbody>
            </table>
            <div class="metric-label" style="margin-top: 15px;">
                Last training: {performance_metrics.get('latest_training', 'N/A')}
            </div>
        </div>

        <div class="timestamp">
            Dashboard generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
    </div>
"""
        # A/B Testing Section
        ab_results = context['ti'].xcom_pull(task_ids='calc_ab_testing')
        retrain_events = context['ti'].xcom_pull(task_ids='calc_retraining_events')

        html = "<div class='section'><h2>A/B Testing (Last 30 Days)</h2>"

        if ab_results:
            html += """
            <table>
                <thead>
                    <tr>
                        <th>Model A</th>
                        <th>Model B</th>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
            """
            for row in ab_results:
                html += f"""
                    <tr>
                        <td>{row.get('model_a', 'N/A')}</td>
                        <td>{row.get('model_b', 'N/A')}</td>
                        <td>{row.get('metric_name', 'N/A')}</td>
                        <td>{row.get('metric_value', 0.0):.3f}</td>
                    </tr>
                """
            html += "</tbody></table>"
        else:
            html += "<p>No recent A/B tests found.</p>"

        html += "</div>"

        # Retraining Section
        html = "<div class='section'><h2>Retraining Events (Last 30 Days)</h2>"
        if retrain_events:
            html += "<table>"
            html += "<tr><th>Decision</th><th>Details</th><th>Created At</th></tr>"

            for row in retrain_events:
                # Format decision as Yes/No
                decision_str = "Yes" if row.get('decision') else "No"

                # Format details using the safe function
                formatted_details = html_module.escape(format_retrain_details(row.get('details', '')))

                # Format created_at nicely
                created_at_str = row.get('created_at')
                if hasattr(created_at_str, 'strftime'):
                    created_at_str = created_at_str.strftime("%Y-%m-%d %H:%M:%S")

                html += f"<tr><td>{decision_str}</td><td>{formatted_details}</td><td>{created_at_str}</td></tr>"

            html += "</table>"
        else:
            html += "<p>No retraining events in the last 30 days.</p>"
        html+="</div>"

        # Append to final HTML
        html = html + "</body></html>"
        
        # Save dashboard
        import os
        os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
        
        with open(DASHBOARD_PATH, 'w') as f:
            f.write(html)
        
        logging.info(f"Dashboard saved to {DASHBOARD_PATH}")
        
        return "Dashboard generated successfully"
        
    except Exception as e:
        logging.error(f"Failed to generate dashboard: {str(e)}")
        raise


def save_monitoring_metrics(**context):
    """
    Save monitoring metrics to database for historical tracking
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        # Create monitoring table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS monitoring_metrics (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                metric_type VARCHAR(50),
                metric_name VARCHAR(100),
                metric_value DECIMAL(10, 6),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_monitoring_timestamp 
            ON monitoring_metrics(timestamp DESC, metric_type);
        """)
        conn.commit()
        
        # Get metrics
        accuracy_metrics = context['ti'].xcom_pull(key='accuracy_metrics') or {}
        quality_metrics = context['ti'].xcom_pull(key='quality_metrics') or {}
        
        timestamp = datetime.utcnow()
        
        # Save accuracy metrics
        if accuracy_metrics:
            cursor.execute("""
                INSERT INTO monitoring_metrics (timestamp, metric_type, metric_name, metric_value, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                timestamp,
                'accuracy',
                'overall_accuracy',
                accuracy_metrics.get('overall_accuracy', 0),
                json.dumps(accuracy_metrics)
            ))
        
        # Save data quality metrics
        if quality_metrics:
            cursor.execute("""
                INSERT INTO monitoring_metrics (timestamp, metric_type, metric_name, metric_value, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                timestamp,
                'data_quality',
                'freshness_minutes',
                quality_metrics.get('data_freshness', [{}])[0].get('minutes_old', 0),
                json.dumps(quality_metrics)
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info("Monitoring metrics saved to database")
        
    except Exception as e:
        logging.error(f"Failed to save monitoring metrics: {str(e)}")
        raise


# Define the DAG
with DAG(
    'crypto_monitoring_dashboard',
    default_args=default_args,
    description='Generate monitoring dashboard for ML pipeline',
    schedule_interval='*/30 * * * *',  # Every 30 minutes
    catchup=False,
    max_active_runs=1,
    tags=['crypto', 'monitoring', 'dashboard'],
) as dag:
    
    # Calculate metrics
    calc_accuracy = PythonOperator(
        task_id='calculate_accuracy',
        python_callable=calculate_prediction_accuracy,
    )
    
    calc_quality = PythonOperator(
        task_id='calculate_data_quality',
        python_callable=calculate_data_quality_metrics,
    )
    
    calc_performance = PythonOperator(
        task_id='calculate_model_performance',
        python_callable=calculate_model_performance_trends,
    )

    calc_ab_testing = PythonOperator(
    task_id="calc_ab_testing",
    python_callable=calculate_ab_testing_metrics
    )

    calc_retraining_events = PythonOperator(
        task_id="calc_retraining_events",
        python_callable=calculate_retraining_events
    )
    
    # Generate dashboard
    generate_dashboard = PythonOperator(
        task_id='generate_html_dashboard',
        python_callable=generate_html_dashboard,
    )
    
    # Save metrics
    save_metrics = PythonOperator(
        task_id='save_monitoring_metrics',
        python_callable=save_monitoring_metrics,
    )
    
    # Dependencies
    [calc_accuracy, calc_quality, calc_performance, calc_ab_testing, calc_retraining_events] >> generate_dashboard >> save_metrics
