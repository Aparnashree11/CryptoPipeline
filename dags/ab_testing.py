"""
A/B Testing Framework for Model Comparison
Allows running multiple model versions simultaneously and comparing their performance
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
import logging
import json
from pathlib import Path

CRYPTO_SYMBOLS = ['bitcoin', 'ethereum', 'cardano', 'solana', 'polkadot']
MODEL_DIR = '/opt/airflow/models'

default_args = {
    'owner': 'ml-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
}


def setup_ab_test(**context):
    """
    Initialize A/B test configuration
    Define which models to compare (Version A vs Version B)
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        # Create A/B test configuration table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_test_config (
                id SERIAL PRIMARY KEY,
                test_name VARCHAR(100) UNIQUE,
                model_a_name VARCHAR(100),
                model_a_path VARCHAR(255),
                model_b_name VARCHAR(100),
                model_b_path VARCHAR(255),
                traffic_split DECIMAL(3, 2) DEFAULT 0.5,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                status VARCHAR(20) DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- A/B test results table
            CREATE TABLE IF NOT EXISTS ab_test_results (
                id SERIAL PRIMARY KEY,
                test_name VARCHAR(100),
                symbol VARCHAR(50),
                model_version VARCHAR(10),
                prediction_timestamp TIMESTAMP,
                predicted_direction INTEGER,
                confidence DECIMAL(5, 4),
                actual_direction INTEGER,
                actual_return DECIMAL(10, 6),
                is_correct BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_ab_test_results 
            ON ab_test_results(test_name, model_version, prediction_timestamp);
        """)
        conn.commit()
        
        # Check if there's an active test
        cursor.execute("""
            SELECT test_name, model_a_name, model_b_name, traffic_split
            FROM ab_test_config
            WHERE status = 'active'
            ORDER BY created_at DESC
            LIMIT 1;
        """)
        
        active_test = cursor.fetchone()
        
        if not active_test:
            # Create default A/B test: Random Forest vs Gradient Boosting
            cursor.execute("""
                INSERT INTO ab_test_config 
                (test_name, model_a_name, model_a_path, model_b_name, model_b_path, 
                 traffic_split, start_date, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (test_name) DO NOTHING;
            """, (
                'rf_vs_gb_v1',
                'random_forest',
                'random_forest.pkl',
                'gradient_boosting',
                'gradient_boosting.pkl',
                0.5,  # 50/50 split
                datetime.utcnow(),
                'active'
            ))
            conn.commit()
            
            logging.info("Created new A/B test: Random Forest vs Gradient Boosting")
            
            test_config = {
                'test_name': 'rf_vs_gb_v1',
                'model_a': 'random_forest',
                'model_b': 'gradient_boosting',
                'split': 0.5
            }
        else:
            test_config = {
                'test_name': active_test[0],
                'model_a': active_test[1],
                'model_b': active_test[2],
                'split': float(active_test[3])
            }
            logging.info(f"Active A/B test: {test_config['test_name']}")
        
        cursor.close()
        conn.close()
        
        context['ti'].xcom_push(key='test_config', value=test_config)
        
        return test_config
        
    except Exception as e:
        logging.error(f"Failed to setup A/B test: {str(e)}")
        raise


def run_ab_predictions(symbol: str, **context):
    """
    Generate predictions using both model versions
    Randomly assign predictions to A or B based on traffic split
    """
    try:
        test_config = context['ti'].xcom_pull(key='test_config')
        
        if not test_config:
            logging.warning("No A/B test config found")
            return None
        
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        # Get latest features
        query = f"""
            SELECT 
                timestamp,
                sma_7, sma_14, ema_7, ema_14,
                rsi_14, macd, macd_signal,
                bb_upper, bb_lower, bb_width,
                volatility_7, volatility_14,
                momentum_1, momentum_3, momentum_7,
                volume_ratio,
                close_lag_1, close_lag_3, close_lag_7,
                hour, day_of_week,
                fear_greed_value
            FROM features
            WHERE symbol = '{symbol}'
            ORDER BY timestamp DESC
            LIMIT 1;
        """
        
        df = hook.get_pandas_df(query)
        
        if df.empty:
            logging.warning(f"No features for {symbol}")
            return None
        
        latest = df.iloc[0]
        timestamp = latest['timestamp']
        features = latest.drop('timestamp').fillna(0)
        
        # Load scaling parameters
        scaling_path = f"{MODEL_DIR}/{symbol}_scaling_params.json"
        with open(scaling_path, 'r') as f:
            scaling_params = json.load(f)
        
        # Prepare features
        feature_values = [features[name] for name in scaling_params['feature_names'] if name in features.index]
        X = np.array(feature_values).reshape(1, -1)
        
        # Scale
        means = np.array([scaling_params['feature_means'][name] for name in scaling_params['feature_names']])
        stds = np.array([scaling_params['feature_stds'][name] for name in scaling_params['feature_names']])
        X_scaled = (X - means) / stds
        
        # Randomly assign to A or B
        random_value = np.random.random()
        use_model_a = random_value < test_config['split']
        
        if use_model_a:
            model_name = test_config['model_a']
            model_version = 'A'
        else:
            model_name = test_config['model_b']
            model_version = 'B'
        
        # Load and predict with assigned model
        model_path = f"{MODEL_DIR}/{symbol}_{model_name}.pkl"
        
        if not Path(model_path).exists():
            logging.error(f"Model not found: {model_path}")
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        prediction = model.predict(X_scaled)[0]
        confidence = model.predict_proba(X_scaled)[0][prediction]
        
        result = {
            'test_name': test_config['test_name'],
            'symbol': symbol,
            'model_version': model_version,
            'model_name': model_name,
            'prediction_timestamp': str(timestamp),
            'predicted_direction': int(prediction),
            'confidence': float(confidence)
        }
        
        logging.info(f"{symbol} - Model {model_version} ({model_name}): {prediction} ({confidence:.2%})")
        
        context['ti'].xcom_push(key=f'ab_prediction_{symbol}', value=result)
        
        return result
        
    except Exception as e:
        logging.error(f"A/B prediction failed for {symbol}: {str(e)}")
        raise


def save_ab_predictions(**context):
    """
    Save A/B test predictions to database
    """
    try:
        test_config = context['ti'].xcom_pull(key='test_config')
        
        hook = PostgresHook(postgres_conn_id='crypto_db')
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        # Collect predictions
        all_predictions = []
        for symbol in CRYPTO_SYMBOLS:
            pred = context['ti'].xcom_pull(key=f'ab_prediction_{symbol}')
            if pred:
                all_predictions.append(pred)
        
        if not all_predictions:
            logging.warning("No A/B predictions to save")
            cursor.close()
            conn.close()
            return
        
        # Insert predictions
        insert_query = """
            INSERT INTO ab_test_results 
            (test_name, symbol, model_version, prediction_timestamp, 
             predicted_direction, confidence)
            VALUES (%s, %s, %s, %s, %s, %s);
        """
        
        for pred in all_predictions:
            cursor.execute(insert_query, (
                pred['test_name'],
                pred['symbol'],
                pred['model_version'],
                pred['prediction_timestamp'],
                pred['predicted_direction'],
                pred['confidence']
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"Saved {len(all_predictions)} A/B test predictions")
        
    except Exception as e:
        logging.error(f"Failed to save A/B predictions: {str(e)}")
        raise


def evaluate_ab_test_performance(**context):
    """
    Compare performance of Model A vs Model B
    Calculate metrics and statistical significance
    """
    try:
        test_config = context['ti'].xcom_pull(key='test_config')
        
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        # Get A/B predictions with actual outcomes
        query = f"""
            WITH ab_with_outcomes AS (
                SELECT 
                    ab.test_name,
                    ab.symbol,
                    ab.model_version,
                    ab.prediction_timestamp,
                    ab.predicted_direction,
                    ab.confidence,
                    -- Get actual outcome from features
                    (
                        SELECT 
                            CASE WHEN f2.close > f1.close THEN 1 ELSE 0 END
                        FROM features f1
                        JOIN features f2 ON f2.symbol = f1.symbol
                        WHERE f1.symbol = ab.symbol
                        AND f1.timestamp = ab.prediction_timestamp
                        AND f2.timestamp >= ab.prediction_timestamp + INTERVAL '50 minutes'
                        AND f2.timestamp <= ab.prediction_timestamp + INTERVAL '70 minutes'
                        ORDER BY ABS(EXTRACT(EPOCH FROM (f2.timestamp - (ab.prediction_timestamp + INTERVAL '1 hour'))))
                        LIMIT 1
                    ) as actual_direction
                FROM ab_test_results ab
                WHERE ab.test_name = '{test_config['test_name']}'
                AND ab.prediction_timestamp >= NOW() - INTERVAL '7 days'
            )
            SELECT 
                model_version,
                COUNT(*) as total_predictions,
                AVG(CASE WHEN predicted_direction = actual_direction THEN 1.0 ELSE 0.0 END) as accuracy,
                AVG(confidence) as avg_confidence,
                COUNT(CASE WHEN predicted_direction = actual_direction THEN 1 END) as correct_predictions
            FROM ab_with_outcomes
            WHERE actual_direction IS NOT NULL
            GROUP BY model_version;
        """
        
        df = hook.get_pandas_df(query)
        
        if df.empty:
            logging.warning("No A/B test results with outcomes yet")
            return {
                'test_name': test_config['test_name'],
                'model_a': {'accuracy': 0, 'predictions': 0},
                'model_b': {'accuracy': 0, 'predictions': 0},
                'winner': None
            }
        
        # Extract metrics
        results = {}
        for _, row in df.iterrows():
            version = row['model_version']
            results[f"model_{version.lower()}"] = {
                'accuracy': float(row['accuracy']),
                'predictions': int(row['total_predictions']),
                'avg_confidence': float(row['avg_confidence']),
                'correct': int(row['correct_predictions'])
            }
        
        # Determine winner
        model_a_acc = results.get('model_a', {}).get('accuracy', 0)
        model_b_acc = results.get('model_b', {}).get('accuracy', 0)
        
        if abs(model_a_acc - model_b_acc) < 0.02:  # Less than 2% difference
            winner = 'tie'
            confidence = 'low'
        elif model_a_acc > model_b_acc:
            winner = 'A'
            confidence = 'high' if (model_a_acc - model_b_acc) > 0.05 else 'medium'
        else:
            winner = 'B'
            confidence = 'high' if (model_b_acc - model_a_acc) > 0.05 else 'medium'
        
        report = {
            'test_name': test_config['test_name'],
            'model_a': test_config['model_a'],
            'model_b': test_config['model_b'],
            'model_a_metrics': results.get('model_a', {}),
            'model_b_metrics': results.get('model_b', {}),
            'winner': winner,
            'confidence': confidence,
            'evaluation_time': datetime.utcnow().isoformat()
        }
        
        # Log results
        logging.info("=" * 70)
        logging.info(f"A/B TEST RESULTS: {test_config['test_name']}")
        logging.info("=" * 70)
        logging.info(f"Model A ({test_config['model_a']}):")
        logging.info(f"  Accuracy: {model_a_acc:.2%}")
        logging.info(f"  Predictions: {results.get('model_a', {}).get('predictions', 0)}")
        logging.info(f"\nModel B ({test_config['model_b']}):")
        logging.info(f"  Accuracy: {model_b_acc:.2%}")
        logging.info(f"  Predictions: {results.get('model_b', {}).get('predictions', 0)}")
        logging.info(f"\nWinner: Model {winner} (Confidence: {confidence})")
        logging.info("=" * 70)
        
        context['ti'].xcom_push(key='ab_results', value=report)
        
        return report
        
    except Exception as e:
        logging.error(f"Failed to evaluate A/B test: {str(e)}")
        raise


def generate_ab_test_report(**context):
    """
    Generate detailed A/B test comparison report
    """
    try:
        test_config = context['ti'].xcom_pull(key='test_config')
        ab_results = context['ti'].xcom_pull(key='ab_results')
        
        if not ab_results:
            logging.warning("No A/B results to report")
            return
        
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        # Per-symbol performance
        symbol_query = f"""
            WITH ab_with_outcomes AS (
                SELECT 
                    ab.symbol,
                    ab.model_version,
                    ab.predicted_direction,
                    (
                        SELECT 
                            CASE WHEN f2.close > f1.close THEN 1 ELSE 0 END
                        FROM features f1
                        JOIN features f2 ON f2.symbol = f1.symbol
                        WHERE f1.symbol = ab.symbol
                        AND f1.timestamp = ab.prediction_timestamp
                        AND f2.timestamp >= ab.prediction_timestamp + INTERVAL '50 minutes'
                        AND f2.timestamp <= ab.prediction_timestamp + INTERVAL '70 minutes'
                        ORDER BY ABS(EXTRACT(EPOCH FROM (f2.timestamp - (ab.prediction_timestamp + INTERVAL '1 hour'))))
                        LIMIT 1
                    ) as actual_direction
                FROM ab_test_results ab
                WHERE ab.test_name = '{test_config['test_name']}'
                AND ab.prediction_timestamp >= NOW() - INTERVAL '7 days'
            )
            SELECT 
                symbol,
                model_version,
                COUNT(*) as predictions,
                AVG(CASE WHEN predicted_direction = actual_direction THEN 1.0 ELSE 0.0 END) as accuracy
            FROM ab_with_outcomes
            WHERE actual_direction IS NOT NULL
            GROUP BY symbol, model_version
            ORDER BY symbol, model_version;
        """
        
        symbol_df = hook.get_pandas_df(symbol_query)
        
        logging.info("\n" + "=" * 70)
        logging.info("PER-SYMBOL A/B COMPARISON")
        logging.info("=" * 70)
        
        if not symbol_df.empty:
            for symbol in symbol_df['symbol'].unique():
                symbol_data = symbol_df[symbol_df['symbol'] == symbol]
                logging.info(f"\n{symbol.upper()}:")
                
                for _, row in symbol_data.iterrows():
                    logging.info(f"  Model {row['model_version']}: {row['accuracy']:.2%} ({row['predictions']} predictions)")
        
        logging.info("\n" + "=" * 70)
        
        # Save comprehensive report to file
        report_path = '/opt/airflow/dashboard/ab_test_report.json'
        import os
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        full_report = {
            **ab_results,
            'per_symbol': symbol_df.to_dict('records') if not symbol_df.empty else []
        }
        
        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        logging.info(f"Full report saved to {report_path}")
        
        return full_report
        
    except Exception as e:
        logging.error(f"Failed to generate A/B report: {str(e)}")
        raise


# Define the DAG
with DAG(
    'crypto_ab_testing',
    default_args=default_args,
    description='A/B testing framework for model comparison',
    schedule_interval='0 */2 * * *',  # Every 2 hours
    catchup=False,
    max_active_runs=1,
    tags=['crypto', 'ml', 'ab-testing'],
) as dag:
    
    # Setup A/B test
    setup_test = PythonOperator(
        task_id='setup_ab_test',
        python_callable=setup_ab_test,
    )
    
    # Run predictions with both models (parallel)
    prediction_tasks = []
    for symbol in CRYPTO_SYMBOLS:
        task = PythonOperator(
            task_id=f'ab_predict_{symbol}',
            python_callable=run_ab_predictions,
            op_kwargs={'symbol': symbol},
        )
        prediction_tasks.append(task)
    
    # Save predictions
    save_predictions = PythonOperator(
        task_id='save_ab_predictions',
        python_callable=save_ab_predictions,
    )
    
    # Evaluate performance
    evaluate_test = PythonOperator(
        task_id='evaluate_ab_performance',
        python_callable=evaluate_ab_test_performance,
    )
    
    # Generate report
    generate_report = PythonOperator(
        task_id='generate_ab_report',
        python_callable=generate_ab_test_report,
    )
    
    # Dependencies
    setup_test >> prediction_tasks >> save_predictions >> evaluate_test >> generate_report