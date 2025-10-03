"""
Automated Retraining Trigger System
Monitors model performance and data drift
Automatically triggers retraining when thresholds are exceeded
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import json

default_args = {
    'owner': 'ml-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Retraining trigger thresholds
ACCURACY_THRESHOLD = 0.48  # Trigger if accuracy drops below 48%
ACCURACY_DEGRADATION = 0.10  # Trigger if accuracy drops 10% from baseline
MIN_PREDICTIONS_FOR_EVAL = 50  # Need at least 50 predictions to evaluate
DATA_DRIFT_THRESHOLD = 0.30  # Trigger if feature distribution changes >30%
DAYS_SINCE_LAST_TRAINING = 7  # Trigger if no training in 7 days


def check_model_accuracy(**context):
    """
    Check if model accuracy has degraded below threshold
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        # Get recent prediction accuracy
        query = """
            WITH prediction_outcomes AS (
                SELECT 
                    p.symbol,
                    p.prediction_timestamp,
                    p.predicted_direction,
                    (
                        SELECT 
                            CASE WHEN f2.close > f1.close THEN 1 ELSE 0 END
                        FROM features f1
                        JOIN features f2 ON f2.symbol = f1.symbol
                        WHERE f1.symbol = p.symbol
                        AND f1.timestamp = p.prediction_timestamp
                        AND f2.timestamp >= p.prediction_timestamp + INTERVAL '50 minutes'
                        AND f2.timestamp <= p.prediction_timestamp + INTERVAL '70 minutes'
                        ORDER BY ABS(EXTRACT(EPOCH FROM (f2.timestamp - (p.prediction_timestamp + INTERVAL '1 hour'))))
                        LIMIT 1
                    ) as actual_direction
                FROM predictions p
                WHERE p.prediction_timestamp >= NOW() - INTERVAL '3 days'
            )
            SELECT 
                symbol,
                COUNT(*) as total_predictions,
                AVG(CASE WHEN predicted_direction = actual_direction THEN 1.0 ELSE 0.0 END) as accuracy,
                -- Get baseline accuracy from older predictions
                (
                    SELECT AVG(CASE WHEN predicted_direction = actual_direction THEN 1.0 ELSE 0.0 END)
                    FROM prediction_outcomes po2
                    WHERE po2.symbol = po.symbol
                    AND po2.actual_direction IS NOT NULL
                    AND po2.prediction_timestamp >= NOW() - INTERVAL '10 days'
                    AND po2.prediction_timestamp < NOW() - INTERVAL '3 days'
                ) as baseline_accuracy
            FROM prediction_outcomes po
            WHERE actual_direction IS NOT NULL
            GROUP BY symbol;
        """
        
        df = hook.get_pandas_df(query)
        
        if df.empty:
            logging.info("Not enough predictions to evaluate accuracy")
            return {'should_retrain': False, 'reason': 'insufficient_data'}
        
        triggers = []
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            accuracy = row['accuracy']
            total_preds = row['total_predictions']
            baseline = row['baseline_accuracy']
            
            # Skip if not enough predictions
            if total_preds < MIN_PREDICTIONS_FOR_EVAL:
                continue
            
            # Check absolute accuracy threshold
            if accuracy < ACCURACY_THRESHOLD:
                triggers.append({
                    'symbol': symbol,
                    'reason': 'low_accuracy',
                    'current_accuracy': float(accuracy),
                    'threshold': ACCURACY_THRESHOLD
                })
                logging.warning(f"{symbol}: Accuracy {accuracy:.2%} below threshold {ACCURACY_THRESHOLD:.2%}")
            
            # Check relative degradation from baseline
            elif baseline and (baseline - accuracy) > ACCURACY_DEGRADATION:
                triggers.append({
                    'symbol': symbol,
                    'reason': 'accuracy_degradation',
                    'current_accuracy': float(accuracy),
                    'baseline_accuracy': float(baseline),
                    'degradation': float(baseline - accuracy)
                })
                logging.warning(f"{symbol}: Accuracy degraded {(baseline-accuracy):.2%} from baseline")
        
        result = {
            'should_retrain': len(triggers) > 0,
            'triggers': triggers,
            'check_type': 'accuracy',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        context['ti'].xcom_push(key='accuracy_check', value=result)
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to check model accuracy: {str(e)}")
        raise


def check_data_drift(**context):
    """
    Detect if feature distributions have changed significantly
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        # Compare recent feature statistics vs historical
        query = """
            WITH recent_stats AS (
                SELECT 
                    symbol,
                    AVG(rsi_14) as avg_rsi,
                    STDDEV(rsi_14) as std_rsi,
                    AVG(volatility_7) as avg_volatility,
                    STDDEV(volatility_7) as std_volatility,
                    AVG(volume_ratio) as avg_volume_ratio,
                    STDDEV(volume_ratio) as std_volume_ratio
                FROM features
                WHERE timestamp >= NOW() - INTERVAL '3 days'
                GROUP BY symbol
            ),
            historical_stats AS (
                SELECT 
                    symbol,
                    AVG(rsi_14) as avg_rsi,
                    STDDEV(rsi_14) as std_rsi,
                    AVG(volatility_7) as avg_volatility,
                    STDDEV(volatility_7) as std_volatility,
                    AVG(volume_ratio) as avg_volume_ratio,
                    STDDEV(volume_ratio) as std_volume_ratio
                FROM features
                WHERE timestamp >= NOW() - INTERVAL '30 days'
                AND timestamp < NOW() - INTERVAL '3 days'
                GROUP BY symbol
            )
            SELECT 
                r.symbol,
                ABS(r.avg_rsi - h.avg_rsi) / NULLIF(h.std_rsi, 0) as rsi_drift,
                ABS(r.avg_volatility - h.avg_volatility) / NULLIF(h.std_volatility, 0) as volatility_drift,
                ABS(r.avg_volume_ratio - h.avg_volume_ratio) / NULLIF(h.std_volume_ratio, 0) as volume_drift
            FROM recent_stats r
            JOIN historical_stats h ON r.symbol = h.symbol;
        """
        
        df = hook.get_pandas_df(query)
        
        if df.empty:
            logging.info("Not enough data to detect drift")
            return {'should_retrain': False, 'reason': 'insufficient_data'}
        
        triggers = []
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            
            # Calculate max drift across features
            drifts = [
                row['rsi_drift'],
                row['volatility_drift'],
                row['volume_drift']
            ]
            
            # Filter out NaN values
            valid_drifts = [d for d in drifts if pd.notna(d)]
            
            if not valid_drifts:
                continue
            
            max_drift = max(valid_drifts)
            
            if max_drift > DATA_DRIFT_THRESHOLD:
                triggers.append({
                    'symbol': symbol,
                    'reason': 'data_drift',
                    'max_drift': float(max_drift),
                    'threshold': DATA_DRIFT_THRESHOLD,
                    'rsi_drift': float(row['rsi_drift']) if pd.notna(row['rsi_drift']) else None,
                    'volatility_drift': float(row['volatility_drift']) if pd.notna(row['volatility_drift']) else None,
                    'volume_drift': float(row['volume_drift']) if pd.notna(row['volume_drift']) else None
                })
                logging.warning(f"{symbol}: Data drift detected - max drift {max_drift:.2f}")
        
        result = {
            'should_retrain': len(triggers) > 0,
            'triggers': triggers,
            'check_type': 'data_drift',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        context['ti'].xcom_push(key='drift_check', value=result)
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to check data drift: {str(e)}")
        raise


def check_training_staleness(**context):
    """
    Check if models haven't been trained in a while
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        query = """
            SELECT 
                MAX(training_timestamp) as last_training,
                EXTRACT(EPOCH FROM (NOW() - MAX(training_timestamp)))/86400 as days_since_training
            FROM model_metrics;
        """
        
        result = hook.get_first(query)
        
        if not result or not result[0]:
            logging.info("No training history found")
            return {
                'should_retrain': True,
                'reason': 'no_training_history',
                'timestamp': datetime.utcnow().isoformat()
            }
        
        last_training = result[0]
        days_since = result[1]
        
        should_retrain = days_since >= DAYS_SINCE_LAST_TRAINING
        
        if should_retrain:
            logging.warning(f"Models haven't been trained in {days_since:.1f} days")
        
        result = {
            'should_retrain': should_retrain,
            'last_training': str(last_training),
            'days_since_training': float(days_since),
            'threshold': DAYS_SINCE_LAST_TRAINING,
            'check_type': 'staleness',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        context['ti'].xcom_push(key='staleness_check', value=result)
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to check training staleness: {str(e)}")
        raise


def decide_retraining(**context):
    """
    Aggregate all checks and decide if retraining should be triggered
    """
    try:
        accuracy_check = context['ti'].xcom_pull(key='accuracy_check') or {}
        drift_check = context['ti'].xcom_pull(key='drift_check') or {}
        staleness_check = context['ti'].xcom_pull(key='staleness_check') or {}
        
        reasons = []
        
        if accuracy_check.get('should_retrain'):
            reasons.append('accuracy_degradation')
        
        if drift_check.get('should_retrain'):
            reasons.append('data_drift')
        
        if staleness_check.get('should_retrain'):
            reasons.append('training_staleness')
        
        should_retrain = len(reasons) > 0
        
        decision = {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'accuracy_check': accuracy_check,
            'drift_check': drift_check,
            'staleness_check': staleness_check,
            'decision_time': datetime.utcnow().isoformat()
        }
        
        # Log decision
        logging.info("=" * 70)
        logging.info("AUTOMATED RETRAINING DECISION")
        logging.info("=" * 70)
        
        if should_retrain:
            logging.info(f"DECISION: TRIGGER RETRAINING")
            logging.info(f"Reasons: {', '.join(reasons)}")
            
            for reason in reasons:
                if reason == 'accuracy_degradation':
                    triggers = accuracy_check.get('triggers', [])
                    logging.info(f"\nAccuracy Issues ({len(triggers)} symbols):")
                    for t in triggers:
                        logging.info(f"  - {t['symbol']}: {t['current_accuracy']:.2%}")
                
                elif reason == 'data_drift':
                    triggers = drift_check.get('triggers', [])
                    logging.info(f"\nData Drift Detected ({len(triggers)} symbols):")
                    for t in triggers:
                        logging.info(f"  - {t['symbol']}: drift={t['max_drift']:.2f}")
                
                elif reason == 'training_staleness':
                    days = staleness_check.get('days_since_training', 0)
                    logging.info(f"\nTraining Staleness: {days:.1f} days since last training")
        else:
            logging.info("DECISION: NO RETRAINING NEEDED")
            logging.info("All checks passed:")
            logging.info(f"  - Accuracy: OK")
            logging.info(f"  - Data Drift: OK")
            logging.info(f"  - Staleness: OK")
        
        logging.info("=" * 70)
        
        # Save decision to database
        hook = PostgresHook(postgres_conn_id='crypto_db')
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrain_decisions (
                id SERIAL PRIMARY KEY,
                decision_time TIMESTAMP,
                should_retrain BOOLEAN,
                reasons TEXT[],
                details JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        cursor.execute("""
            INSERT INTO retrain_decisions (decision_time, should_retrain, reasons, details)
            VALUES (%s, %s, %s, %s);
        """, (
            datetime.utcnow(),
            should_retrain,
            reasons,
            json.dumps(decision) 
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        context['ti'].xcom_push(key='retrain_decision', value=decision)
        
        return decision
        
    except Exception as e:
        logging.error(f"Failed to make retraining decision: {str(e)}")
        raise


def should_trigger_retrain(**context):
    """
    Helper function to determine if TriggerDagRunOperator should run
    """
    decision = context['ti'].xcom_pull(key='retrain_decision')
    return decision and decision.get('should_retrain', False)

def branch_on_retrain_decision(**context):
    """
    Branch based on retraining decision
    """
    decision = context['ti'].xcom_pull(key='retrain_decision')
    
    if decision and decision.get('should_retrain', False):
        return 'trigger_model_training'
    else:
        return 'skip_retraining'

# Define the DAG
with DAG(
    'crypto_auto_retrain_trigger',
    default_args=default_args,
    description='Automated monitoring and retraining trigger system',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    catchup=False,
    max_active_runs=1,
    tags=['crypto', 'ml', 'monitoring', 'auto-retrain'],
) as dag:
    
    # Check various conditions
    check_accuracy = PythonOperator(
        task_id='check_model_accuracy',
        python_callable=check_model_accuracy,
    )
    
    check_drift = PythonOperator(
        task_id='check_data_drift',
        python_callable=check_data_drift,
    )
    
    check_staleness = PythonOperator(
        task_id='check_training_staleness',
        python_callable=check_training_staleness,
    )
    
    # Make decision
    make_decision = PythonOperator(
        task_id='decide_retraining',
        python_callable=decide_retraining,
    )
    
    # Branch based on decision
    branch_task = BranchPythonOperator(
        task_id='branch_on_decision',
        python_callable=branch_on_retrain_decision,
    )
    
    # Trigger retraining if needed
    trigger_retrain = TriggerDagRunOperator(
        task_id='trigger_model_training',
        trigger_dag_id='crypto_model_training',
        wait_for_completion=False,
    )
    
    # Skip retraining dummy task
    skip_retrain = DummyOperator(
        task_id='skip_retraining',
    )
    
    # Dependencies
    [check_accuracy, check_drift, check_staleness] >> make_decision >> branch_task
    branch_task >> [trigger_retrain, skip_retrain]