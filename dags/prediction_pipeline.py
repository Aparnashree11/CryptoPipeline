"""
Cryptocurrency Prediction/Inference Pipeline
Loads trained models and generates predictions on latest data
Runs every hour to produce fresh predictions
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

# Configuration
CRYPTO_SYMBOLS = ['bitcoin', 'ethereum', 'cardano', 'solana', 'polkadot']
MODEL_DIR = '/opt/airflow/models'

default_args = {
    'owner': 'ml-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
    'execution_timeout': timedelta(minutes=15),
}


def check_models_exist(**context):
    """Verify that trained models exist before inference"""
    missing_models = []
    
    for symbol in CRYPTO_SYMBOLS:
        model_path = f"{MODEL_DIR}/{symbol}_random_forest.pkl"
        scaling_path = f"{MODEL_DIR}/{symbol}_scaling_params.json"
        
        if not Path(model_path).exists():
            missing_models.append(f"{symbol}_random_forest.pkl")
        if not Path(scaling_path).exists():
            missing_models.append(f"{symbol}_scaling_params.json")
    
    if missing_models:
        raise Exception(f"Missing models: {missing_models}. Run crypto_model_training first!")
    
    logging.info(f"All models found for {len(CRYPTO_SYMBOLS)} symbols")
    return True


def load_latest_features(symbol: str, **context):
    """
    Load the latest features for inference - matching training features exactly
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        # Get latest features - MUST MATCH TRAINING QUERY
        query = f"""
            SELECT 
                timestamp,
                close,
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
            LIMIT 50;
        """
        
        df = hook.get_pandas_df(query)
        
        if df.empty or len(df) < 10:
            logging.warning(f"Insufficient features for {symbol}: {len(df)} records")
            return None
        
        logging.info(f"Loaded {len(df)} feature records for {symbol}")
        
        # Get the latest row for prediction
        latest = df.iloc[0].copy()
        timestamp = latest['timestamp']
        current_price = latest['close']
        
        # Drop timestamp AND close (close was not used in training)
        features_only = latest.drop(['timestamp', 'close'])
        
        # Handle missing values
        features_only = features_only.fillna(0)  # Use 0 for missing values
        
        data = {
            'timestamp': str(timestamp),
            'current_price': float(current_price),
            'features': features_only.to_dict(),
            'feature_names': features_only.index.tolist()
        }
        
        context['ti'].xcom_push(key=f'features_{symbol}', value=data)
        
        return data
        
    except Exception as e:
        logging.error(f"Failed to load features for {symbol}: {str(e)}")
        raise


def generate_predictions(symbol: str, **context):
    """
    Load model and generate predictions
    """
    try:
        # Get features
        data = context['ti'].xcom_pull(key=f'features_{symbol}')
        
        if not data:
            logging.warning(f"No features available for {symbol}")
            return None
        
        # Load scaling parameters
        scaling_path = f"{MODEL_DIR}/{symbol}_scaling_params.json"
        with open(scaling_path, 'r') as f:
            scaling_params = json.load(f)
        
        # Prepare features
        feature_values = [data['features'][name] for name in scaling_params['feature_names']]
        X = np.array(feature_values).reshape(1, -1)
        
        # Scale features
        means = np.array([scaling_params['feature_means'][name] for name in scaling_params['feature_names']])
        stds = np.array([scaling_params['feature_stds'][name] for name in scaling_params['feature_names']])
        X_scaled = (X - means) / stds
        
        # Load models and generate predictions
        predictions = {
            'symbol': symbol,
            'timestamp': data['timestamp'],
            'current_price': data['current_price']
        }
        
        # Random Forest (Classification) - Primary model
        rf_path = f"{MODEL_DIR}/{symbol}_random_forest.pkl"
        with open(rf_path, 'rb') as f:
            rf_model = pickle.load(f)
        
        rf_pred = rf_model.predict(X_scaled)[0]
        rf_proba = rf_model.predict_proba(X_scaled)[0]
        
        predictions['rf_prediction'] = int(rf_pred)
        predictions['rf_confidence'] = float(rf_proba[rf_pred])
        predictions['rf_prob_up'] = float(rf_proba[1])
        predictions['rf_prob_down'] = float(rf_proba[0])
        
        # Gradient Boosting (Classification)
        gb_path = f"{MODEL_DIR}/{symbol}_gradient_boosting.pkl"
        if Path(gb_path).exists():
            with open(gb_path, 'rb') as f:
                gb_model = pickle.load(f)
            
            gb_pred = gb_model.predict(X_scaled)[0]
            gb_proba = gb_model.predict_proba(X_scaled)[0]
            
            predictions['gb_prediction'] = int(gb_pred)
            predictions['gb_confidence'] = float(gb_proba[gb_pred])
        
        # Ridge Regression (Regression)
        ridge_path = f"{MODEL_DIR}/{symbol}_ridge_regression.pkl"
        if Path(ridge_path).exists():
            with open(ridge_path, 'rb') as f:
                ridge_model = pickle.load(f)
            
            ridge_pred = ridge_model.predict(X_scaled)[0]
            predictions['ridge_predicted_return'] = float(ridge_pred)
            predictions['ridge_predicted_price'] = float(data['current_price'] * (1 + ridge_pred))
        
        # Ensemble prediction (majority vote)
        votes = []
        if 'rf_prediction' in predictions:
            votes.append(predictions['rf_prediction'])
        if 'gb_prediction' in predictions:
            votes.append(predictions['gb_prediction'])
        
        if votes:
            predictions['ensemble_prediction'] = int(np.round(np.mean(votes)))
            predictions['ensemble_confidence'] = float(np.mean([
                predictions.get('rf_confidence', 0.5),
                predictions.get('gb_confidence', 0.5)
            ]))
        
        # Generate trading signal
        confidence_threshold = 0.6
        
        if predictions.get('ensemble_confidence', 0) > confidence_threshold:
            if predictions.get('ensemble_prediction') == 1:
                predictions['signal'] = 'BUY'
            else:
                predictions['signal'] = 'SELL'
        else:
            predictions['signal'] = 'HOLD'
        
        logging.info(f"Predictions for {symbol}:")
        logging.info(f"  Current Price: ${predictions['current_price']:,.2f}")
        logging.info(f"  Direction: {'UP' if predictions['rf_prediction'] == 1 else 'DOWN'}")
        logging.info(f"  Confidence: {predictions['rf_confidence']:.2%}")
        logging.info(f"  Signal: {predictions['signal']}")
        
        if 'ridge_predicted_price' in predictions:
            logging.info(f"  Predicted Price: ${predictions['ridge_predicted_price']:,.2f}")
        
        context['ti'].xcom_push(key=f'predictions_{symbol}', value=predictions)
        
        return predictions
        
    except Exception as e:
        logging.error(f"Prediction failed for {symbol}: {str(e)}")
        raise


def save_predictions_to_db(**context):
    """
    Save predictions to database for tracking and analysis
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(50),
                prediction_timestamp TIMESTAMP,
                current_price DECIMAL(20, 8),
                predicted_direction INTEGER,
                confidence DECIMAL(5, 4),
                prob_up DECIMAL(5, 4),
                prob_down DECIMAL(5, 4),
                predicted_return DECIMAL(10, 6),
                predicted_price DECIMAL(20, 8),
                signal VARCHAR(10),
                ensemble_prediction INTEGER,
                ensemble_confidence DECIMAL(5, 4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timestamp 
            ON predictions(symbol, prediction_timestamp DESC);
        """)
        conn.commit()
        
        # Collect predictions
        all_predictions = []
        for symbol in CRYPTO_SYMBOLS:
            predictions = context['ti'].xcom_pull(key=f'predictions_{symbol}')
            if predictions:
                all_predictions.append(predictions)
        
        if not all_predictions:
            logging.warning("No predictions to save")
            cursor.close()
            conn.close()
            return
        
        # Insert predictions
        insert_query = """
            INSERT INTO predictions 
            (symbol, prediction_timestamp, current_price, predicted_direction,
             confidence, prob_up, prob_down, predicted_return, predicted_price,
             signal, ensemble_prediction, ensemble_confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        saved_count = 0
        for pred in all_predictions:
            try:
                cursor.execute(insert_query, (
                    pred['symbol'],
                    pred['timestamp'],
                    pred['current_price'],
                    pred['rf_prediction'],
                    pred['rf_confidence'],
                    pred['rf_prob_up'],
                    pred['rf_prob_down'],
                    pred.get('ridge_predicted_return'),
                    pred.get('ridge_predicted_price'),
                    pred['signal'],
                    pred.get('ensemble_prediction'),
                    pred.get('ensemble_confidence')
                ))
                saved_count += 1
            except Exception as e:
                logging.error(f"Failed to save prediction: {str(e)}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"Saved {saved_count} predictions to database")
        
    except Exception as e:
        logging.error(f"Failed to save predictions: {str(e)}")
        raise


def calculate_prediction_accuracy(**context):
    """
    Calculate accuracy of past predictions by comparing with actual outcomes
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        # Check predictions from 1 hour ago and compare with actual price movement
        query = """
            WITH predictions_1h_ago AS (
                SELECT 
                    p.symbol,
                    p.prediction_timestamp,
                    p.current_price as predicted_at_price,
                    p.predicted_direction,
                    p.signal,
                    p.confidence
                FROM predictions p
                WHERE p.prediction_timestamp BETWEEN 
                    NOW() - INTERVAL '90 minutes' AND NOW() - INTERVAL '30 minutes'
            ),
            actual_outcomes AS (
                SELECT 
                    f.symbol,
                    f.timestamp,
                    f.close as actual_price
                FROM features f
                WHERE f.timestamp >= NOW() - INTERVAL '30 minutes'
            )
            SELECT 
                p.symbol,
                p.predicted_direction,
                p.signal,
                p.confidence,
                p.predicted_at_price,
                a.actual_price,
                CASE 
                    WHEN a.actual_price > p.predicted_at_price THEN 1
                    ELSE 0
                END as actual_direction,
                (a.actual_price - p.predicted_at_price) / p.predicted_at_price as actual_return
            FROM predictions_1h_ago p
            JOIN LATERAL (
                SELECT symbol, actual_price
                FROM actual_outcomes
                WHERE symbol = p.symbol
                ORDER BY timestamp DESC
                LIMIT 1
            ) a ON true;
        """
        
        df = hook.get_pandas_df(query)
        
        if df.empty:
            logging.info("No predictions old enough to validate yet")
            return
        
        # Calculate accuracy
        df['correct'] = (df['predicted_direction'] == df['actual_direction']).astype(int)
        
        accuracy = df['correct'].mean()
        
        logging.info("=" * 70)
        logging.info("PREDICTION ACCURACY ANALYSIS")
        logging.info("=" * 70)
        logging.info(f"Predictions analyzed: {len(df)}")
        logging.info(f"Overall Accuracy: {accuracy:.2%}")
        
        # Per-symbol accuracy
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            symbol_acc = symbol_df['correct'].mean()
            logging.info(f"  {symbol}: {symbol_acc:.2%} ({len(symbol_df)} predictions)")
        
        # Per-signal accuracy
        for signal in df['signal'].unique():
            signal_df = df[df['signal'] == signal]
            signal_acc = signal_df['correct'].mean()
            logging.info(f"Signal {signal}: {signal_acc:.2%} ({len(signal_df)} predictions)")
        
        logging.info("=" * 70)
        
        return accuracy
        
    except Exception as e:
        logging.error(f"Failed to calculate accuracy: {str(e)}")
        raise


def generate_prediction_report(**context):
    """
    Generate human-readable prediction report
    """
    try:
        # Get all predictions
        all_predictions = []
        for symbol in CRYPTO_SYMBOLS:
            predictions = context['ti'].xcom_pull(key=f'predictions_{symbol}')
            if predictions:
                all_predictions.append(predictions)
        
        if not all_predictions:
            logging.warning("No predictions to report")
            return
        
        logging.info("=" * 70)
        logging.info("CRYPTOCURRENCY PREDICTIONS")
        logging.info(f"Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logging.info("=" * 70)
        
        for pred in sorted(all_predictions, key=lambda x: x['symbol']):
            logging.info(f"\n{pred['symbol'].upper()}:")
            logging.info(f"  Current Price:    ${pred['current_price']:>10,.2f}")
            
            direction = "📈 UP" if pred['rf_prediction'] == 1 else "📉 DOWN"
            logging.info(f"  Predicted Move:   {direction}")
            logging.info(f"  Confidence:       {pred['rf_confidence']:>10.1%}")
            logging.info(f"  Signal:           {pred['signal']:>10}")
            
            if 'ridge_predicted_price' in pred:
                change = pred['ridge_predicted_price'] - pred['current_price']
                change_pct = (change / pred['current_price']) * 100
                logging.info(f"  Predicted Price:  ${pred['ridge_predicted_price']:>10,.2f} ({change_pct:+.2f}%)")
        
        logging.info("\n" + "=" * 70)
        
        # Summary
        buy_signals = sum(1 for p in all_predictions if p['signal'] == 'BUY')
        sell_signals = sum(1 for p in all_predictions if p['signal'] == 'SELL')
        hold_signals = sum(1 for p in all_predictions if p['signal'] == 'HOLD')
        
        logging.info("SIGNAL SUMMARY:")
        logging.info(f"  BUY:  {buy_signals}")
        logging.info(f"  SELL: {sell_signals}")
        logging.info(f"  HOLD: {hold_signals}")
        logging.info("=" * 70)
        
        return all_predictions
        
    except Exception as e:
        logging.error(f"Failed to generate report: {str(e)}")
        raise


# Define the DAG
with DAG(
    'crypto_prediction_inference',
    default_args=default_args,
    description='Generate predictions using trained ML models',
    schedule_interval='0 * * * *',  # Every hour
    catchup=False,
    max_active_runs=1,
    tags=['crypto', 'ml', 'inference', 'prediction'],
) as dag:
    
    # Check models exist
    check_models = PythonOperator(
        task_id='check_models_exist',
        python_callable=check_models_exist,
    )
    
    # Load latest features for each symbol (parallel)
    load_tasks = []
    for symbol in CRYPTO_SYMBOLS:
        task = PythonOperator(
            task_id=f'load_features_{symbol}',
            python_callable=load_latest_features,
            op_kwargs={'symbol': symbol},
        )
        load_tasks.append(task)
    
    # Generate predictions for each symbol (parallel)
    predict_tasks = []
    for symbol in CRYPTO_SYMBOLS:
        task = PythonOperator(
            task_id=f'predict_{symbol}',
            python_callable=generate_predictions,
            op_kwargs={'symbol': symbol},
        )
        predict_tasks.append(task)
    
    # Save predictions
    save_predictions = PythonOperator(
        task_id='save_predictions',
        python_callable=save_predictions_to_db,
    )
    
    # Calculate accuracy
    check_accuracy = PythonOperator(
        task_id='calculate_accuracy',
        python_callable=calculate_prediction_accuracy,
    )
    
    # Generate report
    prediction_report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_prediction_report,
    )
    
    # Dependencies
    check_models >> load_tasks
    
    for load_task, predict_task in zip(load_tasks, predict_tasks):
        load_task >> predict_task
    
    predict_tasks >> save_predictions >> [check_accuracy, prediction_report]