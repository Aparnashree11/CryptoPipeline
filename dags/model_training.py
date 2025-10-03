"""
Cryptocurrency Model Training Pipeline
Trains multiple ML models to predict crypto price movements
Uses features from feature engineering pipeline
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
MODEL_DIR = '/opt/airflow/models'  # Store models here

default_args = {
    'owner': 'ml-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),
}


def check_features_exist(**context):
    """Check if we have enough features for training"""
    hook = PostgresHook(postgres_conn_id='crypto_db')
    
    query = """
        SELECT symbol, COUNT(*) as count
        FROM features
        WHERE timestamp > NOW() - INTERVAL '30 days'
        AND target_direction_1 IS NOT NULL
        GROUP BY symbol;
    """
    
    df = hook.get_pandas_df(query)
    
    if df.empty or df['count'].min() < 100:
        raise Exception(f"Insufficient features for training. Need at least 100 records per symbol.")
    
    logging.info(f"Feature counts per symbol:\n{df.to_string()}")
    return df.to_dict('records')


def prepare_training_data(symbol: str, **context):
    """
    Load and prepare data for training
    - Split into train/validation/test
    - Handle missing values
    - Scale features
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        # Load features
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
                fear_greed_value,
                target_return_1,
                target_direction_1
            FROM features
            WHERE symbol = '{symbol}'
            AND timestamp > NOW() - INTERVAL '30 days'
            AND target_direction_1 IS NOT NULL
            ORDER BY timestamp ASC;
        """
        
        df = hook.get_pandas_df(query)
        
        if len(df) < 100:
            logging.warning(f"Not enough data for {symbol}: {len(df)} records")
            return None
        
        logging.info(f"Loaded {len(df)} records for {symbol}")
        
        # Drop timestamp for training
        df = df.drop('timestamp', axis=1)
        
        # Separate features and targets
        target_cols = ['target_return_1', 'target_direction_1']
        feature_cols = [col for col in df.columns if col not in target_cols and col != 'close']
        
        X = df[feature_cols]
        y_regression = df['target_return_1']
        y_classification = df['target_direction_1']
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Train/validation/test split (70/15/15)
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_reg_train = y_regression[:train_end]
        y_reg_val = y_regression[train_end:val_end]
        y_reg_test = y_regression[val_end:]
        
        y_clf_train = y_classification[:train_end]
        y_clf_val = y_classification[train_end:val_end]
        y_clf_test = y_classification[val_end:]
        
        # Normalize features (using training stats only)
        feature_means = X_train.mean()
        feature_stds = X_train.std().replace(0, 1)  # Avoid division by zero
        
        X_train_scaled = (X_train - feature_means) / feature_stds
        X_val_scaled = (X_val - feature_means) / feature_stds
        X_test_scaled = (X_test - feature_means) / feature_stds
        
        data = {
            'X_train': X_train_scaled.values.tolist(),
            'X_val': X_val_scaled.values.tolist(),
            'X_test': X_test_scaled.values.tolist(),
            'y_reg_train': y_reg_train.values.tolist(),
            'y_reg_val': y_reg_val.values.tolist(),
            'y_reg_test': y_reg_test.values.tolist(),
            'y_clf_train': y_clf_train.values.tolist(),
            'y_clf_val': y_clf_val.values.tolist(),
            'y_clf_test': y_clf_test.values.tolist(),
            'feature_names': feature_cols,
            'feature_means': feature_means.to_dict(),
            'feature_stds': feature_stds.to_dict(),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test)
        }
        
        logging.info(f"Data prepared for {symbol}:")
        logging.info(f"  Train: {data['n_train']} samples")
        logging.info(f"  Val:   {data['n_val']} samples")
        logging.info(f"  Test:  {data['n_test']} samples")
        logging.info(f"  Features: {len(feature_cols)}")
        
        context['ti'].xcom_push(key=f'data_{symbol}', value=data)
        
        return data
        
    except Exception as e:
        logging.error(f"Data preparation failed for {symbol}: {str(e)}")
        raise


def train_models(symbol: str, **context):
    """
    Train multiple models:
    - Logistic Regression (Classification)
    - Random Forest (Classification)
    - Gradient Boosting (Classification)
    - Linear Regression (Regression)
    """
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Get prepared data
        data = context['ti'].xcom_pull(key=f'data_{symbol}')
        
        if not data:
            logging.warning(f"No data available for {symbol}")
            return None
        
        # Convert back to numpy arrays
        X_train = np.array(data['X_train'])
        X_val = np.array(data['X_val'])
        X_test = np.array(data['X_test'])
        
        y_clf_train = np.array(data['y_clf_train'])
        y_clf_val = np.array(data['y_clf_val'])
        y_clf_test = np.array(data['y_clf_test'])
        
        y_reg_train = np.array(data['y_reg_train'])
        y_reg_val = np.array(data['y_reg_val'])
        y_reg_test = np.array(data['y_reg_test'])
        
        logging.info(f"Training models for {symbol}...")
        
        models = {}
        results = {}
        
        # 1. Logistic Regression (Classification)
        logging.info("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_clf_train)
        
        lr_val_pred = lr.predict(X_val)
        lr_val_acc = accuracy_score(y_clf_val, lr_val_pred)
        lr_val_prec = precision_score(y_clf_val, lr_val_pred, zero_division=0)
        lr_val_rec = recall_score(y_clf_val, lr_val_pred, zero_division=0)
        lr_val_f1 = f1_score(y_clf_val, lr_val_pred, zero_division=0)
        
        models['logistic_regression'] = lr
        results['logistic_regression'] = {
            'accuracy': float(lr_val_acc),
            'precision': float(lr_val_prec),
            'recall': float(lr_val_rec),
            'f1': float(lr_val_f1)
        }
        
        logging.info(f"  Logistic Regression - Accuracy: {lr_val_acc:.4f}, F1: {lr_val_f1:.4f}")
        
        # 2. Random Forest (Classification)
        logging.info("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_clf_train)
        
        rf_val_pred = rf.predict(X_val)
        rf_val_acc = accuracy_score(y_clf_val, rf_val_pred)
        rf_val_prec = precision_score(y_clf_val, rf_val_pred, zero_division=0)
        rf_val_rec = recall_score(y_clf_val, rf_val_pred, zero_division=0)
        rf_val_f1 = f1_score(y_clf_val, rf_val_pred, zero_division=0)
        
        models['random_forest'] = rf
        results['random_forest'] = {
            'accuracy': float(rf_val_acc),
            'precision': float(rf_val_prec),
            'recall': float(rf_val_rec),
            'f1': float(rf_val_f1),
            'feature_importance': {
                name: float(importance) 
                for name, importance in zip(data['feature_names'], rf.feature_importances_)
            }
        }
        
        logging.info(f"  Random Forest - Accuracy: {rf_val_acc:.4f}, F1: {rf_val_f1:.4f}")
        
        # 3. Gradient Boosting (Classification)
        logging.info("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb.fit(X_train, y_clf_train)
        
        gb_val_pred = gb.predict(X_val)
        gb_val_acc = accuracy_score(y_clf_val, gb_val_pred)
        gb_val_prec = precision_score(y_clf_val, gb_val_pred, zero_division=0)
        gb_val_rec = recall_score(y_clf_val, gb_val_pred, zero_division=0)
        gb_val_f1 = f1_score(y_clf_val, gb_val_pred, zero_division=0)
        
        models['gradient_boosting'] = gb
        results['gradient_boosting'] = {
            'accuracy': float(gb_val_acc),
            'precision': float(gb_val_prec),
            'recall': float(gb_val_rec),
            'f1': float(gb_val_f1)
        }
        
        logging.info(f"  Gradient Boosting - Accuracy: {gb_val_acc:.4f}, F1: {gb_val_f1:.4f}")
        
        # 4. Ridge Regression (Regression)
        logging.info("Training Ridge Regression...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_reg_train)
        
        ridge_val_pred = ridge.predict(X_val)
        ridge_val_mse = mean_squared_error(y_reg_val, ridge_val_pred)
        ridge_val_mae = mean_absolute_error(y_reg_val, ridge_val_pred)
        ridge_val_r2 = r2_score(y_reg_val, ridge_val_pred)
        
        models['ridge_regression'] = ridge
        results['ridge_regression'] = {
            'mse': float(ridge_val_mse),
            'mae': float(ridge_val_mae),
            'r2': float(ridge_val_r2)
        }
        
        logging.info(f"  Ridge Regression - MAE: {ridge_val_mae:.6f}, R2: {ridge_val_r2:.4f}")
        
        # Save models to disk
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        
        for model_name, model in models.items():
            model_path = f"{MODEL_DIR}/{symbol}_{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logging.info(f"Saved {model_name} to {model_path}")
        
        # Save feature scaling parameters
        scaling_params = {
            'feature_means': data['feature_means'],
            'feature_stds': data['feature_stds'],
            'feature_names': data['feature_names']
        }
        
        scaling_path = f"{MODEL_DIR}/{symbol}_scaling_params.json"
        with open(scaling_path, 'w') as f:
            json.dump(scaling_params, f)
        
        # Prepare results for XCom (no datetime objects)
        results['symbol'] = symbol
        results['timestamp'] = datetime.utcnow().isoformat()
        results['n_features'] = len(data['feature_names'])
        results['n_train'] = data['n_train']
        results['n_val'] = data['n_val']
        
        context['ti'].xcom_push(key=f'results_{symbol}', value=results)
        
        logging.info(f"Training completed for {symbol}")
        
        return results
        
    except Exception as e:
        logging.error(f"Model training failed for {symbol}: {str(e)}")
        raise


def save_model_metrics(**context):
    """Save model performance metrics to database"""
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        # Create model metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(50),
                model_name VARCHAR(100),
                metric_name VARCHAR(50),
                metric_value DECIMAL(10, 6),
                training_timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_model_metrics_symbol 
            ON model_metrics(symbol, training_timestamp DESC);
        """)
        conn.commit()
        
        # Collect results from all symbols
        all_results = []
        for symbol in CRYPTO_SYMBOLS:
            results = context['ti'].xcom_pull(key=f'results_{symbol}')
            if results:
                all_results.append(results)
        
        if not all_results:
            logging.warning("No model results to save")
            cursor.close()
            conn.close()
            return
        
        # Insert metrics
        insert_query = """
            INSERT INTO model_metrics 
            (symbol, model_name, metric_name, metric_value, training_timestamp)
            VALUES (%s, %s, %s, %s, %s);
        """
        
        saved_count = 0
        for result in all_results:
            symbol = result['symbol']
            timestamp = result['timestamp']
            
            for model_name in ['logistic_regression', 'random_forest', 'gradient_boosting', 'ridge_regression']:
                if model_name in result:
                    metrics = result[model_name]
                    for metric_name, metric_value in metrics.items():
                        if metric_name != 'feature_importance':  # Skip dict values
                            cursor.execute(insert_query, (
                                symbol,
                                model_name,
                                metric_name,
                                metric_value,
                                timestamp
                            ))
                            saved_count += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"Saved {saved_count} model metrics to database")
        
    except Exception as e:
        logging.error(f"Failed to save model metrics: {str(e)}")
        raise


def generate_training_report(**context):
    """Generate summary report of model training"""
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        query = """
            SELECT 
                symbol,
                model_name,
                metric_name,
                metric_value
            FROM model_metrics
            WHERE training_timestamp = (
                SELECT MAX(training_timestamp) 
                FROM model_metrics
            )
            ORDER BY symbol, model_name, metric_name;
        """
        
        df = hook.get_pandas_df(query)
        
        if df.empty:
            logging.warning("No metrics found")
            return
        
        logging.info("=" * 70)
        logging.info("MODEL TRAINING REPORT")
        logging.info("=" * 70)
        
        # Group by symbol and model
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            logging.info(f"\n{symbol.upper()}:")
            
            for model in symbol_df['model_name'].unique():
                model_df = symbol_df[symbol_df['model_name'] == model]
                logging.info(f"  {model}:")
                
                for _, row in model_df.iterrows():
                    logging.info(f"    {row['metric_name']}: {row['metric_value']:.4f}")
        
        logging.info("\n" + "=" * 70)
        
        # Find best models
        if 'accuracy' in df['metric_name'].values:
            acc_df = df[df['metric_name'] == 'accuracy']
            best = acc_df.loc[acc_df['metric_value'].idxmax()]
            logging.info(f"Best Classification Model: {best['model_name']} ({best['symbol']}) - {best['metric_value']:.4f}")
        
        if 'r2' in df['metric_name'].values:
            r2_df = df[df['metric_name'] == 'r2']
            best = r2_df.loc[r2_df['metric_value'].idxmax()]
            logging.info(f"Best Regression Model: {best['model_name']} ({best['symbol']}) - {best['metric_value']:.4f}")
        
        logging.info("=" * 70)
        
        return df.to_dict('records')
        
    except Exception as e:
        logging.error(f"Failed to generate training report: {str(e)}")
        raise


# Define the DAG
with DAG(
    'crypto_model_training',
    default_args=default_args,
    description='Train ML models on crypto features',
    schedule_interval='0 0 * * *',  # Daily at midnight
    catchup=False,
    max_active_runs=1,
    tags=['crypto', 'ml', 'training'],
) as dag:
    
    # Check features exist
    check_features = PythonOperator(
        task_id='check_features_exist',
        python_callable=check_features_exist,
    )
    
    # Prepare data for each symbol (parallel)
    prep_tasks = []
    for symbol in CRYPTO_SYMBOLS:
        task = PythonOperator(
            task_id=f'prepare_data_{symbol}',
            python_callable=prepare_training_data,
            op_kwargs={'symbol': symbol},
        )
        prep_tasks.append(task)
    
    # Train models for each symbol (parallel)
    train_tasks = []
    for symbol in CRYPTO_SYMBOLS:
        task = PythonOperator(
            task_id=f'train_models_{symbol}',
            python_callable=train_models,
            op_kwargs={'symbol': symbol},
        )
        train_tasks.append(task)
    
    # Save metrics
    save_metrics = PythonOperator(
        task_id='save_model_metrics',
        python_callable=save_model_metrics,
    )
    
    # Generate report
    training_report = PythonOperator(
        task_id='generate_training_report',
        python_callable=generate_training_report,
    )
    
    # Dependencies
    check_features >> prep_tasks
    
    for prep_task, train_task in zip(prep_tasks, train_tasks):
        prep_task >> train_task
    
    train_tasks >> save_metrics >> training_report