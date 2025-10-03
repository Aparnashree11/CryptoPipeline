"""
Cryptocurrency Feature Engineering Pipeline
Generates technical indicators and features for ML models
Runs after data collection to prepare data for training
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

# Configuration
CRYPTO_SYMBOLS = ['bitcoin', 'ethereum', 'cardano', 'solana', 'polkadot']

default_args = {
    'owner': 'ml-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=15),
}


def calculate_technical_indicators(df):
    """
    Calculate technical indicators for trading
    
    Returns dataframe with:
    - Moving averages (SMA, EMA)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - Volatility measures
    """
    
    # Sort by timestamp
    df = df.sort_values('timestamp').copy()
    
    # Simple Moving Averages
    df['sma_7'] = df['close'].rolling(window=7, min_periods=1).mean()
    df['sma_14'] = df['close'].rolling(window=14, min_periods=1).mean()
    df['sma_30'] = df['close'].rolling(window=30, min_periods=1).mean()
    
    # Exponential Moving Averages
    df['ema_7'] = df['close'].ewm(span=7, adjust=False).mean()
    df['ema_14'] = df['close'].ewm(span=14, adjust=False).mean()
    df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['close'].rolling(window=20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    
    # Volatility measures
    df['volatility_7'] = df['close'].rolling(window=7, min_periods=1).std()
    df['volatility_14'] = df['close'].rolling(window=14, min_periods=1).std()
    df['volatility_30'] = df['close'].rolling(window=30, min_periods=1).std()
    
    # Price momentum
    df['momentum_1'] = df['close'].pct_change(1)
    df['momentum_3'] = df['close'].pct_change(3)
    df['momentum_7'] = df['close'].pct_change(7)
    
    # Volume features
    df['volume_sma_7'] = df['volume'].rolling(window=7, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_7'].replace(0, np.nan)
    
    # Price range features
    df['high_low_range'] = df['high'] - df['low']
    df['high_low_ratio'] = df['high'] / df['low'].replace(0, np.nan)
    
    return df


def calculate_lag_features(df, lags=[1, 3, 7, 14]):
    """
    Create lag features (past values as features)
    """
    df = df.copy()
    
    for lag in lags:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'high_lag_{lag}'] = df['high'].shift(lag)
        df[f'low_lag_{lag}'] = df['low'].shift(lag)
    
    return df


def calculate_rolling_statistics(df, windows=[7, 14, 30]):
    """
    Calculate rolling statistical features
    """
    df = df.copy()
    
    for window in windows:
        # Rolling mean
        df[f'close_mean_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
        
        # Rolling std
        df[f'close_std_{window}'] = df['close'].rolling(window=window, min_periods=1).std()
        
        # Rolling max/min
        df[f'close_max_{window}'] = df['close'].rolling(window=window, min_periods=1).max()
        df[f'close_min_{window}'] = df['close'].rolling(window=window, min_periods=1).min()
        
        # Distance from max/min
        df[f'dist_from_max_{window}'] = (df['close'] - df[f'close_max_{window}']) / df[f'close_max_{window}']
        df[f'dist_from_min_{window}'] = (df['close'] - df[f'close_min_{window}']) / df[f'close_min_{window}']
    
    return df


def create_target_variables(df, horizons=[1, 3, 7]):
    """
    Create target variables for prediction
    - Future returns at different horizons
    - Binary up/down classification targets
    """
    df = df.copy()
    
    for horizon in horizons:
        # Future returns
        df[f'target_return_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon)
        
        # Binary classification (up=1, down=0)
        df[f'target_direction_{horizon}'] = (df[f'target_return_{horizon}'] > 0).astype(int)
        
        # Categorical (strong_up, up, neutral, down, strong_down)
        df[f'target_category_{horizon}'] = pd.cut(
            df[f'target_return_{horizon}'],
            bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
            labels=['strong_down', 'down', 'neutral', 'up', 'strong_up']
        )
    
    return df


def extract_and_engineer_features(symbol: str, **context):
    """
    Main feature engineering function for a single crypto symbol
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        # Fetch OHLCV data
        query = f"""
            SELECT 
                symbol,
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM ohlcv_data
            WHERE symbol = '{symbol}'
            AND timestamp >= NOW() - INTERVAL '90 days'
            ORDER BY timestamp ASC;
        """
        
        df = hook.get_pandas_df(query)
        
        if df.empty:
            logging.warning(f"No data found for {symbol}")
            return None
        
        logging.info(f"Processing {len(df)} records for {symbol}")
        
        # 1. Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # 2. Add lag features
        df = calculate_lag_features(df)
        
        # 3. Add rolling statistics
        df = calculate_rolling_statistics(df)
        
        # 4. Create target variables
        df = create_target_variables(df)
        
        # 5. Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        # 6. Add market sentiment features
        sentiment_query = """
            SELECT 
                timestamp,
                fear_greed_value
            FROM sentiment_data
            WHERE timestamp >= NOW() - INTERVAL '90 days'
            ORDER BY timestamp ASC;
        """
        
        sentiment_df = hook.get_pandas_df(sentiment_query)
        
        if not sentiment_df.empty:
            # Merge sentiment data
            df['date'] = df['timestamp'].dt.date
            sentiment_df['date'] = sentiment_df['timestamp'].dt.date
            sentiment_agg = sentiment_df.groupby('date')['fear_greed_value'].first().reset_index()
            df = df.merge(sentiment_agg, on='date', how='left')
            df['fear_greed_value'] = df['fear_greed_value'].fillna(method='ffill')
            df.drop('date', axis=1, inplace=True)
        
        # 7. Drop rows with NaN in critical columns (from lag features)
        # Keep last 60 days of fully featured data
        df = df.tail(60 * 24)  # 60 days * 24 hours
        
        logging.info(f"Generated {len(df.columns)} features for {symbol}")
        logging.info(f"Feature columns: {df.columns.tolist()}")
        
        # Push to XCom
        df['timestamp'] = df['timestamp'].astype(str)
        context['ti'].xcom_push(key=f'features_{symbol}', value=df.to_dict('records'))
        
        return len(df)
        
    except Exception as e:
        logging.error(f"Feature engineering failed for {symbol}: {str(e)}")
        raise


def save_features_to_db(**context):
    """
    Save engineered features to database
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        # Create features table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                
                -- OHLCV
                open DECIMAL(20, 8),
                high DECIMAL(20, 8),
                low DECIMAL(20, 8),
                close DECIMAL(20, 8),
                volume DECIMAL(20, 8),
                
                -- Technical Indicators
                sma_7 DECIMAL(20, 8),
                sma_14 DECIMAL(20, 8),
                sma_30 DECIMAL(20, 8),
                ema_7 DECIMAL(20, 8),
                ema_14 DECIMAL(20, 8),
                ema_30 DECIMAL(20, 8),
                rsi_14 DECIMAL(10, 4),
                macd DECIMAL(20, 8),
                macd_signal DECIMAL(20, 8),
                macd_histogram DECIMAL(20, 8),
                bb_middle DECIMAL(20, 8),
                bb_upper DECIMAL(20, 8),
                bb_lower DECIMAL(20, 8),
                bb_width DECIMAL(20, 8),
                
                -- Volatility
                volatility_7 DECIMAL(20, 8),
                volatility_14 DECIMAL(20, 8),
                volatility_30 DECIMAL(20, 8),
                
                -- Momentum
                momentum_1 DECIMAL(10, 6),
                momentum_3 DECIMAL(10, 6),
                momentum_7 DECIMAL(10, 6),
                
                -- Volume features
                volume_sma_7 DECIMAL(20, 8),
                volume_ratio DECIMAL(10, 4),
                
                -- Price range
                high_low_range DECIMAL(20, 8),
                high_low_ratio DECIMAL(10, 4),
                
                -- Lag features
                close_lag_1 DECIMAL(20, 8),
                close_lag_3 DECIMAL(20, 8),
                close_lag_7 DECIMAL(20, 8),
                close_lag_14 DECIMAL(20, 8),
                
                -- Rolling stats
                close_mean_7 DECIMAL(20, 8),
                close_std_7 DECIMAL(20, 8),
                close_max_7 DECIMAL(20, 8),
                close_min_7 DECIMAL(20, 8),
                
                -- Time features
                hour INTEGER,
                day_of_week INTEGER,
                day_of_month INTEGER,
                month INTEGER,
                quarter INTEGER,
                
                -- Sentiment
                fear_greed_value INTEGER,
                
                -- Target variables
                target_return_1 DECIMAL(10, 6),
                target_return_3 DECIMAL(10, 6),
                target_return_7 DECIMAL(10, 6),
                target_direction_1 INTEGER,
                target_direction_3 INTEGER,
                target_direction_7 INTEGER,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            );
            
            CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp 
            ON features(symbol, timestamp DESC);
        """)
        
        conn.commit()
        
        # Collect all features from XCom
        all_features = []
        for symbol in CRYPTO_SYMBOLS:
            features = context['ti'].xcom_pull(key=f'features_{symbol}')
            if features:
                all_features.extend(features)
        
        if not all_features:
            logging.warning("No features to save")
            cursor.close()
            conn.close()
            return
        
        logging.info(f"Saving {len(all_features)} feature records to database")
        
        # Prepare insert query with only the columns that exist
        insert_query = """
            INSERT INTO features 
            (symbol, timestamp, open, high, low, close, volume,
             sma_7, sma_14, sma_30, ema_7, ema_14, ema_30,
             rsi_14, macd, macd_signal, macd_histogram,
             bb_middle, bb_upper, bb_lower, bb_width,
             volatility_7, volatility_14, volatility_30,
             momentum_1, momentum_3, momentum_7,
             volume_sma_7, volume_ratio, high_low_range, high_low_ratio,
             close_lag_1, close_lag_3, close_lag_7, close_lag_14,
             close_mean_7, close_std_7, close_max_7, close_min_7,
             hour, day_of_week, day_of_month, month, quarter,
             fear_greed_value,
             target_return_1, target_return_3, target_return_7,
             target_direction_1, target_direction_3, target_direction_7)
            VALUES (
                %(symbol)s, %(timestamp)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s,
                %(sma_7)s, %(sma_14)s, %(sma_30)s, %(ema_7)s, %(ema_14)s, %(ema_30)s,
                %(rsi_14)s, %(macd)s, %(macd_signal)s, %(macd_histogram)s,
                %(bb_middle)s, %(bb_upper)s, %(bb_lower)s, %(bb_width)s,
                %(volatility_7)s, %(volatility_14)s, %(volatility_30)s,
                %(momentum_1)s, %(momentum_3)s, %(momentum_7)s,
                %(volume_sma_7)s, %(volume_ratio)s, %(high_low_range)s, %(high_low_ratio)s,
                %(close_lag_1)s, %(close_lag_3)s, %(close_lag_7)s, %(close_lag_14)s,
                %(close_mean_7)s, %(close_std_7)s, %(close_max_7)s, %(close_min_7)s,
                %(hour)s, %(day_of_week)s, %(day_of_month)s, %(month)s, %(quarter)s,
                %(fear_greed_value)s,
                %(target_return_1)s, %(target_return_3)s, %(target_return_7)s,
                %(target_direction_1)s, %(target_direction_3)s, %(target_direction_7)s
            )
            ON CONFLICT (symbol, timestamp) 
            DO UPDATE SET
                close = EXCLUDED.close,
                rsi_14 = EXCLUDED.rsi_14,
                macd = EXCLUDED.macd,
                target_return_1 = EXCLUDED.target_return_1,
                target_direction_1 = EXCLUDED.target_direction_1;
        """
        
        saved_count = 0
        for record in all_features:
            try:
                cursor.execute(insert_query, record)
                saved_count += 1
            except Exception as e:
                logging.error(f"Failed to save record: {str(e)}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"Successfully saved {saved_count} feature records")
        
        # Push metrics to XCom
        context['ti'].xcom_push(key='features_saved', value=saved_count)
        
    except Exception as e:
        logging.error(f"Failed to save features: {str(e)}")
        raise


def generate_feature_summary(**context):
    """
    Generate summary statistics about engineered features
    """
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        # Get feature statistics
        query = """
            SELECT 
                symbol,
                COUNT(*) as record_count,
                MIN(timestamp) as earliest_data,
                MAX(timestamp) as latest_data,
                AVG(rsi_14) as avg_rsi,
                AVG(volatility_7) as avg_volatility,
                AVG(fear_greed_value) as avg_sentiment
            FROM features
            WHERE timestamp > NOW() - INTERVAL '7 days'
            GROUP BY symbol
            ORDER BY symbol;
        """
        
        df = hook.get_pandas_df(query)
        
        logging.info("Feature Engineering Summary:")
        logging.info(f"\n{df.to_string()}")

        if 'earliest_data' in df.columns:
            df['earliest_data'] = df['earliest_data'].astype(str)
        if 'latest_data' in df.columns:
            df['latest_data'] = df['latest_data'].astype(str)
        
        summary = df.to_dict('records')
        context['ti'].xcom_push(key='feature_summary', value=summary)
        
        return summary
        
    except Exception as e:
        logging.error(f"Failed to generate feature summary: {str(e)}")
        raise

def check_data_exists(**context):
    """Check if we have recent data before processing"""
    hook = PostgresHook(postgres_conn_id='crypto_db')
    
    query = """
        SELECT COUNT(*) 
        FROM ohlcv_data 
        WHERE timestamp > NOW() - INTERVAL '2 hours';
    """
    
    result = hook.get_first(query)[0]
    
    if result < 10:  # Need at least 10 records
        raise Exception(f"Insufficient data ({result} records). Run data collection first.")
    
    logging.info(f"✓ Found {result} recent records")
    return result


# Define the DAG
with DAG(
    'crypto_feature_engineering',
    default_args=default_args,
    description='Engineer features from crypto data for ML',
    schedule_interval='0 */1 * * *',  # Every hour
    catchup=False,
    max_active_runs=1,
    tags=['crypto', 'features', 'ml'],
) as dag:
    
    # Check if data exists
    check_data = PythonOperator(
        task_id='check_data_exists',
        python_callable=check_data_exists,
    )
    
    # Feature engineering tasks (parallel for each symbol)
    feature_tasks = []
    for symbol in CRYPTO_SYMBOLS:
        task = PythonOperator(
            task_id=f'engineer_features_{symbol}',
            python_callable=extract_and_engineer_features,
            op_kwargs={'symbol': symbol},
        )
        feature_tasks.append(task)
    
    # Save all features to database
    save_features = PythonOperator(
        task_id='save_features_to_db',
        python_callable=save_features_to_db,
    )
    
    # Generate summary
    feature_summary = PythonOperator(
        task_id='generate_feature_summary',
        python_callable=generate_feature_summary,
    )
    
    # Define dependencies
    check_data >> feature_tasks >> save_features >> feature_summary