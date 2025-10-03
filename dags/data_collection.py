from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import requests
import pandas as pd
import logging

# Configuration
CRYPTO_SYMBOLS = ['bitcoin', 'ethereum', 'cardano', 'solana', 'polkadot']
CRYPTO_IDS = {
    'bitcoin': 'BTC',
    'ethereum': 'ETH',
    'cardano': 'ADA',
    'solana': 'SOL',
    'polkadot': 'DOT'
}
PAPRIKA_IDS = {
    'bitcoin': 'btc-bitcoin',
    'ethereum': 'eth-ethereum',
    'cardano': 'ada-cardano',
    'solana': 'sol-solana',
    'polkadot': 'dot-polkadot'
}

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=10),
}


def fetch_coinpaprika_data(symbol: str, **context):
    try:
        coin_id = PAPRIKA_IDS.get(symbol)
        if not coin_id:
            logging.warning(f"No CoinPaprika ID for {symbol}")
            return None
        
        url = f"https://api.coinpaprika.com/v1/tickers/{coin_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        record = {
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'price_usd': data['quotes']['USD']['price'],
            'market_cap': data['quotes']['USD'].get('market_cap'),
            'total_volume': data['quotes']['USD'].get('volume_24h'),
            'price_change_24h': data['quotes']['USD'].get('percent_change_24h'),
            'price_change_7d': data['quotes']['USD'].get('percent_change_7d'),
            'price_change_30d': data['quotes']['USD'].get('percent_change_30d'),
            'high_24h': data['quotes']['USD'].get('ath_price'),
            'low_24h': None,
            'circulating_supply': data.get('circulating_supply'),
            'total_supply': data.get('total_supply'),
        }
        
        logging.info(f"Successfully fetched data for {symbol}: ${record['price_usd']:,.2f}")
        
        # Push to XCom for next task
        context['ti'].xcom_push(key=f'market_data_{symbol}', value=record)
        
        return record
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch CoinPaprika data for {symbol}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error for {symbol}: {str(e)}")
        raise


def fetch_cryptocompare_ohlcv(symbol: str, **context):
    try:
        crypto_symbol = CRYPTO_IDS.get(symbol)
        if not crypto_symbol:
            logging.warning(f"No symbol mapping for {symbol}")
            return None
        
        url = "https://min-api.cryptocompare.com/data/v2/histohour"
        params = {
            'fsym': crypto_symbol,
            'tsym': 'USD',
            'limit': 24  # Last 24 hours
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['Response'] != 'Success':
            logging.error(f"CryptoCompare error for {symbol}: {data.get('Message')}")
            raise Exception(f"CryptoCompare API error: {data.get('Message')}")
        
        records = []
        for candle in data['Data']['Data']:
            records.append({
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(candle['time']),
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': float(candle['volumeto']),
                'trades': None
            })
        
        logging.info(f"Fetched {len(records)} OHLCV records for {symbol}")
        
        # Push to XCom
        context['ti'].xcom_push(key=f'ohlcv_{symbol}', value=records)
        
        return records
        
    except Exception as e:
        logging.error(f"Failed to fetch CryptoCompare data for {symbol}: {str(e)}")
        raise


def fetch_fear_greed_index(**context):
    try:
        url = "https://api.alternative.me/fng/"
        params = {'limit': 30}  # Last 30 days
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        records = []
        for item in data.get('data', []):
            records.append({
                'timestamp': datetime.fromtimestamp(int(item['timestamp'])),
                'value': int(item['value']),
                'value_classification': item['value_classification']
            })
        
        logging.info(f"Fetched {len(records)} Fear & Greed Index records")
        context['ti'].xcom_push(key='fear_greed_index', value=records)
        
        return records
        
    except Exception as e:
        logging.error(f"Failed to fetch Fear & Greed Index: {str(e)}")
        raise


def save_market_data_to_db(**context):
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        records = []
        for symbol in CRYPTO_SYMBOLS:
            data = context['ti'].xcom_pull(key=f'market_data_{symbol}')
            if data:
                records.append(data)
        
        if not records:
            logging.warning("No market data to save")
            return
        
        df = pd.DataFrame(records)
        
        # Insert into database
        insert_query = """
            INSERT INTO market_data 
            (symbol, timestamp, price_usd, market_cap, total_volume, 
             price_change_24h, price_change_7d, price_change_30d,
             high_24h, low_24h, circulating_supply, total_supply)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, timestamp) 
            DO UPDATE SET
                price_usd = EXCLUDED.price_usd,
                market_cap = EXCLUDED.market_cap,
                total_volume = EXCLUDED.total_volume,
                price_change_24h = EXCLUDED.price_change_24h,
                price_change_7d = EXCLUDED.price_change_7d,
                price_change_30d = EXCLUDED.price_change_30d,
                high_24h = EXCLUDED.high_24h,
                low_24h = EXCLUDED.low_24h,
                circulating_supply = EXCLUDED.circulating_supply,
                total_supply = EXCLUDED.total_supply;
        """
        
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        for _, row in df.iterrows():
            cursor.execute(insert_query, tuple(row))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"Successfully saved {len(records)} market data records to database")
        
    except Exception as e:
        logging.error(f"Failed to save market data: {str(e)}")
        raise


def save_ohlcv_to_db(**context):
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        all_records = []
        for symbol in CRYPTO_SYMBOLS:
            data = context['ti'].xcom_pull(key=f'ohlcv_{symbol}')
            if data:
                all_records.extend(data)
        
        if not all_records:
            logging.warning("No OHLCV data to save")
            return
        
        insert_query = """
            INSERT INTO ohlcv_data 
            (symbol, timestamp, open, high, low, close, volume, trades)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, timestamp) 
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                trades = EXCLUDED.trades;
        """
        
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        for record in all_records:
            cursor.execute(insert_query, (
                record['symbol'],
                record['timestamp'],
                record['open'],
                record['high'],
                record['low'],
                record['close'],
                record['volume'],
                record['trades']
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"Successfully saved {len(all_records)} OHLCV records to database")
        
    except Exception as e:
        logging.error(f"Failed to save OHLCV data: {str(e)}")
        raise


def save_sentiment_to_db(**context):
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        
        records = context['ti'].xcom_pull(key='fear_greed_index')
        if not records:
            logging.warning("No sentiment data to save")
            return
        
        insert_query = """
            INSERT INTO sentiment_data 
            (timestamp, fear_greed_value, fear_greed_classification)
            VALUES (%s, %s, %s)
            ON CONFLICT (timestamp) 
            DO UPDATE SET
                fear_greed_value = EXCLUDED.fear_greed_value,
                fear_greed_classification = EXCLUDED.fear_greed_classification;
        """
        
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        for record in records:
            cursor.execute(insert_query, (
                record['timestamp'],
                record['value'],
                record['value_classification']
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"Successfully saved {len(records)} sentiment records to database")
        
    except Exception as e:
        logging.error(f"Failed to save sentiment data: {str(e)}")
        raise


def validate_data_quality(**context):
    try:
        hook = PostgresHook(postgres_conn_id='crypto_db')
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        # Check data freshness (should have data from last 30 minutes)
        freshness_query = """
            SELECT symbol, MAX(timestamp) as last_update
            FROM market_data
            GROUP BY symbol
            HAVING MAX(timestamp) < NOW() - INTERVAL '30 minutes';
        """
        
        cursor.execute(freshness_query)
        stale_data = cursor.fetchall()
        
        if stale_data:
            logging.warning(f"Stale data detected for: {stale_data}")
        
        # Check for NULL prices
        null_check_query = """
            SELECT COUNT(*) as null_count
            FROM market_data
            WHERE price_usd IS NULL
            AND timestamp > NOW() - INTERVAL '1 hour';
        """
        
        cursor.execute(null_check_query)
        null_count = cursor.fetchone()[0]
        
        if null_count > 0:
            logging.warning(f"Found {null_count} NULL prices in last hour")
        
        # Check for extreme price changes (>50% in 1 hour - likely bad data)
        outlier_query = """
            WITH price_changes AS (
                SELECT 
                    symbol,
                    timestamp,
                    price_usd,
                    LAG(price_usd) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_price,
                    ABS((price_usd - LAG(price_usd) OVER (PARTITION BY symbol ORDER BY timestamp)) 
                        / NULLIF(LAG(price_usd) OVER (PARTITION BY symbol ORDER BY timestamp), 0)) as pct_change
                FROM market_data
                WHERE timestamp > NOW() - INTERVAL '2 hours'
            )
            SELECT symbol, timestamp, price_usd, prev_price, pct_change
            FROM price_changes
            WHERE pct_change > 0.5;
        """
        
        cursor.execute(outlier_query)
        outliers = cursor.fetchall()
        
        if outliers:
            logging.warning(f"Potential outliers detected: {outliers}")
        
        logging.info("Data quality validation completed")
        
        # Push validation results to XCom for monitoring
        validation_results = {
            'stale_data_count': len(stale_data),
            'null_price_count': null_count,
            'outlier_count': len(outliers),
            'validation_timestamp': datetime.utcnow().isoformat()
        }
        
        cursor.close()
        conn.close()
        
        context['ti'].xcom_push(key='validation_results', value=validation_results)
        
        return validation_results
        
    except Exception as e:
        logging.error(f"Data quality validation failed: {str(e)}")
        raise


# Define the DAG
with DAG(
    'crypto_data_collection',
    default_args=default_args,
    description='Collect cryptocurrency market data every 15 minutes',
    schedule_interval='*/15 * * * *',  # Every 15 minutes
    catchup=False,
    max_active_runs=1,
    tags=['crypto', 'data-collection', 'ml-pipeline'],
) as dag:
    
    # Create tables if they don't exist
    create_tables = PostgresOperator(
        task_id='create_tables',
        postgres_conn_id='crypto_db',
        sql="""
            -- Market data table
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                price_usd DECIMAL(20, 8),
                market_cap BIGINT,
                total_volume BIGINT,
                price_change_24h DECIMAL(10, 4),
                price_change_7d DECIMAL(10, 4),
                price_change_30d DECIMAL(10, 4),
                high_24h DECIMAL(20, 8),
                low_24h DECIMAL(20, 8),
                circulating_supply BIGINT,
                total_supply BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            );
            
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
            ON market_data(symbol, timestamp DESC);
            
            -- OHLCV data table
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DECIMAL(20, 8),
                high DECIMAL(20, 8),
                low DECIMAL(20, 8),
                close DECIMAL(20, 8),
                volume DECIMAL(20, 8),
                trades INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            );
            
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp 
            ON ohlcv_data(symbol, timestamp DESC);
            
            -- Sentiment data table
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL UNIQUE,
                fear_greed_value INTEGER,
                fear_greed_classification VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp 
            ON sentiment_data(timestamp DESC);
        """
    )
    
    # Task group for parallel market data collection
    with TaskGroup('collect_market_data', tooltip='Collect data from CoinPaprika') as market_data_group:
        market_data_tasks = []
        for symbol in CRYPTO_SYMBOLS:
            task = PythonOperator(
                task_id=f'fetch_market_data_{symbol}',
                python_callable=fetch_coinpaprika_data,
                op_kwargs={'symbol': symbol},
            )
            market_data_tasks.append(task)
    
    # Task group for OHLCV data
    with TaskGroup('collect_ohlcv_data', tooltip='Collect OHLCV from CryptoCompare') as ohlcv_group:
        ohlcv_tasks = []
        for symbol in CRYPTO_SYMBOLS:
            task = PythonOperator(
                task_id=f'fetch_ohlcv_{symbol}',
                python_callable=fetch_cryptocompare_ohlcv,
                op_kwargs={'symbol': symbol},
            )
            ohlcv_tasks.append(task)
    
    # Fetch sentiment data
    fetch_sentiment = PythonOperator(
        task_id='fetch_fear_greed_index',
        python_callable=fetch_fear_greed_index,
    )
    
    # Save data to database
    save_market_data = PythonOperator(
        task_id='save_market_data',
        python_callable=save_market_data_to_db,
    )
    
    save_ohlcv = PythonOperator(
        task_id='save_ohlcv',
        python_callable=save_ohlcv_to_db,
    )
    
    save_sentiment = PythonOperator(
        task_id='save_sentiment',
        python_callable=save_sentiment_to_db,
    )
    
    # Data quality validation
    validate_quality = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality,
    )
    
    # Define task dependencies
    create_tables >> [market_data_group, ohlcv_group, fetch_sentiment]
    
    market_data_group >> save_market_data
    ohlcv_group >> save_ohlcv
    fetch_sentiment >> save_sentiment
    
    [save_market_data, save_ohlcv, save_sentiment] >> validate_quality