"""
Extended Historical Data Backfill
Fetches up to 2 years of historical data to improve model training
"""

import requests
import psycopg2
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CRYPTO_SYMBOLS = ['bitcoin', 'ethereum', 'cardano', 'solana', 'polkadot']
CRYPTO_IDS = {
    'bitcoin': 'BTC',
    'ethereum': 'ETH',
    'cardano': 'ADA',
    'solana': 'SOL',
    'polkadot': 'DOT'
}

DB_CONFIG = {
    'host': 'postgres',
    'database': 'airflow',
    'user': 'airflow',
    'password': 'airflow',
    'port': 5432
}

# Fetch more data - CryptoCompare allows 2000 hours per request
# We'll make multiple requests to get up to 1 year of data
HOURS_TO_FETCH = 8760  # 365 days * 24 hours


def fetch_historical_ohlcv_batch(symbol, to_timestamp, limit=2000):
    """
    Fetch a batch of historical data from CryptoCompare
    """
    crypto_symbol = CRYPTO_IDS.get(symbol)
    if not crypto_symbol:
        return []
    
    try:
        url = "https://min-api.cryptocompare.com/data/v2/histohour"
        params = {
            'fsym': crypto_symbol,
            'tsym': 'USD',
            'limit': limit,
            'toTs': int(to_timestamp.timestamp())
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data['Response'] != 'Success':
            logging.error(f"API error for {symbol}: {data.get('Message')}")
            return []
        
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
        
        return records
        
    except Exception as e:
        logging.error(f"Failed to fetch batch for {symbol}: {str(e)}")
        return []


def fetch_all_historical_data(symbol, hours=8760):
    """
    Fetch multiple batches to get extended history
    """
    all_records = []
    batch_size = 2000  # Max allowed by API
    num_batches = (hours // batch_size) + 1
    
    current_time = datetime.utcnow()
    
    logging.info(f"Fetching {hours} hours ({hours//24} days) of data for {symbol} in {num_batches} batches...")
    
    for i in range(num_batches):
        # Calculate timestamp for this batch
        hours_back = min(batch_size * (i + 1), hours)
        to_timestamp = current_time - timedelta(hours=hours_back - batch_size)
        
        logging.info(f"  Batch {i+1}/{num_batches}: Fetching data up to {to_timestamp.strftime('%Y-%m-%d')}")
        
        records = fetch_historical_ohlcv_batch(symbol, to_timestamp, batch_size)
        
        if records:
            all_records.extend(records)
            logging.info(f"    Retrieved {len(records)} records")
        
        # Rate limiting - be nice to the API
        time.sleep(2)
        
        if len(records) < batch_size:
            logging.info(f"  Reached end of available data")
            break
    
    # Remove duplicates and sort
    unique_records = {}
    for record in all_records:
        key = (record['symbol'], record['timestamp'])
        unique_records[key] = record
    
    sorted_records = sorted(unique_records.values(), key=lambda x: x['timestamp'])
    
    logging.info(f"Total unique records for {symbol}: {len(sorted_records)}")
    
    return sorted_records


def save_ohlcv_data(conn, records):
    """Save OHLCV data to database"""
    if not records:
        return 0
    
    cursor = conn.cursor()
    
    insert_query = """
        INSERT INTO ohlcv_data 
        (symbol, timestamp, open, high, low, close, volume, trades)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, timestamp) DO NOTHING;
    """
    
    saved_count = 0
    for record in records:
        try:
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
            saved_count += 1
        except Exception as e:
            logging.error(f"Failed to save record: {str(e)}")
            continue
    
    conn.commit()
    cursor.close()
    
    return saved_count


def main():
    print("=" * 80)
    print("EXTENDED HISTORICAL DATA BACKFILL")
    print("=" * 80)
    print(f"Fetching {HOURS_TO_FETCH} hours ({HOURS_TO_FETCH//24} days) of data")
    print(f"Symbols: {', '.join(CRYPTO_SYMBOLS)}")
    print()
    
    # Connect to database
    print("Connecting to database...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logging.info("Connected to database")
    except Exception as e:
        logging.error(f"Database connection failed: {str(e)}")
        return
    
    # Fetch and save data for each symbol
    total_saved = 0
    
    for symbol in CRYPTO_SYMBOLS:
        print(f"\n{'='*80}")
        print(f"Processing {symbol.upper()}")
        print('='*80)
        
        # Fetch all historical data
        records = fetch_all_historical_data(symbol, HOURS_TO_FETCH)
        
        if records:
            # Save to database
            saved = save_ohlcv_data(conn, records)
            total_saved += saved
            
            logging.info(f"Saved {saved} new records for {symbol}")
            
            # Show date range
            if records:
                earliest = min(r['timestamp'] for r in records)
                latest = max(r['timestamp'] for r in records)
                logging.info(f"Date range: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
        else:
            logging.warning(f"No data retrieved for {symbol}")
    
    # Summary
    print("\n" + "=" * 80)
    print("BACKFILL COMPLETE")
    print("=" * 80)
    print(f"Total new records saved: {total_saved:,}")
    
    # Verify data
    cursor = conn.cursor()
    
    print("\nDatabase Statistics:")
    cursor.execute("""
        SELECT 
            symbol,
            COUNT(*) as total_records,
            MIN(timestamp) as earliest,
            MAX(timestamp) as latest,
            EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp)))/86400 as days_span
        FROM ohlcv_data
        GROUP BY symbol
        ORDER BY symbol;
    """)
    
    for row in cursor.fetchall():
        symbol, count, earliest, latest, days = row
        print(f"\n{symbol.upper()}:")
        print(f"  Total records: {count:,}")
        print(f"  Date range: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
        print(f"  Time span: {days:.0f} days")
    
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("1. Run feature engineering: crypto_feature_engineering")
    print("2. Retrain models: crypto_model_training")
    print("=" * 80)


if __name__ == "__main__":
    main()