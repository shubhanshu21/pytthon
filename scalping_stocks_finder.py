import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta
import time

# Set your maximum price for filtering stocks
MAX_PRICE = 1000  # Change this value as needed (in INR)

# Increase timeout and add retry functionality
CONNECTION_TIMEOUT = 5  # Faster timeout to avoid long waits
MAX_RETRIES = 3  # Number of times to retry failed requests
RETRY_DELAY = 2  # Seconds to wait between retries
TOTAL_STOCKS = 1  # stocks to fetch

def get_with_retries(url, headers=None, session=None, timeout=CONNECTION_TIMEOUT):
    """Helper function to get URL with retries"""
    if session is None:
        session = requests.Session()
        
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                print(f"Retry attempt {attempt+1} for {url}")
                time.sleep(RETRY_DELAY)
            response = session.get(url, headers=headers, timeout=timeout)
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                return None
            print(f"Attempt {attempt+1} failed: {str(e)}")
    return None

def get_nifty100_symbols():
    """
    Fetch the list of NIFTY 100 stocks automatically using updated sources
    
    Returns:
    -------
    list
        List of NIFTY 100 stock symbols
    """
    try:
        # Method 1: Try alternate approach - NSE stock list with filter for larger companies
        try:
            print("Trying to fetch all NSE stocks first...")
            url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = get_with_retries(url, headers=headers)
            if response and response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                
                # Filter for companies with high market cap
                if 'SYMBOL' in df.columns and 'SERIES' in df.columns:
                    # Keep only stocks in EQ series (removes derivatives etc.)
                    df = df[df['SERIES'] == 'EQ']
                    all_symbols = df['SYMBOL'].tolist()
                    
                    # Some of the known Nifty 100 constituent stocks
                    known_nifty100 = [
                        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK", "ITC", 
                        "KOTAKBANK", "SBIN", "BAJFINANCE", "BHARTIARTL", "ASIANPAINT", 
                        "LT", "AXISBANK", "MARUTI", "HCLTECH", "SUNPHARMA", "WIPRO", "ULTRACEMCO",
                        "ADANIENT", "ADANIPORTS", "POWERGRID", "TITAN", "NTPC", "ONGC", "M&M", 
                        "TATAMOTORS", "COALINDIA", "HINDALCO", "DRREDDY"
                    ]
                    
                    # Start with known stocks
                    nifty100_approx = [s for s in known_nifty100 if s in all_symbols]
                    
                    # Then get market cap data for others
                    remaining_needed = 100 - len(nifty100_approx)
                    if remaining_needed > 0:
                        print(f"Found {len(nifty100_approx)} known Nifty 100 stocks, getting {remaining_needed} more...")
                        
                        # Get a batch of stocks to check their market cap
                        # (avoiding checking all NSE stocks to save time)
                        potential_stocks = [s for s in all_symbols if s not in nifty100_approx][:200]
                        
                        market_caps = {}
                        for i, symbol in enumerate(potential_stocks):
                            if i % 10 == 0:
                                print(f"Checking market cap for stocks {i}-{min(i+10, len(potential_stocks))} of {len(potential_stocks)}...")
                            try:
                                ticker = yf.Ticker(f"{symbol}.NS")
                                info = ticker.info
                                if 'marketCap' in info and info['marketCap'] is not None:
                                    market_caps[symbol] = info['marketCap']
                            except Exception as e:
                                continue
                        
                        # Sort by market cap and add to our list
                        sorted_by_market_cap = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
                        additional_stocks = [s[0] for s in sorted_by_market_cap[:remaining_needed]]
                        nifty100_approx.extend(additional_stocks)
                    
                    print(f"Successfully approximated Nifty 100 with {len(nifty100_approx)} stocks")
                    return nifty100_approx
        except Exception as e:
            print(f"Failed to create Nifty 100 approximation: {e}")
        
        # Method 2: NSE's API endpoint for Nifty 100 constituents
        try:
            print("Trying NSE API...")
            url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate, br'
            }
            
            session = requests.Session()
            # First request to get cookies
            cookies_resp = get_with_retries("https://www.nseindia.com", headers=headers, session=session)
            if cookies_resp:
                # Then request the actual data
                response = get_with_retries(url, headers=headers, session=session)
                if response and response.status_code == 200:
                    data = response.json()
                    if 'data' in data:
                        symbols = [item['symbol'] for item in data['data']]
                        print(f"Successfully fetched {len(symbols)} Nifty 100 symbols from NSE API")
                        return symbols
        except Exception as e:
            print(f"Failed to fetch from NSE API: {e}")
        
        # If all methods fail, fall back to the hardcoded list
        print("All automated methods failed. Using backup hardcoded list...")
        
    except Exception as e:
        print(f"Error in fetching NIFTY 100 symbols: {str(e)}")
    
    # Fallback to a recent hardcoded list
    nifty100_symbols = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK", "ITC", 
        "KOTAKBANK", "SBIN", "BAJFINANCE", "BHARTIARTL", "ASIANPAINT", 
        "LT", "AXISBANK", "MARUTI", "HCLTECH", "SUNPHARMA", "WIPRO", "ULTRACEMCO",
        "ADANIENT", "ADANIPORTS", "ADANIPOWER", "BAJAJFINSV", "POWERGRID", "TITAN", 
        "NTPC", "ONGC", "M&M", "TATAMOTORS", "COALINDIA", "HINDALCO", "DRREDDY", 
        "TECHM", "TATACONSUM", "HDFCLIFE", "BRITANNIA", "DIVISLAB", "INDUSINDBK", 
        "JSWSTEEL", "ADANIGREEN", "BAJAJ-AUTO", "TATASTEEL", "CIPLA", "EICHERMOT", 
        "NESTLEIND", "BPCL", "GRASIM", "IOC", "HEROMOTOCO", "APOLLOHOSP", "SBILIFE", 
        "UPL", "SHREECEM", "DMART", "BERGEPAINT", "PIDILITIND", "HAVELLS", "DABUR", 
        "SIEMENS", "DLF", "ABB", "ADANIENSOL", "AMBUJACEM", "BANKBARODA", "BOSCHLTD", 
        "CHOLAFIN", "GAIL", "GODREJCP", "HDFCAMC", "HINDPETRO", "MARICO", "NAUKRI", 
        "ICICIGI", "NYKAA", "PAYTM", "PIIND", "PGHH", "SAIL", 
        "SRF", "TATAPOWER", "TIINDIA", "TORNTPHARM", "VEDL", "ZOMATO", "ZYDUSLIFE",
        "BAJAJHLDNG", "BIOCON", "COLPAL", "GLAND", "INDIGO", "JINDALSTEL", "JUBLFOOD",
        "LUPIN", "MRF", "MOTHERSON", "PNB", "TATACHEM", "AUROPHARMA", "BANDHANBNK", "BEL"
    ]
    print(f"Using hardcoded list with {len(nifty100_symbols)} symbols")
    return nifty100_symbols

def get_best_scalping_stocks(num_stocks=TOTAL_STOCKS, max_price=MAX_PRICE):
    """
    Get the best stocks for scalping from the NIFTY 100 index
    
    Parameters:
    ----------
    num_stocks : int
        Number of top stocks to return
    max_price : float
        Maximum price per share to consider (in INR)
        
    Returns:
    -------
    list
        List of stock symbols that are best for scalping
    """
    # Get list of NIFTY 100 stocks
    print("Fetching NIFTY 100 stocks...")
    nifty100_symbols = get_nifty100_symbols()
    
    if not nifty100_symbols:
        print("Failed to fetch NIFTY 100 symbols.")
        return []
    
    print(f"Successfully retrieved {len(nifty100_symbols)} NIFTY 100 symbols.")
    
    # Append .NS suffix for NSE stocks
    nifty100_yahoo = [symbol + ".NS" for symbol in nifty100_symbols]
    
    # Define time period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days data
    
    results = []
    filtered_by_price = []
    
    print(f"Analyzing {len(nifty100_yahoo)} stocks with price filter of ₹{max_price}...")
    
    # Process in batches to avoid API limitations
    batch_size = 25
    for i in range(0, len(nifty100_yahoo), batch_size):
        batch = nifty100_yahoo[i:i+batch_size]
        try:
            # Get historical data
            data = yf.download(
                batch,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                group_by='ticker',
                auto_adjust=True,
                progress=False  # Disable progress bar for cleaner output
            )
            
            for ticker in batch:
                try:
                    # Skip if data is empty for this ticker
                    if ticker not in data.columns.levels[0]:
                        continue
                        
                    stock_data = data[ticker]
                    
                    if len(stock_data) < 10:  # Need at least 10 days of data
                        continue
                    
                    # Get current price
                    current_price = stock_data['Close'].iloc[-1]
                    
                    # Apply price filter
                    if current_price > max_price:
                        continue
                    
                    filtered_by_price.append(ticker[:-3])
                    
                    # Calculate metrics
                    avg_volume = stock_data['Volume'].mean()
                    daily_returns = stock_data['Close'].pct_change().dropna()
                    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
                    
                    # Average daily range (in percentage)
                    avg_range = ((stock_data['High'] - stock_data['Low']) / stock_data['Close']).mean() * 100
                    
                    # Average True Range (ATR) - another volatility measure
                    high_low = stock_data['High'] - stock_data['Low']
                    high_close = abs(stock_data['High'] - stock_data['Close'].shift())
                    low_close = abs(stock_data['Low'] - stock_data['Close'].shift())
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = ranges.max(axis=1)
                    atr = true_range.rolling(window=14).mean().iloc[-1]
                    
                    # Relative ATR (ATR as a percentage of price)
                    rel_atr = (atr / current_price) * 100
                    
                    # Momentum (ratio of current price to 10-day moving average)
                    ma10 = stock_data['Close'].rolling(window=10).mean().iloc[-1]
                    momentum = current_price / ma10 - 1
                    
                    # Volume consistency (coefficient of variation - lower is more consistent)
                    vol_consistency = stock_data['Volume'].std() / stock_data['Volume'].mean()
                    
                    # Calculate scalping score (custom formula)
                    scalping_score = (
                        np.log(avg_volume) * 0.35 +    # Weight for volume
                        volatility * 100 * 0.2 +       # Weight for volatility
                        abs(momentum) * 100 * 0.15 +   # Weight for momentum
                        avg_range * 0.15 +             # Weight for average daily range
                        rel_atr * 0.15                 # Weight for relative ATR
                    )
                    
                    results.append({
                        'Symbol': ticker[:-3],  # Remove .NS suffix
                        'Price': current_price,
                        'Avg_Volume': avg_volume,
                        'Volatility': volatility * 100,  # Convert to percentage
                        'Avg_Range_%': avg_range,
                        'ATR': atr,
                        'ATR_%': rel_atr,
                        'Momentum_%': momentum * 100,  # Convert to percentage
                        'Vol_Consistency': vol_consistency,
                        'Scalping_Score': scalping_score
                    })
                    
                except Exception as e:
                    # Skip detailed error messages for cleaner output
                    continue
                    
        except Exception as e:
            print(f"Error fetching batch {i}-{i+batch_size}: {str(e)}")
            continue
            
    # Convert to DataFrame
    if not results:
        print(f"No valid stocks found under price ₹{max_price}. Try increasing the price limit.")
        return []
        
    df_results = pd.DataFrame(results)
    
    # Print price filter statistics
    print(f"\nFiltered {len(filtered_by_price)} stocks under price ₹{max_price} out of {len(nifty100_symbols)} Nifty 100 stocks")
    
    # Sort by scalping score
    df_results = df_results.sort_values('Scalping_Score', ascending=False)
    
    # Format the DataFrame for better readability
    df_display = df_results.copy()
    df_display['Price'] = df_display['Price'].map('₹{:.2f}'.format)
    df_display['Volatility'] = df_display['Volatility'].map('{:.2f}%'.format)
    df_display['Avg_Range_%'] = df_display['Avg_Range_%'].map('{:.2f}%'.format)
    df_display['ATR_%'] = df_display['ATR_%'].map('{:.2f}%'.format)
    df_display['Momentum_%'] = df_display['Momentum_%'].map('{:.2f}%'.format)
    
    # Print detailed results
    print("\nDetailed analysis of top stocks for scalping:")
    print(df_display[['Symbol', 'Price', 'Volatility', 'Avg_Range_%', 'ATR_%', 'Momentum_%', 'Scalping_Score']].head(num_stocks).to_string(index=False))
    
    # Get top N symbols
    top_symbols = df_results['Symbol'].head(num_stocks).tolist()
    
    return top_symbols

if __name__ == "__main__":
    # You can change the price filter here
    MAX_PRICE = 1000  # Default maximum price in INR
    
    # Get command line arguments if provided
    import sys
    if len(sys.argv) > 1:
        try:
            MAX_PRICE = float(sys.argv[1])
        except:
            pass
    
    print(f"Running analysis with maximum price filter of ₹{MAX_PRICE}")
    
    # Get top 10 stocks for scalping under the price limit
    top_scalping_stocks = get_best_scalping_stocks(10, MAX_PRICE)
    
    print("\nTop 10 NIFTY 100 stocks for scalping under ₹{:.2f}:".format(MAX_PRICE))
    print(top_scalping_stocks)
    
    # Display as a Python list
    print("\nList format:")
    print(f"symbols = {top_scalping_stocks}")