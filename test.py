import yfinance as yf
import datetime

# TCS NSE symbol
symbol = "TCS.NS"

# yfinance only allows 1m data for the past 7 days, max 1-day range recommended
end = datetime.datetime.now()
start = end - datetime.timedelta(days=7)

# Download 1-minute data
df = yf.download(symbol, start=start, end=end, interval="1m", progress=False)

print(df)
