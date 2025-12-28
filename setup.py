from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timezone
import json
import pandas as pd
#from polars as pl

# Gets API info
with open('APIData.json', 'r') as file:
      data = json.load(file)

keys = data['alpaca_paper']
api_k = keys["api_key"]
api_s = keys["api_secret"]

# Creates instances in the script to get data from API
data_client = StockHistoricalDataClient(api_k, api_s)
account = TradingClient(api_k, api_s, paper=True)

# Gets the data
stock = input("Enter the stock symbol you want to check. ")
request_params = StockBarsRequest(
    symbol_or_symbols=[f"{stock}"],
    timeframe=TimeFrame.Minute,
    start=datetime(2024, 8, 1, tzinfo=timezone.utc),
    end=datetime(2024, 8, 31, tzinfo=timezone.utc)
)

bars = data_client.get_stock_bars(request_params)
df = bars.df.reset_index()

# Data Manipulation before Rust and C++
df = df[df["symbol"] == stock]

path_name = "market_data_prediction_test"
df.to_csv(f"Market_Data/{path_name}.csv", index=True)
print(f"Saved this data to the .csv '{path_name}.csv'")

# Check account info
print(f"Equity: ${account.get_account().equity}")
print(f"Buying Power: ${account.get_account().buying_power}")

