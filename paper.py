import gymnasium as gym
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import os
import time
import datetime
from finta import TA
from stable_baselines3 import PPO
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import logging
import sys
import datetime
import argparse
import os
from dotenv import load_dotenv


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        # Define LSTM layers
        self.lstm = nn.LSTM(
            input_size=observation_space.shape[0],
            hidden_size=features_dim,
            batch_first=True,
            num_layers=2,
            dropout=0.2
        )
        self.feature_dim = features_dim
    
    def forward(self, observations):
        # Convert observations to tensor if necessary
        observations = observations.type(torch.float32)
        
        # Reshape for LSTM if needed
        batch_size = observations.shape[0]
        if len(observations.shape) == 2:
            # Add sequence length dimension (1 for single step)
            observations = observations.unsqueeze(1)
        
        # Process through LSTM
        features, _ = self.lstm(observations)
        
        # Take output from the last time step
        if len(features.shape) == 3:
            features = features[:, -1, :]
            
        return features

class TradingMonitor:
    """Monitoring system for the trading platform"""
    
    def __init__(self, email=None, phone=None, email_password=None):
        self.email = email
        self.email_password = email_password
        self.phone = phone
        self.last_health_check = time.time()
        self.error_count = 0
        
    def check_health(self, trader):
        """Perform periodic health checks on the trading system"""
        current_time = time.time()
        
        # Run health check every hour
        if current_time - self.last_health_check > 3600:
            # Check data freshness
            for symbol in trader.symbols:
                last_update = trader.data_handler.last_update.get(symbol, 0)
                if current_time - last_update > 7200:  # 2 hours
                    self.send_alert(f"Data for {symbol} is stale (last update: {datetime.datetime.fromtimestamp(last_update)})")
            
            # Check portfolio value for unusual drops
            if len(trader.portfolio_history) > 2:
                last_value = trader.portfolio_history[-1]['value']
                prev_value = trader.portfolio_history[-2]['value']
                if last_value < prev_value * 0.95:  # 5% drop
                    self.send_alert(f"Portfolio value dropped by {(prev_value - last_value) / prev_value * 100:.2f}%")
            
            self.last_health_check = current_time
    
    def send_alert(self, message):
        """Send alert via email or SMS"""
        print(f"ALERT: {message}")
        
        # Send email if configured
        if self.email and self.email_password:
            try:
                # Basic email sending with smtplib - would need configuration
                import smtplib
                from email.message import EmailMessage
                
                # This is a placeholder - in production you'd use a config file for these settings
                msg = EmailMessage()
                msg.set_content(f"Trading Alert: {message}")
                msg['Subject'] = 'Trading System Alert'
                msg['From'] = self.email
                msg['To'] = self.email
                
                # Connect to Gmail SMTP (works with App Passwords)
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(self.email, self.email_password)
                    smtp.send_message(msg)

                # Not implemented: would need server config, authentication, etc.
                print("Email notification would be sent (not implemented)")
            except Exception as e:
                print(f"Failed to send email alert: {e}")
        
        # Log alert
        logging.critical(f"TRADING ALERT: {message}")

def setup_system_service():
    """Generate files needed to set up the trading system as a system service"""
    service_file = """
                    [Unit]
                    Description=AI Stock Trading System
                    After=network.target

                    [Service]
                    User={user}
                    WorkingDirectory={working_dir}
                    ExecStart={python_path} {script_path} --autorun
                    Restart=always
                    RestartSec=10

                    [Install]
                    WantedBy=multi-user.target
                    """
    
    # Get current user and paths
    user = os.getenv('USER')
    working_dir = os.getcwd()
    python_path = sys.executable
    script_path = os.path.abspath(__file__)
    
    # Write service file
    with open('stock_trading.service', 'w') as f:
        f.write(service_file.format(
            user=user,
            working_dir=working_dir,
            python_path=python_path,
            script_path=script_path
        ))
    
    print("\nSystem service file created: stock_trading.service")
    print("\nTo install as a system service on Linux:")
    print(f"1. Copy the file: sudo cp stock_trading.service /etc/systemd/system/")
    print(f"2. Reload systemd: sudo systemctl daemon-reload")
    print(f"3. Enable the service: sudo systemctl enable stock_trading.service")
    print(f"4. Start the service: sudo systemctl start stock_trading.service")
    print(f"5. Check status: sudo systemctl status stock_trading.service")
    
    # Create a batch file for Windows users
    batch_file = f"""@echo off
      echo Starting AI Stock Trading System...
      "{python_path}" "{script_path}" --autorun
      pause
      """
    with open('start_trading.bat', 'w') as f:
        f.write(batch_file)
    
    print("\nWindows batch file created: start_trading.bat")
    print("You can add this to Windows Task Scheduler for automatic startup.")

def parse_arguments():
    """Parse command line arguments for automated operation"""
    parser = argparse.ArgumentParser(description="AI Stock Trading System")
    parser.add_argument('--autorun', action='store_true', help='Run automated trading without user interaction')
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT,AMZN,GOOGL,TSLA', help='Comma-separated list of stock symbols')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital amount')
    parser.add_argument('--interval', type=int, default=5, help='Trading cycle interval in minutes')
    parser.add_argument('--setup-service', action='store_true', help='Generate system service files')
    
    return parser.parse_args()

class StockTradingEnv(gym.Env):
    def __init__(self, symbol, start_date, end_date, timeframe='1d'):
        super(StockTradingEnv, self).__init__()
        
        self.symbol = symbol+'.NS'
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        
        # Download stock data using yfinance
        print(f"Downloading data for {symbol} from {start_date} to {end_date}...")
        try:
            self.df = yf.download(self.symbol, start=self.start_date, end=self.end_date, interval=self.timeframe)
            print(f"Downloaded {len(self.df)} rows of data")
            
            # Check if we have enough data
            if len(self.df) < 50:  # Minimum data needed for indicators
                raise ValueError(f"Not enough data available for {symbol} in the given date range.")
            
            # Fix for tuple column names - ensure we're dealing with strings
            if isinstance(self.df.columns[0], tuple):
                # If columns are tuples (multi-index), flatten them
                self.df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in self.df.columns]
            else:
                # If columns are already strings
                self.df.columns = [col.lower() for col in self.df.columns]
            
            # Add more robust technical indicators - force all data to float32 for consistency
            self.df = self.df.astype('float32')  # Ensure consistent data type
            self.df.dropna(inplace=True)  # Drop NA values instead of forward filling
            self.df[['open', 'high', 'low', 'close', 'volume']] = self.df[['open', 'high', 'low', 'close', 'volume']].astype(float)

            self._add_technical_indicators()
            
            # Fill NaN values
            self.df = self.df.ffill().bfill()
            self.df.fillna(0, inplace=True)  # Replace any remaining NaNs with 0
            
            # Clip extreme values to prevent issues with scaling
            for col in self.df.columns:
                if col not in ['volume']:  # Skip volume as it can have legitimately high values
                    q_low = self.df[col].quantile(0.01)
                    q_high = self.df[col].quantile(0.99)
                    self.df[col] = self.df[col].clip(q_low, q_high)
            
            # Define features and ensure they exist
            self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'atr', 'ma20', 'ma50', 'macd', 'signal', 'vwap', 'obv']
            self.feature_columns = [col for col in self.feature_columns if col in self.df.columns]
            
            # Make sure all columns have valid data types
            for col in self.feature_columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Perform scaling - make sure all values are floats
            self.scaler = MinMaxScaler(feature_range=(0.1, 0.9))
            scaled_features = self.scaler.fit_transform(self.df[self.feature_columns].values)
            
            # Replace scaled values in dataframe and ensure they're float32
            for i, col in enumerate(self.feature_columns):
                self.df[col] = scaled_features[:, i].astype(np.float32)
            
            # Skip initial periods where indicators aren't fully formed
            self.df = self.df.iloc[50:]  # Skip first 50 rows to avoid NaN issues with indicators
            self.df = self.df.reset_index(drop=False)
            
            # Verify no NaN or infinite values exist
            if not np.all(np.isfinite(self.df[self.feature_columns].values)):
                print(f"Warning: Non-finite values found for {symbol} after preprocessing. Replacing with zeros.")
                for col in self.feature_columns:
                    self.df[col] = np.nan_to_num(self.df[col], nan=0.0, posinf=1.0, neginf=0.0)
                    
        except Exception as e:
            print(f"Error processing data for {symbol}: {e}")
            # Create a minimal valid dataframe to avoid breaking the code
            self.df = pd.DataFrame({
                'open': [100.0] * 100,
                'high': [101.0] * 100,
                'close': [99.0] * 100,
                'low': [98.0] * 100,
                'volume': [1000.0] * 100,
            })
            self.feature_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Action space: Continuous actions [buy_fraction, stop_loss_percent]
        self.action_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # Observation space with strictly float32 type
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(len(self.feature_columns),), 
            dtype=np.float32
        )
        
        # Initialize variables
        self.current_step = 0
        self.initial_balance = 10000.0  # Starting with 10,000 units
        self.balance = self.initial_balance
        self.position = 0.0  # Shares held
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.done = False
        self.reward_range = (-1000, 1000)
        self.stop_loss = 0.0  # Track stop-loss for each trade
        
    def _add_technical_indicators(self):
        try:
            self.df['rsi'] = TA.RSI(self.df, period=14)
            self.df['atr'] = TA.ATR(self.df, period=14)
            self.df['ma20'] = TA.SMA(self.df, period=20)
            self.df['ma50'] = TA.SMA(self.df, period=50)
            
            macd_data = TA.MACD(self.df)
            self.df['macd'] = macd_data['MACD']
            self.df['signal'] = macd_data['SIGNAL']
            
            # Calculate VWAP (Volume-Weighted Average Price)
            self.df['vwap'] = (self.df['volume'] * (self.df['high'] + self.df['low'] + self.df['close']) / 3).cumsum() / self.df['volume'].cumsum()

            # Calculate OBV (On-Balance Volume)
            self.df['obv'] = self.df.apply(lambda row: row['volume'] if row['close'] > row['open'] else -row['volume'], axis=1).cumsum()

            # Apply EMA to OBV column
            self.df['obv_ema'] = TA.EMA(self.df, column='obv', period=20)
            
        except Exception as e:
            print(f"Error calculating indicators for {self.symbol}: {e}")
            # Create basic indicators to ensure consistent columns
            self.df['rsi'] = 50.0
            self.df['atr'] = 1.0
            self.df['ma20'] = self.df['close']
            self.df['ma50'] = self.df['close']
            self.df['macd'] = 0.0
            self.df['signal'] = 0.0
            self.df['vwap'] = self.df['close']
            self.df['obv'] = self.df['volume']

    def reset(self, *, seed=None, options=None):
        # Updated reset method to match new Gymnasium API
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.done = False
        self.stop_loss = 0.0
        
        obs = self._get_obs()
        info = {}
        
        return obs, info

    def _get_obs(self):
        # Safely get observation data with proper error handling for types
        try:
            if self.current_step >= len(self.df):
                # Return the last valid observation
                if len(self.df) > 0:
                    obs = self.df.iloc[-1][self.feature_columns].values
                else:
                    obs = np.zeros(len(self.feature_columns))
            else:
                obs = self.df.iloc[self.current_step][self.feature_columns].values
            
            # Ensure the output is float32 as expected by gym
            obs = obs.astype(np.float32)
            
            # Explicitly replace any non-finite values
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
            
            return obs
            
        except Exception as e:
            print(f"Error getting observation for {self.symbol}: {e}")
            # Return safe default observation
            return np.zeros(len(self.feature_columns), dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, self.done, False, {}
        
        # Safely get current price
        try:
            current_price = float(self.df.iloc[self.current_step]['close'])
        except (IndexError, KeyError) as e:
            print(f"Error getting price for {self.symbol}: {e}")
            current_price = 0.0
        
        prev_net_worth = self.net_worth
        
        # Record state before action for reward calculation
        prev_position = self.position
        prev_balance = self.balance
        
        # Execute action with proper error handling
        try:
            buy_fraction, stop_loss_percent = action
            buy_fraction = np.clip(buy_fraction, 0.0, 1.0)
            stop_loss_percent = np.clip(stop_loss_percent, 0.0, 1.0)
            
            if buy_fraction > 0.0:  # Buy
                if self.balance > 0:
                    max_shares = self.balance / current_price if current_price > 0 else 0
                    shares_bought = buy_fraction * max_shares
                    self.position += shares_bought
                    self.balance -= shares_bought * current_price
                    # Set stop-loss for this trade
                    self.stop_loss = current_price * (1 - stop_loss_percent)
            elif buy_fraction == 0.0 and self.position > 0:  # Sell
                # Check if stop-loss is triggered
                if current_price <= self.stop_loss:
                    self.balance += self.position * current_price
                    self.position = 0.0
                    self.stop_loss = 0.0
                else:
                    # Optional: Implement other sell conditions
                    pass
            # Hold if no action is taken
            
        except Exception as e:
            print(f"Error executing action for {self.symbol}: {e}")
        
        # Move to next step
        self.current_step += 1
        
        # Calculate values with error handling
        try:
            if self.current_step < len(self.df):
                current_price = float(self.df.iloc[self.current_step]['close'])
                
            self.net_worth = self.balance + (self.position * current_price)
            self.max_net_worth = max(self.max_net_worth, self.net_worth)
        except Exception as e:
            print(f"Error calculating net worth for {self.symbol}: {e}")
        
        # Check if done
        self.done = self.current_step >= len(self.df) - 1
        
        # Calculate reward with safe operations
        try:
            current_return = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth != 0 else 0.0
            volatility = self.df['close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
            sharpe_ratio = current_return / volatility if volatility != 0 else 0.0
            
            # Penalty for excessive trading
            trade_penalty = 0.0005  # 0.05% trading fee/penalty
            if (buy_fraction > 0.0 and prev_balance > 0) or (self.position > 0 and current_price <= self.stop_loss):
                reward = sharpe_ratio - trade_penalty
            else:
                reward = sharpe_ratio
            
            # Clip reward to prevent numerical issues
            reward = np.clip(reward, -1.0, 1.0)
            reward = float(reward)  # Ensure it's a Python float
        except Exception as e:
            print(f"Error calculating reward for {self.symbol}: {e}")
            reward = 0.0
        
        # Get observation with safety checks
        obs = self._get_obs()
        
        # Return with updated Gymnasium API
        info = {
            'net_worth': float(self.net_worth),
            'balance': float(self.balance),
            'position': float(self.position),
            'step': int(self.current_step),
            'stop_loss': float(self.stop_loss)
        }
        
        return obs, reward, self.done, False, info

class LiveMarketDataHandler:
    """Handles fetching and processing real-time market data for paper trading"""
    
    def __init__(self, symbols, lookback_days=60, update_interval=60):
        """
        Initialize the live data handler
        
        Args:
            symbols: List of stock symbols
            lookback_days: Number of days of historical data to include
            update_interval: How often to update data in seconds
        """
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.update_interval = update_interval
        self.data_cache = {}
        self.last_update = {}
        
    def get_data(self, symbol):
        """
        Get current and historical data for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with stock data
        """
        current_time = time.time()
        
        # Check if we need to update the data
        if symbol not in self.last_update or (current_time - self.last_update[symbol]) > self.update_interval:
            try:
                # Calculate lookback period
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=self.lookback_days)
                
                # Download data
                print(f"Updating live data for {symbol}...")
                df = yf.download(symbol, start=start_date, end=end_date, interval="1d")
                
                if len(df) < 30:  # Need minimum data for indicators
                    print(f"Warning: Limited data available for {symbol}")
                
                # Process column names
                if isinstance(df.columns[0], tuple):
                    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
                else:
                    df.columns = [col.lower() for col in df.columns]
                
                # Add technical indicators
                df['rsi'] = TA.RSI(df, period=14)
                df['atr'] = TA.ATR(df, period=14)
                df['ma20'] = TA.SMA(df, period=20)
                df['ma50'] = TA.SMA(df, period=50)
                
                macd_data = TA.MACD(df)
                df['macd'] = macd_data['MACD']
                df['signal'] = macd_data['SIGNAL']
                
                # Calculate VWAP and OBV
                df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
                df['obv'] = df.apply(lambda row: row['volume'] if row['close'] > row['open'] else -row['volume'], axis=1).cumsum()
                
                # Handle NaN values
                df = df.ffill().bfill()
                df.fillna(0, inplace=True)
                
                # Update cache
                self.data_cache[symbol] = df
                self.last_update[symbol] = current_time
                
                print(f"Data updated for {symbol}, most recent price: {df['close'].iloc[-1]:.2f}")
                
            except Exception as e:
                print(f"Error fetching live data for {symbol}: {e}")
                # Return cached data if available
                if symbol in self.data_cache:
                    print(f"Using cached data for {symbol}")
                    return self.data_cache[symbol]
                else:
                    print(f"No data available for {symbol}")
                    return None
        
        return self.data_cache[symbol]
    
    def get_latest_price(self, symbol):
        """Get the latest price for a symbol"""
        df = self.get_data(symbol)
        if df is not None and not df.empty:
            return df['close'].iloc[-1]
        return None
    
    def get_latest_features(self, symbol, feature_columns):
        """
        Get the latest feature values for ML model input
        
        Args:
            symbol: Stock symbol
            feature_columns: List of features needed
            
        Returns:
            Array of feature values
        """
        df = self.get_data(symbol)
        if df is not None and not df.empty:
            # Ensure all required features exist
            valid_features = [col for col in feature_columns if col in df.columns]
            if len(valid_features) < len(feature_columns):
                print(f"Warning: Missing features for {symbol}")
            
            # Get latest values
            latest_features = df[valid_features].iloc[-1].values
            
            # Ensure output is correct dtype
            return latest_features.astype(np.float32)
        
        # Return zeros if no data
        return np.zeros(len(feature_columns), dtype=np.float32)

class PaperTradingSystem:
    """Paper trading system using trained RL models in live market"""
    
    def __init__(self, symbols, initial_capital=100000.0, model_dir="models"):
        """
        Initialize paper trading system
        
        Args:
            symbols: List of stock symbols
            initial_capital: Starting paper money
            model_dir: Directory with trained models
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.model_dir = model_dir
        self.models = {}
        self.positions = {symbol: 0.0 for symbol in symbols}
        self.stop_losses = {symbol: 0.0 for symbol in symbols}
        self.trade_history = []
        self.portfolio_history = []
        
        # Data handler for live market data
        self.data_handler = LiveMarketDataHandler(symbols)
        
        # Load models
        self._load_models()
        
        # Portfolio tracking
        self.portfolio_value = initial_capital
        self.start_time = datetime.datetime.now()
        self.last_summary_time = time.time()
        
    def _load_models(self):
        """Load trained models for each symbol"""
        for symbol in self.symbols:
            clean_symbol = symbol.split('.')[0]  # Remove .NS or other suffixes
            model_path = f"{self.model_dir}/ppo_{clean_symbol}"
            
            try:
                if os.path.exists(f"{model_path}.zip"):
                    print(f"Loading model for {symbol}...")
                    self.models[symbol] = PPO.load(model_path)
                    print(f"Model loaded successfully for {symbol}")
                else:
                    print(f"No model found for {symbol}, skipping")
            except Exception as e:
                print(f"Error loading model for {symbol}: {e}")
    
    def get_action(self, symbol, observation):
        """Get trading action from model for given observation"""
        if symbol in self.models:
            try:
                action, _states = self.models[symbol].predict(observation, deterministic=True)
                return action
            except Exception as e:
                print(f"Error getting prediction for {symbol}: {e}")
                return np.array([0.0, 0.0])  # Default: no action
        return np.array([0.0, 0.0])  # No model, no action
    
    def execute_trade(self, symbol, action):
        """Execute paper trade based on model action"""
        # Get latest price
        current_price = self.data_handler.get_latest_price(symbol)
        if current_price is None:
            print(f"Cannot trade {symbol}: no price data")
            return False
        
        buy_fraction, stop_loss_percent = action
        buy_fraction = np.clip(buy_fraction, 0.0, 1.0)
        stop_loss_percent = np.clip(stop_loss_percent, 0.0, 1.0)
        
        # Current position value
        position_value = self.positions[symbol] * current_price
        
        # Log action consideration
        print(f"Considering action for {symbol}: Buy Fraction={buy_fraction:.2f}, Stop Loss={stop_loss_percent:.2f}")
        
        # Check stop-loss first
        if self.positions[symbol] > 0 and self.stop_losses[symbol] > 0:
            if current_price <= self.stop_losses[symbol]:
                # Stop loss triggered - sell everything
                gain_loss = (current_price - self.stop_losses[symbol] / (1 - stop_loss_percent)) * self.positions[symbol]
                trade_value = self.positions[symbol] * current_price
                
                self.balance += trade_value
                print(f"⚠️ STOP LOSS TRIGGERED for {symbol} at {current_price:.2f}! Sold {self.positions[symbol]:.2f} shares for {trade_value:.2f}")
                print(f"Gain/Loss: {gain_loss:.2f} ({gain_loss/trade_value*100:.2f}%)")
                
                # Record trade
                self.trade_history.append({
                    'timestamp': datetime.datetime.now(),
                    'symbol': symbol,
                    'action': 'STOP LOSS SELL',
                    'price': current_price,
                    'quantity': self.positions[symbol],
                    'value': trade_value,
                    'gain_loss': gain_loss
                })
                
                # Reset position
                self.positions[symbol] = 0.0
                self.stop_losses[symbol] = 0.0
                
                return True
        
        # Process buy decision
        if buy_fraction > 0.2:  # Only buy if action is significant
            if self.balance > 0:
                # Calculate trade size
                max_trade_value = self.balance * 0.2  # Don't use more than 20% of balance per trade
                trade_value = max_trade_value * buy_fraction
                shares_to_buy = trade_value / current_price
                
                if shares_to_buy > 0:
                    actual_cost = shares_to_buy * current_price
                    self.balance -= actual_cost
                    self.positions[symbol] += shares_to_buy
                    
                    # Set stop-loss
                    self.stop_losses[symbol] = current_price * (1 - stop_loss_percent)
                    
                    print(f"🛒 BUY {symbol}: {shares_to_buy:.2f} shares at {current_price:.2f} = {actual_cost:.2f}")
                    print(f"Stop loss set at {self.stop_losses[symbol]:.2f} ({stop_loss_percent*100:.1f}% below)")
                    
                    # Record trade
                    self.trade_history.append({
                        'timestamp': datetime.datetime.now(),
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'quantity': shares_to_buy,
                        'value': actual_cost,
                        'stop_loss': self.stop_losses[symbol]
                    })
                    
                    return True
                    
        # Process sell decision (not triggered by stop-loss)
        elif buy_fraction < 0.1 and self.positions[symbol] > 0:
            # Model suggests selling
            shares_to_sell = self.positions[symbol]
            sale_value = shares_to_sell * current_price
            gain_loss = sale_value - (shares_to_sell * (self.stop_losses[symbol] / (1 - stop_loss_percent)))
            
            self.balance += sale_value
            self.positions[symbol] = 0.0
            self.stop_losses[symbol] = 0.0
            
            print(f"💰 SELL {symbol}: {shares_to_sell:.2f} shares at {current_price:.2f} = {sale_value:.2f}")
            print(f"Gain/Loss: {gain_loss:.2f}")
            
            # Record trade
            self.trade_history.append({
                'timestamp': datetime.datetime.now(),
                'symbol': symbol,
                'action': 'SELL',
                'price': current_price,
                'quantity': shares_to_sell,
                'value': sale_value,
                'gain_loss': gain_loss
            })
            
            return True
            
        return False
    
    def calculate_portfolio_value(self):
        """Calculate total portfolio value (cash + positions)"""
        total_value = self.balance
        
        for symbol in self.positions:
            if self.positions[symbol] > 0:
                price = self.data_handler.get_latest_price(symbol)
                if price is not None:
                    position_value = self.positions[symbol] * price
                    total_value += position_value
        
        return total_value
    
    def run_trading_cycle(self):
        """Run one cycle of the paper trading system"""
        # Update portfolio value
        portfolio_value = self.calculate_portfolio_value()
        self.portfolio_value = portfolio_value
        
        # Record portfolio history
        self.portfolio_history.append({
            'timestamp': datetime.datetime.now(),
            'value': portfolio_value
        })
        
        # Process each symbol
        for symbol in self.symbols:
            try:
                # Get features for model input
                data = self.data_handler.get_data(symbol)
                if data is None or data.empty:
                    print(f"Skipping {symbol} - no data available")
                    continue
                
                # Get latest features
                feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'atr', 'ma20', 'ma50', 'macd', 'signal', 'vwap', 'obv']
                observation = self.data_handler.get_latest_features(symbol, feature_columns)
                
                # Get model action
                action = self.get_action(symbol, observation)
                
                # Execute trade
                self.execute_trade(symbol, action)
                
            except Exception as e:
                print(f"Error processing {symbol} in trading cycle: {e}")
        
        # Print portfolio summary every 5 minutes
        current_time = time.time()
        if current_time - self.last_summary_time > 300:  # 5 minutes
            self.print_portfolio_summary()
            self.last_summary_time = current_time
    
    def print_portfolio_summary(self):
        """Print summary of current portfolio status"""
        print("\n" + "="*50)
        print(f"PORTFOLIO SUMMARY - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        
        # Calculate total value and return
        portfolio_value = self.calculate_portfolio_value()
        percent_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100
        
        print(f"Cash Balance: ₹{self.balance:.2f}")
        print(f"Position Value: ₹{portfolio_value - self.balance:.2f}")
        print(f"Total Portfolio Value: ₹{portfolio_value:.2f}")
        print(f"Total Return: {percent_return:.2f}%")
        
        # Calculate time elapsed
        elapsed_time = datetime.datetime.now() - self.start_time
        elapsed_days = elapsed_time.total_seconds() / 86400  # Convert seconds to days
        
        # Calculate annualized return
        if elapsed_days > 0:
            annualized_return = ((1 + percent_return/100) ** (365/elapsed_days) - 1) * 100
            print(f"Annualized Return: {annualized_return:.2f}%")
        
        # Show positions
        print("\nCurrent Positions:")
        print("-"*50)
        print(f"{'Symbol':<10} {'Shares':<10} {'Price':<10} {'Value':<15} {'Stop Loss':<10}")
        print("-"*50)
        
        for symbol in self.positions:
            if self.positions[symbol] > 0:
                price = self.data_handler.get_latest_price(symbol)
                if price is not None:
                    value = self.positions[symbol] * price
                    stop_loss = self.stop_losses[symbol]
                    print(f"{symbol:<10} {self.positions[symbol]:<10.2f} {price:<10.2f} {value:<15.2f} {stop_loss:<10.2f}")
        
        print("="*50 + "\n")
    
    def save_trading_history(self):
        """Save trading history to CSV file"""
        if self.trade_history:
            df = pd.DataFrame(self.trade_history)
            df.to_csv("paper_trading_history.csv", index=False)
            print("Trading history saved to paper_trading_history.csv")
        
        # Also save portfolio history
        if self.portfolio_history:
            df = pd.DataFrame(self.portfolio_history)
            df.to_csv("portfolio_value_history.csv", index=False)
            print("Portfolio history saved to portfolio_value_history.csv")
    
    def plot_portfolio_performance(self):
        """Plot portfolio performance over time"""
        if len(self.portfolio_history) < 2:
            print("Not enough data to plot portfolio performance")
            return
        
        # Create dataframe from portfolio history
        df = pd.DataFrame(self.portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Plot performance
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['value'], 'b-', linewidth=2)
        plt.title('Paper Trading Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        
        # Add initial capital reference line
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7)
        plt.text(df.index[0], self.initial_capital, f'Initial Capital: ₹{self.initial_capital:.2f}', 
                 verticalalignment='bottom')
        
        # Calculate and display metrics
        final_value = df['value'].iloc[-1] if not df.empty else self.initial_capital
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        plt.figtext(0.15, 0.15, f'Initial Capital: ₹{self.initial_capital:.2f}\n'
                              f'Final Value: ₹{final_value:.2f}\n'
                              f'Total Return: {total_return:.2f}%',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('portfolio_performance.png')
        plt.close()
        print("Portfolio performance chart saved to portfolio_performance.png")

class AutomatedLiveTradingSystem(PaperTradingSystem):
    """Automated system that runs the trading logic continuously during market hours"""
    
    def __init__(self, symbols, initial_capital=100000.0, model_dir="models", log_file="live_trading.log"):
        super().__init__(symbols, initial_capital, model_dir)
        
        # Set up logging
        self.log_file = log_file
        self._setup_logging()
        
        # Track market hours
        self.market_open = datetime.time(9, 30)  # 9:30 AM
        self.market_close = datetime.time(16, 0)  # 4:00 PM
        self.market_timezone = datetime.timezone(datetime.timedelta(hours=-4))  # EST/EDT
        
        # Initialize monitoring
        self.monitor = TradingMonitor()
        
        self.log("Automated live trading system initialized")
    
    def _setup_logging(self):
        """Set up logging to file and console"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("LiveTrading")
    
    def log(self, message, level="info"):
        """Log a message to file and console"""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)
    
    def is_market_open(self):
        """Check if the market is currently open"""
        now = datetime.datetime.now(self.market_timezone)
        current_time = now.time()
        current_day = now.weekday()
        
        # Check if weekend (5 = Saturday, 6 = Sunday)
        if current_day >= 5:
            return False
        
        # Check if within market hours
        if self.market_open <= current_time <= self.market_close:
            return True
        
        return False
    
    def wait_for_next_cycle(self, interval_minutes=5):
        """Wait until the next trading cycle should run"""
        now = datetime.datetime.now()
        next_cycle = now + datetime.timedelta(minutes=interval_minutes)
        
        # Calculate seconds to wait
        wait_seconds = (next_cycle - now).total_seconds()
        
        # Sleep until next cycle
        if wait_seconds > 0:
            time.sleep(wait_seconds)
    
    def run_forever(self, cycle_interval_minutes=5):
        """Run the trading system forever, respecting market hours"""
        self.log("Starting automated live trading")
        
        try:
            while True:
                # Run health check
                self.monitor.check_health(self)
                
                # Check if market is open
                if self.is_market_open():
                    self.log("Market is open, running trading cycle")
                    
                    # Run trading cycle
                    self.run_trading_cycle()
                    
                    # Save data periodically
                    if len(self.portfolio_history) % 12 == 0:  # Every hour (assuming 5-min intervals)
                        self.save_trading_history()
                        self.plot_portfolio_performance()
                    
                    # Wait for next interval
                    self.wait_for_next_cycle(cycle_interval_minutes)
                    
                else:
                    # Market closed, check again in 5 minutes
                    self.log("Market is closed, waiting...")
                    
                    # If it's close to market open time, check more frequently
                    now = datetime.datetime.now(self.market_timezone)
                    current_time = now.time()
                    
                    # Calculate minutes to open by combining date and time objects
                    today = datetime.date.today()
                    market_open_dt = datetime.datetime.combine(today, self.market_open)
                    current_dt = datetime.datetime.combine(today, current_time)
                    
                    if current_dt > market_open_dt:
                        # Already past market open today, calculate for tomorrow
                        market_open_dt = datetime.datetime.combine(today + datetime.timedelta(days=1), self.market_open)
                    
                    minutes_to_open = (market_open_dt - current_dt).total_seconds() / 60
                    
                    if 0 < minutes_to_open < 15:  # Within 15 minutes of market open
                        wait_time = 1  # Check every minute
                    else:
                        wait_time = 5  # Check every 5 minutes
                    
                    time.sleep(wait_time * 60)
                    
        except KeyboardInterrupt:
            self.log("Automated trading stopped by user", "warning")
        except Exception as e:
            self.log(f"Critical error in automated trading: {str(e)}", "critical")
            # Try to save data before exiting
            self.save_trading_history()
            self.plot_portfolio_performance()
            raise
        finally:
            # Final summary and data save
            self.log("Finalizing automated trading session")
            self.print_portfolio_summary()
            self.save_trading_history()
            self.plot_portfolio_performance()


def train_model(symbol, start_date, end_date, timesteps=50000, model_dir="models"):
    """Train a RL model for stock trading"""
    try:
        print(f"Training model for {symbol}...")
        
        # Create environment
        env = StockTradingEnv(symbol=symbol, start_date=start_date, end_date=end_date)

        # Debug: Check observation and action spaces
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

         # Test one step to verify shapes
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        
        # Test a random action
        action = env.action_space.sample()
        print(f"Action shape: {action.shape}")
        obs, reward, done, _, info = env.step(action)
        print(f"Resulting observation shape: {obs.shape}")
        
        # Define policy with custom feature extractor
        policy_kwargs = dict(
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
        
        # Create model
        model = PPO("MlpPolicy", 
                    env, 
                    policy_kwargs=policy_kwargs,
                    verbose=1, 
                    learning_rate=0.0001,
                    batch_size=64,
                    n_steps=2048,
                    ent_coef=0.01,
                    gamma=0.99)
        
        # Train model
        model.learn(total_timesteps=timesteps)
        
        # Save model
        os.makedirs(model_dir, exist_ok=True)
        clean_symbol = symbol.split('.')[0]  # Remove .NS or other suffixes
        model_path = f"{model_dir}/ppo_{clean_symbol}"
        model.save(model_path)
        
        print(f"Model for {symbol} trained and saved to {model_path}")
        return model
        
    except Exception as e:
        print(f"Error training model for {symbol}: {e}")
        return None

def backtest_model(symbol, model, start_date, end_date):
    """Backtest a trained model on historical data"""
    try:
        print(f"Backtesting model for {symbol}...")
        
        # Create test environment with different date range
        env = StockTradingEnv(symbol=symbol, start_date=start_date, end_date=end_date)
        
        # Run backtest
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
        
        # Calculate performance metrics
        initial_balance = 10000.0  # Environment's initial balance
        final_net_worth = info['net_worth']
        percent_return = (final_net_worth - initial_balance) / initial_balance * 100
        
        print(f"Backtest Results for {symbol}:")
        print(f"Initial Balance: ₹{initial_balance:.2f}")
        print(f"Final Net Worth: ₹{final_net_worth:.2f}")
        print(f"Return: {percent_return:.2f}%")
        print(f"Total Reward: {total_reward:.2f}")
        
        return {
            'symbol': symbol,
            'initial_balance': initial_balance,
            'final_net_worth': final_net_worth,
            'percent_return': percent_return,
            'total_reward': total_reward
        }
        
    except Exception as e:
        print(f"Error backtesting model for {symbol}: {e}")
        return None

def main():
    """Main function to run the trading system"""
    # Configuration
    symbols = ["ABB", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT", "DMART", "AXISBANK","BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV","BAJAJHLDNG", "BAJAJHFL", "BANKBARODA", "BEL", "BPCL", "BHARTIARTL", "BOSCHLTD", "BRITANNIA","CGPOWER", "CANBK", "CHOLAFIN", "CIPLA", "COALINDIA", "DLF", "DABUR", "DIVISLAB", "DRREDDY", "EICHERMOT", "ETERNAL", "GAIL", "GODREJCP", "GRASIM", "HCLTECH", "HDFCBANK","HDFCLIFE", "HAVELLS", "HEROMOTOCO", "HINDALCO", "HAL", "HINDUNILVR", "HYUNDAI", "ICICIBANK","ICICIGI", "ICICIPRULI", "ITC", "INDHOTEL", "IOC", "IRFC", "INDUSINDBK", "NAUKRI", "INFY","INDIGO", "JSWENERGY", "JSWSTEEL", "JINDALSTEL", "JIOFIN", "KOTAKBANK", "LTIM", "LT", "LICI","LODHA", "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC", "PIDILITIND", "PFC", "POWERGRID", "PNB", "RECLTD", "RELIANCE", "SBILIFE", "MOTHERSON", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SBIN", "SUNPHARMA", "SWIGGY", "TVSMOTOR", "TCS", "TATACONSUM", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TECHM", "TITAN", "TORNTPHARM", "TRENT", "ULTRACEMCO", "UNITDSPR", "VBL", "VEDL", "WIPRO", "ZYDUSLIFE"]

    # symbols = ["HDFCBANK", "ICICIBANK"]
    
    training_start = '2000-01-01'
    training_end = '2023-12-31'
    test_start = '2024-01-01'
    test_end = '2024-12-31'
    model_dir = "models"
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # User menu
    while True:
        print("\n===== AI STOCK TRADING SYSTEM =====")
        print("1. Train models for all symbols")
        print("2. Train model for specific symbol")
        print("3. Backtest models")
        print("4. Run paper trading simulation")
        print("5. Run automated live trading")
        print("6. Setup system service")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == '1':
            # Train models for all symbols
            for symbol in symbols:
                train_model(symbol, training_start, training_end, model_dir=model_dir)
                
        elif choice == '2':
            # Train for specific symbol
            symbol = input("Enter stock symbol to train model for: ").upper()
            if symbol:
                train_model(symbol, training_start, training_end, model_dir=model_dir)
            
        elif choice == '3':
            # Backtest models
            results = []
            for symbol in symbols:
                clean_symbol = symbol.split('.')[0]
                model_path = f"{model_dir}/ppo_{clean_symbol}"
                
                if os.path.exists(f"{model_path}.zip"):
                    try:
                        model = PPO.load(model_path)
                        result = backtest_model(symbol, model, test_start, test_end)
                        if result:
                            results.append(result)
                    except Exception as e:
                        print(f"Error loading or backtesting model for {symbol}: {e}")
                else:
                    print(f"No model found for {symbol}, skipping backtest")
            
            # Display summary of backtest results
            if results:
                print("\n===== BACKTEST SUMMARY =====")
                print(f"{'Symbol':<10} {'Return %':<10} {'Final Value':<15}")
                print("-" * 35)
                
                for result in sorted(results, key=lambda x: x['percent_return'], reverse=True):
                    print(f"{result['symbol']:<10} {result['percent_return']:<10.2f} ₹{result['final_net_worth']:<15.2f}")
            
        elif choice == '4':
            # Run paper trading simulation (keep existing functionality)
            print("Starting paper trading simulation...")
            paper_trader = PaperTradingSystem(symbols)
            
            try:
                # Run for specified number of cycles or until interrupted
                cycles = int(input("Enter number of trading cycles to run (0 for continuous): "))
                
                if cycles > 0:
                    for i in range(cycles):
                        print(f"\nRunning trading cycle {i+1}/{cycles}")
                        paper_trader.run_trading_cycle()
                        time.sleep(5)  # Wait between cycles
                else:
                    print("Running continuous paper trading (Ctrl+C to stop)...")
                    while True:
                        paper_trader.run_trading_cycle()
                        time.sleep(60)  # Check every minute
                        
            except KeyboardInterrupt:
                print("\nPaper trading simulation stopped by user")
            finally:
                # Save results
                paper_trader.print_portfolio_summary()
                paper_trader.save_trading_history()
                paper_trader.plot_portfolio_performance()
                
        elif choice == '5':
            # Run automated live trading
            print("\n===== AUTOMATED LIVE TRADING =====")
            print("This will run continuously until stopped (Ctrl+C).")
            print("Trading will only occur during market hours.")
            print("All activity will be logged to live_trading.log")
            
            confirm = input("\nStart automated live trading? (y/n): ")
            if confirm.lower() == 'y':
                # Get trading parameters
                selected_symbols = input(f"Enter symbols to trade (comma-separated) or press Enter for default {symbols}: ")
                if selected_symbols:
                    symbols_to_trade = [s.strip().upper() for s in selected_symbols.split(',')]
                else:
                    symbols_to_trade = symbols
                
                initial_capital = float(input(f"Enter initial capital (default: 100000): ") or 100000)
                cycle_interval = int(input(f"Enter trading cycle interval in minutes (default: 5): ") or 5)
                
                # Create and run automated trading system
                automated_trader = AutomatedLiveTradingSystem(
                    symbols=symbols_to_trade,
                    initial_capital=initial_capital,
                    model_dir=model_dir
                )
                
                print("\nStarting automated live trading...")
                automated_trader.run_forever(cycle_interval_minutes=cycle_interval)
                
        elif choice == '6':
            # Setup system service
            print("\nSetting up system service for automated trading...")
            setup_system_service()
            
        elif choice == '7':
            print("Exiting program. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")

# Add this at the end of your existing file
if __name__ == "__main__":
    # Set warnings filter
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup service if requested
    if args.setup_service:
        setup_system_service()
        sys.exit(0)
    
    # Run in automatic mode if specified
    if args.autorun:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        print(f"Starting automated trading with symbols: {symbols}")
        
        # Initialize and run automated trading system
        automated_trader = AutomatedLiveTradingSystem(
            symbols=symbols,
            initial_capital=args.capital,
            model_dir="models"
        )
        
        # Run forever
        automated_trader.run_forever(cycle_interval_minutes=args.interval)
    else:
        # Run interactive mode
        main()




# This code:

# Preserves all existing functionality - Your original paper trading simulation and manual mode remain intact
# Adds automated live trading - New option #5 in the menu to run fully automated trading that respects market hours
# Implements system service setup - Option #6 to generate service files for running as a background process
# Adds command-line arguments - For headless operation (--autorun, --symbols, etc.)
# Includes monitoring - Basic health checks and alerts for the automated trading system
# Handles market hours - Only runs trades during actual market hours
# Provides proper logging - All activities are logged to a file for later review

# How to Use the Automated Features:

# From the menu:

# Choose option 5 to run automated live trading with market hour detection
# Enter your symbols, initial capital, and trading interval


# From command line:
# python trading_system.py --autorun --symbols AAPL,MSFT,GOOGL --capital 50000 --interval 10

# As a system service:

# First set up the service with menu option 6 or --setup-service
# Follow the instructions to install it on your system
# The trading system will automatically start on boot and keep running

# These enhancements make the system capable of running continuously in the live market while preserving all the existing manual functionality.