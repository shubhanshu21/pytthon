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
import argparse
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from typing import Dict, Any
import pytz
from scalping_stocks_finder import get_best_scalping_stocks
from stable_baselines3.common.callbacks import BaseCallback

# Load environment variables
load_dotenv()

class TrainingMonitor(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.actions = []
        self.rewards = []
        self.step_counter = 0
        self.env = None
    
    def _on_training_start(self):
        self.env = self.training_env.envs[0]
    
    def _on_step(self):
        # Store last action and reward every 1000 steps
        if self.step_counter % 1000 == 0:
            if hasattr(self.env, 'last_action') and self.env.last_action is not None:
                self.actions.append(self.env.last_action)
            
            if 'rewards' in self.locals and len(self.locals['rewards']) > 0:
                self.rewards.append(self.locals['rewards'][0])
            
            if len(self.actions) > 0 and len(self.rewards) > 0:
                print(f"Step {self.step_counter}: Action={self.actions[-1]}, Reward={self.rewards[-1]}")
        
        self.step_counter += 1
        return True
    
    def plot_training_progress(self):
        # Plot actions and rewards over time
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        
        if self.actions:
            plt.subplot(2, 1, 1)
            plt.plot([a[0] for a in self.actions], label='Buy Fraction')
            plt.plot([a[1] for a in self.actions], label='Stop Loss')
            plt.legend()
            plt.title('Actions Over Training')
        
        if self.rewards:
            plt.subplot(2, 1, 2)
            plt.plot(self.rewards)
            plt.title('Rewards Over Training')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

class TradingFees:
    """
    Class to calculate all trading fees for Indian markets including:
    - Brokerage
    - STT (Securities Transaction Tax)
    - Exchange Transaction Charges
    - GST
    - SEBI Turnover Fee
    - Stamp Duty
    """
    def __init__(self, broker_type="discount"):
        """
        Initialize with broker type
        
        Args:
            broker_type: "discount" or "full_service"
        """
        # Set fee structure based on broker type
        if broker_type == "discount":
            # Flat fee (Zerodha-like)
            self.brokerage_buy = 20  # Flat fee per executed order
            self.brokerage_sell = 20  # Flat fee per executed order
            self.brokerage_percent = 0.0  # No percentage brokerage
            self.brokerage_cap = 20  # Maximum brokerage per order
        else:
            # Traditional broker
            self.brokerage_buy = 0.0
            self.brokerage_sell = 0.0
            self.brokerage_percent = 0.03  # 0.03% brokerage
            self.brokerage_cap = 0  # No cap
        
        # Standard regulatory charges
        self.stt_buy = 0.00025  # 0.025% STT on buy (delivery)
        self.stt_sell = 0.00025  # 0.025% STT on sell (delivery)
        self.stt_intraday = 0.0001  # 0.01% STT on sell side for intraday
        self.exchange_txn_charge = 0.0000325  # 0.00325% exchange transaction charge
        self.gst = 0.18  # 18% GST on brokerage and exchange transaction charges
        self.sebi_charges = 0.0000001  # SEBI turnover fee 0.00001%
        self.stamp_duty = 0.00002  # 0.002% stamp duty (varies by state, taking average)
    
    def calculate_total_charges(self, trade_value, trade_type="INTRADAY"):
        """
        Calculate all charges for a trade
        
        Args:
            trade_value: Total value of the trade (price * quantity)
            trade_type: "INTRADAY" or "DELIVERY"
            
        Returns:
            Dictionary with all charges
        """
        # Initialize charges
        charges = {
            "brokerage": 0,
            "stt": 0,
            "exchange_charges": 0,
            "gst": 0,
            "sebi_charges": 0,
            "stamp_duty": 0,
            "total_charges": 0
        }
        
        # Calculate brokerage
        if self.brokerage_percent > 0:
            # Percentage brokerage
            brokerage = trade_value * self.brokerage_percent
            if self.brokerage_cap > 0:
                brokerage = min(brokerage, self.brokerage_cap)
            charges["brokerage"] = brokerage
        else:
            # Flat brokerage
            charges["brokerage"] = self.brokerage_buy  # Same as brokerage_sell for flat
        
        # Calculate STT based on trade type
        if trade_type == "INTRADAY":
            charges["stt"] = trade_value * self.stt_intraday
        else:  # DELIVERY
            charges["stt"] = trade_value * self.stt_sell  # Both buy and sell have same STT for delivery
        
        # Exchange transaction charges
        charges["exchange_charges"] = trade_value * self.exchange_txn_charge
        
        # GST on brokerage and exchange charges
        charges["gst"] = (charges["brokerage"] + charges["exchange_charges"]) * self.gst
        
        # SEBI charges
        charges["sebi_charges"] = trade_value * self.sebi_charges
        
        # Stamp duty
        charges["stamp_duty"] = trade_value * self.stamp_duty
        
        # Total charges
        charges["total_charges"] = sum(charges.values())
        
        return charges
    
    def calculate_breakeven(self, entry_price, quantity, trade_type="INTRADAY"):
        """
        Calculate breakeven price considering all fees
        
        Args:
            entry_price: Entry price per share
            quantity: Number of shares traded
            trade_type: "INTRADAY" or "DELIVERY"
            
        Returns:
            Breakeven price per share
        """
        entry_value = entry_price * quantity
        
        # Calculate charges for both entry and exit
        entry_charges = self.calculate_total_charges(entry_value, trade_type)
        exit_charges = self.calculate_total_charges(entry_value, trade_type)  # Using same value as estimate
        
        # Total cost
        total_cost = entry_value + entry_charges["total_charges"] + exit_charges["total_charges"]
        
        # Breakeven price
        breakeven_price = total_cost / quantity
        
        return breakeven_price
    
    def calculate_net_profit(self, entry_price, exit_price, quantity, trade_type="INTRADAY"):
        """
        Calculate net profit after all charges
        
        Args:
            entry_price: Entry price per share
            exit_price: Exit price per share
            quantity: Number of shares traded
            trade_type: "INTRADAY" or "DELIVERY"
            
        Returns:
            Dictionary with gross and net P&L details
        """
        entry_value = entry_price * quantity
        exit_value = exit_price * quantity
        
        # Gross P&L
        gross_pnl = exit_value - entry_value
        
        # Calculate charges
        entry_charges = self.calculate_total_charges(entry_value, trade_type)
        exit_charges = self.calculate_total_charges(exit_value, trade_type)
        
        # Total charges
        total_charges = entry_charges["total_charges"] + exit_charges["total_charges"]
        
        # Net P&L
        net_pnl = gross_pnl - total_charges
        
        # Return detailed breakdown
        return {
            "entry_value": entry_value,
            "exit_value": exit_value,
            "gross_pnl": gross_pnl,
            "gross_pnl_percent": (gross_pnl / entry_value) * 100 if entry_value > 0 else 0,
            "total_charges": total_charges,
            "charges_percent": (total_charges / entry_value) * 100 if entry_value > 0 else 0,
            "net_pnl": net_pnl,
            "net_pnl_percent": (net_pnl / entry_value) * 100 if entry_value > 0 else 0
        }


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
    
    def __init__(self, email_sender: str = None, email_password: str = None, email_recipient: str = None, 
                 phone: str = None, enable_sms: bool = False):
        self.email_sender = email_sender or os.getenv('EMAIL_SENDER')
        self.email_password = email_password or os.getenv('EMAIL_PASSWORD')
        self.email_recipient = email_recipient or os.getenv('EMAIL_RECIPIENT')
        self.phone = phone
        self.enable_sms = enable_sms
        self.last_health_check = time.time()
        self.error_count = 0
        
        # Initialize SMTP connection
        self.smtp_server = 'smtp.gmail.com'
        self.smtp_port = 465
        
    def check_health(self, trader):
        """Perform periodic health checks on the trading system"""
        current_time = time.time()
        
        # Run health check every 30 minutes
        if current_time - self.last_health_check > 1800:
            # Check portfolio value for unusual drops
            portfolio_value = trader.calculate_portfolio_value()
            if len(trader.portfolio_history) > 2:
                last_value = trader.portfolio_history[-1]['value']
                prev_value = trader.portfolio_history[-2]['value']
                if last_value < prev_value * 0.95:  # 5% drop
                    self.send_notification(
                        subject="Portfolio Value Alert",
                        message=f"Portfolio value dropped by {(prev_value - last_value) / prev_value * 100:.2f}%"
                    )
            
            self.last_health_check = current_time
    
    def send_notification(self, subject: str, message: str):
        """Send email notification"""
        if not self.email_sender or not self.email_password or not self.email_recipient:
            print("Email configuration not set up. Skipping notification.")
            return
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = self.email_recipient
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(message, 'html'))
            
            # Connect to SMTP server
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.email_sender, self.email_password)
                server.sendmail(self.email_sender, self.email_recipient.split(','), msg.as_string())
                
            print("Email notification sent successfully")
            
        except Exception as e:
            print(f"Failed to send email notification: {e}")

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
    parser.add_argument('--symbols', type=str, default='ABB, ADANIENSOL, ADANIENT', help='Comma-separated list of stock symbols')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital amount')
    parser.add_argument('--interval', type=int, default=5, help='Trading cycle interval in minutes')
    parser.add_argument('--setup-service', action='store_true', help='Generate system service files')
    
    return parser.parse_args()

class StockTradingEnv(gym.Env):
    def __init__(self, symbol, start_date, end_date, timeframe='1m',min_holding_period=5,position_held_steps = 0):
        super(StockTradingEnv, self).__init__()
        
        # self.symbol = symbol+'.NS'
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.metadata = {'render.modes': ['human'], 'render_fps': 60}
        self.render_mode = None
        self.min_holding_period = 5  # Minimum steps to hold a position
        self.position_held_steps = 0  # Counter for how long position has been held

        # Download stock data using yfinance
        print(f"Downloading data for {symbol} from {start_date} to {end_date}...")
        try:
            # self.df = yf.download(self.symbol, start=self.start_date, end=self.end_date, interval=self.timeframe)
            # print(f"Downloaded {len(self.df)} rows of data")
            
            filename = f"datasets/{symbol}_minute.csv"

            if not os.path.exists(filename):
                raise FileNotFoundError(f"{filename} not found. Please generate it or download using yfinance.")

            # Read CSV into DataFrame
            self.df = pd.read_csv(filename)

            # ADD THIS VALIDATION CODE HERE
            print(f"Data statistics for {symbol}:")
            print(f"Close price range: {self.df['close'].min()} to {self.df['close'].max()}")
            print(f"Sample values: {self.df['close'].head().tolist()}")

            # Ensure data is not improperly scaled
            if self.df['close'].max() < 1.0:
                print("WARNING: Close prices seem to be improperly scaled or normalized!")
                # If you confirm this is a scaling issue, you can uncomment this fix:
                # self.df['close'] = self.df['close'] * 100  # Example scaling adjustment

            self.df.drop(columns=['date'], inplace=True)

            print(f"Head data for {symbol}")

            # Show first 5 rows
            print(self.df.head())

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
            self.df.bfill(inplace=True)
            
            # Clip extreme values to prevent issues with scaling
            for col in self.df.columns:
                if col not in ['volume']:  # Skip volume as it can have legitimately high values
                    q_low = self.df[col].quantile(0.01)
                    q_high = self.df[col].quantile(0.99)
                    self.df[col] = self.df[col].clip(q_low, q_high)
            
            # Define features and ensure they exist
            self.feature_columns = ['close', 'rsi', 'atr', 'ema8', 'ema21', 'macd', 'signal', 'ma20', 'ma50',
                        'bb_upper', 'bb_lower', 'vwap', 'obv', 'obv_ema', 'momentum', 'cci',
                        'price_change', 'volume_change', 'volatility', 'is_bullish_engulfing']
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
                'rsi': [50.0] * 100,        # Default RSI as 50 (neutral)
                'atr': [1.0] * 100,         # Default ATR as 1.0
                'ma20': [99.0] * 100,       # Simple moving average (close price)
                'ma50': [99.0] * 100,       # Simple moving average (close price)
                'macd': [0.0] * 100,        # Default MACD value (no momentum)
                'signal': [0.0] * 100,      # Default MACD signal value
                'vwap': [99.0] * 100,       # Use the close price as default VWAP
                'obv': [1000.0] * 100,      # Default OBV (should be calculated properly)
                'obv_ema': [1000.0] * 100,  # Default EMA on OBV
                'momentum': [0.0] * 100,    # Momentum is zero if not calculated
                'cci': [0.0] * 100,         # CCI default (neutral)
                'price_change': [0.0] * 100, # Price change is zero
                'volume_change': [0.0] * 100, # Volume change is zero
                'volatility': [0.0] * 100,  # Volatility is zero
                'is_bullish_engulfing': [0] * 100  # No candlestick pattern by default
            })
            
        self.feature_columns = ['close', 'rsi', 'atr', 'ema8', 'ema21', 'macd', 'signal', 'ma20', 'ma50',
                        'bb_upper', 'bb_lower', 'vwap', 'obv', 'obv_ema', 'momentum', 'cci',
                        'price_change', 'volume_change', 'volatility', 'is_bullish_engulfing']
        
        # Action space: Continuous actions [buy_fraction, stop_loss_percent]
        self.action_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # Observation space with strictly float32 type
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.feature_columns),), dtype=np.float32)
        
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


    def render(self):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Balance: {self.balance:.2f}, Position: {self.position:.2f}")
    

    def _add_technical_indicators(self):
        try:
            # Basic Indicators (Shorter periods for scalping)
            self.df['rsi'] = TA.RSI(self.df, period=7)
            self.df['atr'] = TA.ATR(self.df, period=7)
            self.df['ema8'] = TA.EMA(self.df, period=8)
            self.df['ema21'] = TA.EMA(self.df, period=21)
            
            # MACD
            macd_data = TA.MACD(self.df)
            self.df['macd'] = macd_data['MACD']
            self.df['signal'] = macd_data['SIGNAL']

            # SMA (still useful for longer context)
            self.df['ma20'] = TA.SMA(self.df, period=20)
            self.df['ma50'] = TA.SMA(self.df, period=50)
            
            # Bollinger Bands
            bbands = TA.BBANDS(self.df)
            self.df['bb_upper'] = bbands['BB_UPPER']
            self.df['bb_lower'] = bbands['BB_LOWER']
            
            # VWAP
            typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
            self.df['vwap'] = (self.df['volume'] * typical_price).cumsum() / self.df['volume'].cumsum()
            
            # OBV + EMA on OBV
            self.df['obv'] = self.df.apply(lambda row: row['volume'] if row['close'] > row['open'] else -row['volume'], axis=1).cumsum()
            self.df['obv_ema'] = TA.EMA(self.df, column='obv', period=20)

            # Price and volume change
            self.df['price_change'] = self.df['close'].pct_change() * 100
            self.df['volume_change'] = self.df['volume'].pct_change() * 100
            self.df['volatility'] = self.df['high'] - self.df['low']

            # Momentum and CCI
            self.df['momentum'] = self.df['close'] - self.df['close'].shift(5)
            self.df['cci'] = TA.CCI(self.df, period=14)

            # Simple bullish engulfing pattern (candlestick pattern)
            self.df['is_bullish_engulfing'] = (
                (self.df['open'].shift(1) > self.df['close'].shift(1)) &
                (self.df['open'] < self.df['close']) &
                (self.df['open'] < self.df['close'].shift(1)) &
                (self.df['close'] > self.df['open'].shift(1))
            ).astype(int)

            self.df['is_breakout'] = ((self.df['close'] > self.df['bb_upper']) & (self.df['volume_change'] > 30)).astype(int)

            # Handle infinite or NaN values
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinity with NaN
            self.df.bfill(inplace=True)

            # Normalize selected features
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            to_scale = ['rsi', 'atr', 'macd', 'signal', 'obv', 'obv_ema', 
                        'momentum', 'cci', 'price_change', 'volume_change', 'volatility']
            self.df[to_scale] = scaler.fit_transform(self.df[to_scale])

            # Final cleanup
            self.df.bfill(inplace=True)
            self.df.dropna(inplace=True)

        except Exception as e:
            print(f"Error calculating indicators for {self.symbol}: {e}")
            # Fallback to dummy values
            self.df['rsi'] = 50.0
            self.df['atr'] = 1.0
            self.df['ema8'] = self.df['close']
            self.df['ema21'] = self.df['close']
            self.df['macd'] = 0.0
            self.df['signal'] = 0.0
            self.df['ma20'] = self.df['close']
            self.df['ma50'] = self.df['close']
            self.df['bb_upper'] = self.df['close']
            self.df['bb_lower'] = self.df['close']
            self.df['vwap'] = self.df['close']
            self.df['obv'] = self.df['volume']
            self.df['obv_ema'] = self.df['volume']
            self.df['price_change'] = 0.0
            self.df['volume_change'] = 0.0
            self.df['volatility'] = 0.0
            self.df['momentum'] = 0.0
            self.df['cci'] = 0.0
            self.df['is_bullish_engulfing'] = 0



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

    
    # Update this portion of your step() method in StockTradingEnv class
    def step(self, action):
        # Store the action for debugging/monitoring purposes
        self.last_action = action.copy() if hasattr(action, 'copy') else action

        # Check if done or at end of data
        if self.done or self.current_step >= len(self.df) - 1:
            self.done = True
            return self._get_obs(), 0.0, self.done, False, {
                'net_worth': float(self.net_worth),
                'balance': float(self.balance),
                'position': float(self.position),
                'step': int(self.current_step),
                'stop_loss': float(self.stop_loss)
            }

        # Safely get current price
        try:
            current_price = float(self.df.iloc[self.current_step]['close'])
        except (IndexError, KeyError) as e:
            print(f"Error getting price for {self.symbol}: {e}")
            current_price = 0.0

        prev_net_worth = self.net_worth
        prev_position = self.position
        prev_balance = self.balance

        # Execute action with clearer buy/sell logic
        try:
            buy_fraction, stop_loss_percent = action
            buy_fraction = np.clip(buy_fraction, 0.0, 1.0)
            stop_loss_percent = np.clip(stop_loss_percent, 0.0, 1.0)

            # IMPORTANT CHANGE: More explicit buy/sell conditions
            if buy_fraction > 0.1:  # Only buy if fraction is significant (>10%)
                if self.balance > 0:
                    max_shares = self.balance / current_price if current_price > 0 else 0
                    shares_bought = buy_fraction * max_shares
                    cost = shares_bought * current_price
                    
                    if cost > 0:
                        print(f"BUYING: {shares_bought:.2f} shares at {current_price:.2f}, cost: {cost:.2f}")
                        self.position += shares_bought
                        self.balance -= cost
                        self.stop_loss = current_price * (1 - max(0.05, stop_loss_percent))  # Minimum 5% stop-loss
                        print(f"New position: {self.position:.2f}, Balance: {self.balance:.2f}, Stop loss: {self.stop_loss:.4f}")
            
            # Check for sell conditions (either model says sell OR stop loss hit)
            if self.position > 0:
                self.position_held_steps += 1

                should_sell = (buy_fraction < 0.1 or current_price <= self.stop_loss) and self.position_held_steps >= self.min_holding_period

                
                if should_sell:
                    proceeds = self.position * current_price
                    print(f"SELLING: {self.position:.2f} shares at {current_price:.2f}, proceeds: {proceeds:.2f}")
                    print(f"Sell reason: {'Stop-loss triggered' if current_price <= self.stop_loss else 'Model decision'}")
                    self.balance += proceeds
                    self.position = 0.0
                    # self.stop_loss = 0.0
                    self.stop_loss = current_price * (1 - max(0.05, stop_loss_percent))  # Minimum 5% stop-loss
                    print(f"New position: {self.position:.2f}, Balance: {self.balance:.2f}")
                    self.position_held_steps = 0  # Reset counter

                else:
                    # If trying to sell but haven't held long enough
                    if buy_fraction < 0.1 or current_price <= self.stop_loss:
                        print(f"Wanted to sell but holding period ({self.position_held_steps}/{self.min_holding_period}) not met")    

            else:
                # Reset counter when no position
                self.position_held_steps = 0

        except Exception as e:
            print(f"Error executing action for {self.symbol}: {e}")

        # Move to next step
        self.current_step += 1

        # Recalculate net worth
        try:
            if self.current_step < len(self.df):
                current_price = float(self.df.iloc[self.current_step]['close'])

            self.net_worth = self.balance + (self.position * current_price)
            self.max_net_worth = max(self.max_net_worth, self.net_worth)
        except Exception as e:
            print(f"Error calculating net worth for {self.symbol}: {e}")


        # Reward calculation 
        try:
            # Calculate raw return
            pct_change = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
            
            # Base reward on actual return, not just Sharpe ratio
            reward = pct_change * 100  # Scale up percentage change
            
            # Add trading signals to encourage action
            # 1. Reward for buying when oversold (RSI < 30)
            rsi = self.df.iloc[self.current_step]['rsi'] 
            price_change = self.df.iloc[self.current_step]['price_change']
            
            # Encourage buying at appropriate times
            if buy_fraction > 0.1 and prev_position == 0:  # When entering position
                # Extra reward for buying when oversold
                if rsi < 0.3:  # RSI < 30%
                    reward += 0.3  # Increased from 0.2
                # Reduced penalty for buying when overbought
                elif rsi > 0.7:  # RSI > 70%
                    reward -= 0.1  # Reduced from 0.2
            
            # Encourage selling at appropriate times
            if buy_fraction < 0.1 and self.position > 0:  # When exiting position
                # Extra reward for selling when overbought
                if rsi > 0.7:  # RSI > 70%
                    reward += 0.3  # Increased from 0.2
                # Reduced penalty for selling when oversold
                elif rsi < 0.3:  # RSI < 30%
                    reward -= 0.1  # Reduced from 0.2
            
            # Add incentive for taking positions (combat do-nothing strategy)
            if self.position > 0:
                reward += 0.05  # Small positive reward for holding positions
            
            # Trading cost penalty (smaller than before)
            trading_cost = 0.0001  # Reduced from 0.0002
            if (buy_fraction > 0.1 and prev_position == 0) or (buy_fraction < 0.1 and self.position == 0 and prev_position > 0):
                reward -= trading_cost
            
            # Stronger penalty for staying in cash too long
            if prev_position == 0 and self.position == 0:
                reward -= 0.001  # Increased from 0.0001
            
            # Clip reward to reasonable range
            reward = np.clip(reward, -2.0, 2.0)
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            reward = 0.0

        # Get observation and info
        obs = self._get_obs()
        info = {
            'net_worth': float(self.net_worth),
            'balance': float(self.balance),
            'position': float(self.position),
            'step': int(self.current_step),
            'stop_loss': float(self.stop_loss)
        }

        # Check end condition
        if self.current_step >= len(self.df) - 1:
            self.done = True

        return obs, reward, self.done, False, info

class LiveMarketDataHandler:
    """Handles fetching and processing real-time market data for paper trading"""
    
    def __init__(self, symbols, lookback_days=6, update_interval=60):
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
        Get current and historical 1-min data for a symbol with technical indicators.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with enriched stock data
        """
        import yfinance as yf
        import datetime
        import time
        from sklearn.preprocessing import MinMaxScaler
        
        current_time = time.time()

        # Check if we need to update the data
        if symbol not in self.last_update or (current_time - self.last_update[symbol]) > self.update_interval:
            try:
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=self.lookback_days)

                print(f"Fetching 1-min data for {symbol} from {start_date} to {end_date}...")
                df = yf.download(symbol, start=start_date, end=end_date, interval="1m")

                if df.empty or len(df) < 30:
                    print(f"Warning: Not enough data for {symbol}")
                    return None

                df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]

                # Technical Indicators
                df['rsi'] = TA.RSI(df, period=7)
                df['atr'] = TA.ATR(df, period=7)
                df['ema8'] = TA.EMA(df, period=8)
                df['ema21'] = TA.EMA(df, period=21)
                df['ma20'] = TA.SMA(df, period=20)
                df['ma50'] = TA.SMA(df, period=50)

                macd = TA.MACD(df)
                df['macd'] = macd['MACD']
                df['signal'] = macd['SIGNAL']

                bbands = TA.BBANDS(df)
                df['bb_upper'] = bbands['BB_UPPER']
                df['bb_lower'] = bbands['BB_LOWER']

                df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
                df['obv'] = df.apply(lambda row: row['volume'] if row['close'] > row['open'] else -row['volume'], axis=1).cumsum()
                df['obv_ema'] = TA.EMA(df, column='obv', period=20)

                # Additional Features
                df['momentum'] = df['close'] - df['close'].shift(5)
                df['cci'] = TA.CCI(df, period=14)
                df['price_change'] = df['close'].pct_change() * 100
                df['volume_change'] = df['volume'].pct_change() * 100
                df['volatility'] = df['high'] - df['low']

                # Candlestick Pattern: Bullish Engulfing
                df['is_bullish_engulfing'] = (
                    (df['open'].shift(1) > df['close'].shift(1)) &
                    (df['open'] < df['close']) &
                    (df['open'] < df['close'].shift(1)) &
                    (df['close'] > df['open'].shift(1))
                ).astype(int)

                # Normalize selected columns
                scaler = MinMaxScaler()
                norm_cols = ['rsi', 'atr', 'macd', 'signal', 'obv', 'obv_ema', 
                            'momentum', 'cci', 'price_change', 'volume_change', 'volatility']
                df[norm_cols] = scaler.fit_transform(df[norm_cols])

                self.df.bfill(inplace=True)
                df.dropna(inplace=True)

                # Cache data
                self.data_cache[symbol] = df
                self.last_update[symbol] = current_time

                print(f"Data updated for {symbol}, last price: {df['close'].iloc[-1]:.2f}")

            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                if symbol in self.data_cache:
                    print(f"Using cached data for {symbol}")
                    return self.data_cache[symbol]
                else:
                    print(f"No cached data available for {symbol}")
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

         # Set up trading monitor with email configuration
        self.monitor = TradingMonitor()
        
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

            # self.monitor.send_notification(
            #         subject=f"Model loading for Symbol - {symbol}",
            #         message=f"""
            #         <html>
            #           <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px; color: #333;">
            #             <div style="background-color: #ffffff; border-radius: 8px; padding: 20px; max-width: 600px; margin: auto; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);">
            #               <div style="font-size: 20px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;">
            #                 🔄 Model Loading Notification
            #               </div>
            #               <div style="font-size: 16px; line-height: 1.6;">
            #                 <p>Hello,</p>
            #                 <p>The model is currently being loaded for the symbol <strong>{symbol}</strong>.</p>
            #                 <p>Please wait while we initialize everything.</p>
            #               </div>
            #               <div style="margin-top: 20px; font-size: 13px; color: #888; text-align: center;">
            #                 This is an automated message from your Trading Bot.
            #               </div>
            #             </div>
            #           </body>
            #         </html>
            #         """
            #     )

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
        """Execute paper trade based on model action with realistic fees"""
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
        
        # Initialize fee calculator
        fee_calculator = TradingFees(broker_type="discount")  # Use discount broker like Zerodha
        
        # Log action consideration
        print(f"Considering action for {symbol}: Buy Fraction={buy_fraction:.2f}, Stop Loss={stop_loss_percent:.2f}")
        
        # Check stop-loss first
        if self.positions[symbol] > 0 and self.stop_losses[symbol] > 0:
            if current_price <= self.stop_losses[symbol]:
                # Stop loss triggered - sell everything
                trade_value = self.positions[symbol] * current_price
                
                # Calculate fees for selling
                entry_price = self.stop_losses[symbol] / (1 - stop_loss_percent)  # Calculate back the entry price
                sell_fees = fee_calculator.calculate_net_profit(
                    entry_price=entry_price,
                    exit_price=current_price,
                    quantity=self.positions[symbol],
                    trade_type="INTRADAY"
                )
                
                # Adjust balance with fees deducted
                self.balance += trade_value - sell_fees["total_charges"]
                
                # Record actual gain/loss including fees
                gain_loss = sell_fees["net_pnl"]
                
                self.monitor.send_notification(
                    subject=f"Stop Loss Triggered - {symbol}",
                    message=f"""
                    <html>
                    <body>
                    <h3>Stop Loss Triggered for {symbol}</h3>
                    <table border="1" cellpadding="5">
                        <tr><td>Sell Price</td><td>₹{current_price:.2f}</td></tr>
                        <tr><td>Quantity</td><td>{self.positions[symbol]:.2f} shares</td></tr>
                        <tr><td>Trade Value</td><td>₹{trade_value:.2f}</td></tr>
                        <tr><td>Trading Fees</td><td>₹{sell_fees['total_charges']:.2f}</td></tr>
                        <tr><td>Net P&L</td><td>₹{gain_loss:.2f} ({sell_fees['net_pnl_percent']:.2f}%)</td></tr>
                    </table>
                    </body>
                    </html>
                    """
                )
                
                print(f"⚠️ STOP LOSS TRIGGERED for {symbol} at {current_price:.2f}! Sold {self.positions[symbol]:.2f} shares for {trade_value:.2f}")
                print(f"Trading Fees: ₹{sell_fees['total_charges']:.2f}")
                print(f"Net Gain/Loss: ₹{gain_loss:.2f} ({sell_fees['net_pnl_percent']:.2f}%)")
                
                # Record trade with fees
                self.trade_history.append({
                    'timestamp': datetime.datetime.now(),
                    'symbol': symbol,
                    'action': 'STOP LOSS SELL',
                    'price': current_price,
                    'quantity': self.positions[symbol],
                    'value': trade_value,
                    'fees': sell_fees['total_charges'],
                    'net_pnl': gain_loss
                })
                
                # Reset position
                self.positions[symbol] = 0.0
                self.stop_losses[symbol] = 0.0
                
                return True
            
        # Process buy decision
        if buy_fraction > 0.2:  
            if self.balance > 0:
                # Calculate trade size
                max_trade_value = self.balance * 0.2  
                trade_value = max_trade_value * buy_fraction
                
                # Calculate fees and adjust shares to buy
                estimated_fees = fee_calculator.calculate_total_charges(trade_value, "INTRADAY")
                adjusted_trade_value = trade_value - estimated_fees["total_charges"]
                shares_to_buy = adjusted_trade_value / current_price
                
                if shares_to_buy > 0:
                    actual_cost = shares_to_buy * current_price
                    total_cost = actual_cost + estimated_fees["total_charges"]
                    self.balance -= total_cost
                    self.positions[symbol] += shares_to_buy
                    
                    # Calculate breakeven price
                    breakeven = fee_calculator.calculate_breakeven(current_price, shares_to_buy, "INTRADAY")
                    
                    self.monitor.send_notification(
                        subject=f"New Position - {symbol}",
                        message=f"""
                        <html>
                        <body>
                        <h3>New Position Opened for {symbol}</h3>
                        <table border="1" cellpadding="5">
                            <tr><td>Entry Price</td><td>₹{current_price:.2f}</td></tr>
                            <tr><td>Quantity</td><td>{shares_to_buy:.2f} shares</td></tr>
                            <tr><td>Trade Value</td><td>₹{actual_cost:.2f}</td></tr>
                            <tr><td>Trading Fees</td><td>₹{estimated_fees['total_charges']:.2f}</td></tr>
                            <tr><td>Total Cost</td><td>₹{total_cost:.2f}</td></tr>
                            <tr><td>Breakeven Price</td><td>₹{breakeven:.2f}</td></tr>
                            <tr><td>Stop Loss Price</td><td>₹{current_price * (1 - stop_loss_percent):.2f}</td></tr>
                        </table>
                        </body>
                        </html>
                        """
                    )
                    
                    # Set stop-loss
                    self.stop_loss = current_price * (1 - max(0.05, stop_loss_percent))  # Minimum 5% stop-loss
                    
                    print(f"🛒 BUY {symbol}: {shares_to_buy:.2f} shares at {current_price:.2f} = ₹{actual_cost:.2f}")
                    print(f"Trading Fees: ₹{estimated_fees['total_charges']:.2f}")
                    print(f"Breakeven Price: ₹{breakeven:.2f}")
                    print(f"Stop loss set at ₹{self.stop_losses[symbol]:.2f} ({stop_loss_percent*100:.1f}% below)")
                    
                    # Record trade with fees
                    self.trade_history.append({
                        'timestamp': datetime.datetime.now(),
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'quantity': shares_to_buy,
                        'value': actual_cost,
                        'fees': estimated_fees['total_charges'],
                        'total_cost': total_cost,
                        'breakeven': breakeven,
                        'stop_loss': self.stop_losses[symbol]
                    })
                    
                    return True
                    
        # Process sell decision 
        elif buy_fraction < 0.1 and self.positions[symbol] > 0:
            # Model suggests selling
            shares_to_sell = self.positions[symbol]
            sale_value = shares_to_sell * current_price
            
            # Calculate original entry price (approximate)
            entry_price = self.stop_losses[symbol] / (1 - stop_loss_percent)
            
            # Calculate fees and net profit
            sell_result = fee_calculator.calculate_net_profit(
                entry_price=entry_price,
                exit_price=current_price,
                quantity=shares_to_sell,
                trade_type="INTRADAY"
            )
            
            # Adjust balance with fees deducted
            self.balance += sale_value - sell_result["total_charges"]
            net_pnl = sell_result["net_pnl"]
            
            self.monitor.send_notification(
                subject=f"Position Closed - {symbol}",
                message=f"""
                <html>
                <body>
                <h3>Position Closed for {symbol}</h3>
                <table border="1" cellpadding="5">
                    <tr><td>Entry Price</td><td>₹{entry_price:.2f}</td></tr>
                    <tr><td>Exit Price</td><td>₹{current_price:.2f}</td></tr>
                    <tr><td>Quantity</td><td>{shares_to_sell:.2f} shares</td></tr>
                    <tr><td>Trade Value</td><td>₹{sale_value:.2f}</td></tr>
                    <tr><td>Trading Fees</td><td>₹{sell_result['total_charges']:.2f}</td></tr>
                    <tr><td>Gross P&L</td><td>₹{sell_result['gross_pnl']:.2f} ({sell_result['gross_pnl_percent']:.2f}%)</td></tr>
                    <tr><td>Net P&L</td><td>₹{net_pnl:.2f} ({sell_result['net_pnl_percent']:.2f}%)</td></tr>
                </table>
                </body>
                </html>
                """
            )
            
            print(f"💰 SELL {symbol}: {shares_to_sell:.2f} shares at {current_price:.2f} = ₹{sale_value:.2f}")
            print(f"Trading Fees: ₹{sell_result['total_charges']:.2f}")
            print(f"Gross P&L: ₹{sell_result['gross_pnl']:.2f} ({sell_result['gross_pnl_percent']:.2f}%)")
            print(f"Net P&L after fees: ₹{net_pnl:.2f} ({sell_result['net_pnl_percent']:.2f}%)")
            
            # Record trade with fees
            self.trade_history.append({
                'timestamp': datetime.datetime.now(),
                'symbol': symbol,
                'action': 'SELL',
                'price': current_price,
                'quantity': shares_to_sell,
                'value': sale_value,
                'fees': sell_result['total_charges'],
                'gross_pnl': sell_result['gross_pnl'],
                'net_pnl': net_pnl
            })
            
            # Reset position
            self.positions[symbol] = 0.0
            self.stop_losses[symbol] = 0.0
            
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
                feature_columns = ['close', 'rsi', 'atr', 'ema8', 'ema21', 'macd', 'signal', 'ma20', 'ma50',
                        'bb_upper', 'bb_lower', 'vwap', 'obv', 'obv_ema', 'momentum', 'cci',
                        'price_change', 'volume_change', 'volatility', 'is_bullish_engulfing']
                
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
        """Print summary of current portfolio status with tax and fees impact"""
        print("\n" + "="*60)
        print(f"PORTFOLIO SUMMARY - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Calculate total value and return
        portfolio_value = self.calculate_portfolio_value()
        percent_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100
        
        print(f"Cash Balance: ₹{self.balance:.2f}")
        print(f"Position Value: ₹{portfolio_value - self.balance:.2f}")
        print(f"Total Portfolio Value: ₹{portfolio_value:.2f}")
        
        # Calculate fee impact
        if hasattr(self, 'trade_history') and self.trade_history:
            total_fees = sum(trade.get('fees', 0) for trade in self.trade_history)
            total_gross_pnl = sum(trade.get('gross_pnl', 0) for trade in self.trade_history if 'gross_pnl' in trade)
            total_net_pnl = sum(trade.get('net_pnl', 0) for trade in self.trade_history if 'net_pnl' in trade)
            
            fee_impact = 0
            if total_gross_pnl != 0:
                fee_impact = (total_fees / abs(total_gross_pnl)) * 100
            
            print(f"\nTrading Performance:")
            print(f"Total Gross P&L: ₹{total_gross_pnl:.2f}")
            print(f"Total Trading Fees: ₹{total_fees:.2f} ({fee_impact:.2f}% of gross P&L)")
            print(f"Total Net P&L: ₹{total_net_pnl:.2f}")
        
        print(f"Total Return: {percent_return:.2f}%")
        
        # Calculate tax implications (for annual reporting)
        # In India, intraday trading is considered non-delivery based and taxed as business income
        if hasattr(self, 'trade_history') and self.trade_history:
            # Calculate short-term capital gains for completed trades (simplified)
            # For actual tax calculation, would need to track financial year and tax brackets
            net_profit = total_net_pnl if 'total_net_pnl' in locals() else sum(trade.get('net_pnl', 0) for trade in self.trade_history if 'net_pnl' in trade)
            
            # Simplified tax calculation (assuming business income)
            # In reality, this depends on income tax bracket, but using 30% as example
            estimated_tax = 0
            if net_profit > 0:
                estimated_tax = net_profit * 0.30  # 30% income tax rate (simplified)
                
            # Show after-tax return
            after_tax_profit = net_profit - estimated_tax if net_profit > 0 else net_profit
            
            print(f"\nTax Implications (Simplified):")
            print(f"Net Trading Profit (Pre-tax): ₹{net_profit:.2f}")
            print(f"Estimated Income Tax (30%): ₹{estimated_tax:.2f}")
            print(f"Net Profit After Tax: ₹{after_tax_profit:.2f}")
        
        # Calculate time elapsed
        elapsed_time = datetime.datetime.now() - self.start_time
        elapsed_days = elapsed_time.total_seconds() / 86400  # Convert seconds to days
        
        # Calculate annualized return
        if elapsed_days > 0:
            annualized_return = ((1 + percent_return/100) ** (365/elapsed_days) - 1) * 100
            print(f"\nAnnualized Return (Pre-tax): {annualized_return:.2f}%")
            
            # After-tax annualized return (simplified)
            if 'after_tax_profit' in locals():
                after_tax_return = (after_tax_profit / self.initial_capital) * 100
                after_tax_annualized = ((1 + after_tax_return/100) ** (365/elapsed_days) - 1) * 100
                print(f"Annualized Return (After-tax): {after_tax_annualized:.2f}%")
        
        # Show positions with current status
        print("\nCurrent Positions:")
        print("-"*60)
        print(f"{'Symbol':<10} {'Shares':<10} {'Price':<10} {'Value':<15} {'Stop Loss':<10} {'Breakeven':<10}")
        print("-"*60)
        
        for symbol in self.positions:
            if self.positions[symbol] > 0:
                price = self.data_handler.get_latest_price(symbol)
                if price is not None:
                    value = self.positions[symbol] * price
                    stop_loss = self.stop_losses[symbol]
                    
                    # Find the breakeven price from trade history
                    breakeven = 0
                    for trade in reversed(self.trade_history):
                        if trade['symbol'] == symbol and trade['action'] == 'BUY' and 'breakeven' in trade:
                            breakeven = trade['breakeven']
                            break
                    
                    print(f"{symbol:<10} {self.positions[symbol]:<10.2f} {price:<10.2f} {value:<15.2f} {stop_loss:<10.2f} {breakeven:<10.2f}")
        
        print("="*60 + "\n")


    def save_trading_history(self):
        """Save trading history to CSV file with tax and fee calculations"""
        if self.trade_history:
            # Enhanced trade history with more columns
            df = pd.DataFrame(self.trade_history)
            df.to_csv("paper_trading_history.csv", index=False)
            print("Trading history saved to paper_trading_history.csv")
            
            # Create a tax and fee summary report
            tax_report = []
            
            # Group trades by symbol
            symbols = set(trade['symbol'] for trade in self.trade_history)
            
            for symbol in symbols:
                symbol_trades = [t for t in self.trade_history if t['symbol'] == symbol]
                
                # Calculate metrics
                total_buy_value = sum(t['value'] for t in symbol_trades if t['action'] == 'BUY')
                total_sell_value = sum(t['value'] for t in symbol_trades if t['action'] in ['SELL', 'STOP LOSS SELL'])
                total_fees = sum(t.get('fees', 0) for t in symbol_trades)
                total_gross_pnl = sum(t.get('gross_pnl', 0) for t in symbol_trades if 'gross_pnl' in t)
                total_net_pnl = sum(t.get('net_pnl', 0) for t in symbol_trades if 'net_pnl' in t)
                
                # Add to report
                tax_report.append({
                    'symbol': symbol,
                    'total_buy_value': total_buy_value,
                    'total_sell_value': total_sell_value,
                    'total_fees': total_fees,
                    'total_gross_pnl': total_gross_pnl,
                    'total_net_pnl': total_net_pnl,
                    'fee_impact_percent': (total_fees / abs(total_gross_pnl) * 100) if total_gross_pnl != 0 else 0,
                    'estimated_tax': total_net_pnl * 0.30 if total_net_pnl > 0 else 0,  # Simplified 30% tax
                    'after_tax_profit': total_net_pnl * 0.70 if total_net_pnl > 0 else total_net_pnl  # After tax profit
                })
            
            # Save tax report
            tax_df = pd.DataFrame(tax_report)
            tax_df.to_csv("trading_tax_report.csv", index=False)
            print("Tax and fee report saved to trading_tax_report.csv")
            
            # Generate a daily P&L report
            if self.trade_history:
                # Convert timestamps to date strings
                for trade in self.trade_history:
                    if isinstance(trade['timestamp'], datetime.datetime):
                        trade['date'] = trade['timestamp'].strftime('%Y-%m-%d')
                    else:
                        trade['date'] = str(trade['timestamp']).split(' ')[0]
                
                # Group by date
                daily_pnl = {}
                for trade in self.trade_history:
                    date = trade['date']
                    if date not in daily_pnl:
                        daily_pnl[date] = {
                            'date': date,
                            'gross_pnl': 0,
                            'fees': 0,
                            'net_pnl': 0,
                            'trades': 0
                        }
                    
                    daily_pnl[date]['trades'] += 1
                    daily_pnl[date]['fees'] += trade.get('fees', 0)
                    
                    if 'gross_pnl' in trade:
                        daily_pnl[date]['gross_pnl'] += trade['gross_pnl']
                    
                    if 'net_pnl' in trade:
                        daily_pnl[date]['net_pnl'] += trade['net_pnl']
                
                # Convert to dataframe and save
                daily_df = pd.DataFrame(list(daily_pnl.values()))
                if not daily_df.empty:
                    # Calculate tax and after-tax profit
                    daily_df['estimated_tax'] = daily_df['net_pnl'].apply(lambda x: x * 0.30 if x > 0 else 0)
                    daily_df['after_tax_profit'] = daily_df['net_pnl'] - daily_df['estimated_tax']
                    
                    daily_df.to_csv("daily_trading_pnl.csv", index=False)
                    print("Daily P&L report saved to daily_trading_pnl.csv")
        
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
        
        # Set up trading monitor with email configuration
        self.monitor = TradingMonitor()
        
        # Set up logging
        self.log_file = log_file
        self._setup_logging()
        
        # Track market hours
        self.market_open = datetime.time(9, 15)  # NSE/BSE Open Time
        self.market_close = datetime.time(15, 30)  # NSE/BSE Close Time
        self.market_timezone = datetime.timezone(datetime.timedelta(hours=5, minutes=30))  # IST (UTC+5:30)
        self.ist = pytz.timezone('Asia/Kolkata')

        # Optional: add some known Indian market holidays (YYYY-MM-DD)
        self.holidays = {
            '2025-01-26', '2025-03-29', '2025-08-15', '2025-10-02', '2025-11-12'
            # You can add more from NSE/BSE official holiday list
        }
        
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

        now_ist = datetime.datetime.now(self.ist)
        self.log(f"now_ist -----------{now_ist}")
        today_str = now_ist.strftime('%Y-%m-%d')
        self.log(f"today_str ==========={today_str}")

        if now_ist.weekday() >= 5:  # Saturday or Sunday
            self.log(f"Market closed on weekends.")
            return False, "Market closed on weekends."

        if today_str in self.holidays:
            self.log(f"Market holiday on {today_str}.")
            return False

        market_open_dt = datetime.datetime.combine(now_ist.date(), self.market_open, tzinfo=self.ist)
        market_close_dt = datetime.datetime.combine(now_ist.date(), self.market_close, tzinfo=self.ist)

        if market_open_dt <= now_ist <= market_close_dt:
            self.log(f"Market is OPEN (Current IST: {now_ist.strftime('%H:%M:%S')})")
            return True
        else:
            self.log(f"Market is CLOSED (Current IST: {now_ist.strftime('%H:%M:%S')})")
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
    """
    Train a PPO RL model for stock trading using custom features and save it to disk.

    Parameters:
        symbol (str): Stock symbol (e.g., 'TCS.NS')
        start_date (str): Training start date (e.g., '2020-01-01')
        end_date (str): Training end date (e.g., '2024-12-31')
        timesteps (int): Number of training steps
        model_dir (str): Directory to save the trained model

    Returns:
        model (PPO): Trained PPO model, or None if training fails
    """
    try:
        print(f"🚀 Training model for {timesteps:,} timesteps...")
        # Create and use the training monitor
        monitor = TrainingMonitor(verbose=1)
        model.learn(total_timesteps=timesteps, callback=monitor)

        # Plot training progress
        monitor.plot_training_progress()

        print(f"\n🔧 Starting training for: {symbol}")
        
        # Initialize custom environment
        env = StockTradingEnv(symbol=symbol, start_date=start_date, end_date=end_date)

        # Debug: Inspect environment spaces
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Quick check: Take a random action
        obs, _ = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        random_action = env.action_space.sample()
        print(f"Sample action: {random_action}")
        obs, reward, done, _, _ = env.step(random_action)
        print(f"Observation after one step: {obs.shape}, Reward: {reward:.2f}")

        # Custom feature extractor for improved learning
        policy_kwargs = dict(
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )

        # Create PPO model
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=5e-4,  # Increased from 1e-4
            batch_size=128,      # Increased from 64
            n_steps=1024,        # Reduced from 2048 for more frequent updates
            ent_coef=0.05,       # Increased from 0.01 to encourage exploration
            gamma=0.99,
            # Add clip range to prevent extreme policy changes
            clip_range=0.2
        )

        print(f"🚀 Training model for {timesteps:,} timesteps...")
        model.learn(total_timesteps=timesteps)
        
        # Save the model
        os.makedirs(model_dir, exist_ok=True)
        clean_symbol = symbol.replace(".", "_")
        model_path = os.path.join(model_dir, f"ppo_{clean_symbol}")
        model.save(model_path)

        print(f"✅ Model saved to: {model_path}")
        return model

    except Exception as e:
        print(f"❌ Error training model for {symbol}: {e}")
        return None


def backtest_model(symbol, model, start_date, end_date):
    """Backtest a trained model on historical data with Zerodha trading fees"""
    try:
        print(f"Backtesting model for {symbol} with Zerodha fees...")

        # Create test environment
        env = StockTradingEnv(symbol=symbol, start_date=start_date, end_date=end_date)
        fee_calculator = TradingFees()  # Zerodha fee model

        # Initialize fee tracking
        fee_components = {
            'total': 0.0,
            'brokerage': 0.0,
            'stt': 0.0,
            'exchange_charges': 0.0,
            'gst': 0.0,
            'sebi_charges': 0.0,
            'stamp_duty': 0.0
        }

        total_trades = 0
        winning_trades = 0
        trade_history = []
        entry_price = 0.0
        entry_step = 0

        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        slippage = 0.001  # 0.1% slippage
        
        # Add maximum steps to prevent infinite loops
        max_steps = len(env.df)
        step_count = 0

        while not (done or truncated) and step_count < max_steps:
            step_count += 1
            
            try:
                action, _states = model.predict(obs, deterministic=True)
                buy_fraction, stop_loss_percent = action
                current_price = env.df.iloc[env.current_step]['close']
                
                # Debug information
                print(f"Step {env.current_step}/{max_steps}, Position: {env.position}, "
                      f"Price: {current_price:.2f}, Action: Buy={buy_fraction:.4f}, SL={stop_loss_percent:.4f}")
                
                # ENTER POSITION: Only try to enter if buy_fraction is significant
                if env.position == 0 and buy_fraction > 0.1:  # Threshold for buying
                    entry_price = current_price * (1 + slippage)
                    entry_step = env.current_step
                    print(f"ENTER: Step {env.current_step}, Price: {entry_price:.2f}")
                
                # EXIT POSITION
                if env.position > 0 and (current_price <= env.stop_loss or buy_fraction < 0.1):
                    exit_price = current_price * (1 - slippage)
                    print(f"EXIT: Step {env.current_step}, Entry: {entry_price:.2f}, Exit: {exit_price:.2f}")

                    sell_result = fee_calculator.calculate_net_profit(
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=env.position,
                        trade_type="INTRADAY"
                    )

                    for fee_type in fee_components:
                        if fee_type in sell_result:
                            fee_components[fee_type] += sell_result[fee_type]

                    trade_history.append({
                        'entry_step': entry_step,
                        'exit_step': env.current_step,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': env.position,
                        'pnl': sell_result['net_pnl'],
                        'fees': sell_result['total_charges'],
                        'return_pct': sell_result['net_pnl_percent']
                    })

                    total_trades += 1
                    if sell_result['net_pnl'] > 0:
                        winning_trades += 1

                    entry_price = 0.0

            except Exception as e:
                print(f"Model prediction error at step {env.current_step}: {e}")
                break

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Check if we've reached the end of data
            if env.current_step >= len(env.df) - 1:
                print("Reached end of data. Terminating backtest.")
                break

        print(f"Episode ended. Total steps: {step_count}/{max_steps}. Total reward: {total_reward}")

        # Close final open position
        if env.position > 0:
            current_price = env.df.iloc[env.current_step]['close']
            exit_price = current_price * (1 - slippage)
            print(f"FINAL EXIT: Step {env.current_step}, Entry: {entry_price:.2f}, Exit: {exit_price:.2f}")

            sell_result = fee_calculator.calculate_net_profit(
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=env.position,
                trade_type="INTRADAY"
            )

            for fee_type in fee_components:
                if fee_type in sell_result:
                    fee_components[fee_type] += sell_result[fee_type]

            trade_history.append({
                'entry_step': entry_step,
                'exit_step': env.current_step,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': env.position,
                'pnl': sell_result['net_pnl'],
                'fees': sell_result['total_charges'],
                'return_pct': sell_result['net_pnl_percent']
            })

            total_trades += 1
            if sell_result['net_pnl'] > 0:
                winning_trades += 1

        # Metrics
        initial_balance = env.initial_balance
        final_net_worth = info['net_worth']
        gross_return = (final_net_worth - initial_balance) / initial_balance * 100
        net_return_after_fees = gross_return - (fee_components['total'] / initial_balance * 100)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        percent_return = (final_net_worth - initial_balance) / initial_balance * 100

        print(f"\n=== Backtest Results for {symbol} (Zerodha Fees) ===")
        print(f"\nPerformance Metrics:")
        print(f"Initial Balance: ₹{initial_balance:.2f}")
        print(f"Final Net Worth: ₹{final_net_worth:.2f}")
        print(f"Gross Return: {gross_return:.2f}%")
        print(f"Net Return After Fees: {net_return_after_fees:.2f}%")
        print(f"Total Reward: {total_reward:.2f}")

        print(f"\nTrade Statistics:")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")

        print(f"\nZerodha Fee Breakdown:")
        print(f"Total Fees: ₹{fee_components['total']:.2f}")
        print(f" - STT: ₹{fee_components['stt']:.2f} (0.025% on sell side)")
        print(f" - Exchange Charges: ₹{fee_components['exchange_charges']:.2f} (0.00325%)")
        print(f" - GST: ₹{fee_components['gst']:.2f} (18% on exchange charges)")
        print(f" - SEBI Charges: ₹{fee_components['sebi_charges']:.2f} (₹10 per crore)")
        print(f" - Stamp Duty: ₹{fee_components['stamp_duty']:.2f} (0.003% on buy)")
        print("Note: Brokerage is ₹0 for equity trades on Zerodha")

        if trade_history:
            import os
            import pandas as pd
            os.makedirs("backtest_trades", exist_ok=True)
            file_path = os.path.join("backtest_trades", f"trades_{symbol}.csv")
            df = pd.DataFrame(trade_history)
            df['symbol'] = symbol
            df.to_csv(file_path, index=False)
            print(f"\nDetailed trade history saved to {file_path}")

        return {
            'symbol': symbol,
            'initial_balance': initial_balance,
            'final_net_worth': final_net_worth,
            'gross_return': gross_return,
            'fee_breakdown': fee_components,
            'net_return': net_return_after_fees,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'percent_return': percent_return,
            'total_reward': total_reward,
            'trade_history': trade_history
        }

    except Exception as e:
        import traceback
        print(f"Error backtesting model for {symbol}: {e}")
        print(traceback.format_exc())
        return None

def curriculum_training(symbol, start_date, end_date, timesteps=50000, model_dir="models"):
    """
    Train a model using curriculum learning - starting with easier environment settings
    and progressively making it harder
    """
    print(f"\n🔧 Starting curriculum training for: {symbol}")
    
    # Create policy kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    # Start with an easier environment (higher reward for trading)
    print("Phase 1: Training with easier environment (higher trading incentives)")
    env_easy = StockTradingEnv(
        symbol=symbol, 
        start_date=start_date, 
        end_date=end_date,
        min_holding_period=2  # Easier constraint
    )
    
    # Create model with higher exploration
    model = PPO(
        policy="MlpPolicy", 
        env=env_easy, 
        policy_kwargs=policy_kwargs,
        verbose=1, 
        ent_coef=0.1,  # High exploration
        learning_rate=1e-3,
        batch_size=128
    )
    
    # Phase 1 training
    monitor1 = TrainingMonitor(verbose=1)
    model.learn(total_timesteps=int(timesteps * 0.3), callback=monitor1)
    monitor1.plot_training_progress()
    
    # Progress to standard environment
    print("Phase 2: Training with standard environment")
    env_standard = StockTradingEnv(
        symbol=symbol, 
        start_date=start_date, 
        end_date=end_date,
        min_holding_period=5  # Standard constraint
    )
    model.set_env(env_standard)
    
    # Phase 2 training with slightly lower exploration
    model.ent_coef = 0.05
    model.learning_rate = 5e-4
    
    monitor2 = TrainingMonitor(verbose=1)
    model.learn(total_timesteps=int(timesteps * 0.7), callback=monitor2)
    monitor2.plot_training_progress()
    
    # Save final model
    os.makedirs(model_dir, exist_ok=True)
    clean_symbol = symbol.replace(".", "_")
    model_path = os.path.join(model_dir, f"ppo_{clean_symbol}")
    model.save(model_path)
    
    print(f"✅ Curriculum training complete for {symbol}")
    return model
  
def main():
    """Main function to run the trading system"""
    # Configuration
    # symbols = get_best_scalping_stocks()

    # print(f"Below are the symbols for scalping")
    # print(symbols)

    symbols = ["ABB"]
    
    training_start = '2015-02-02'
    training_end = '2023-12-31'
    test_start = '2024-01-01'
    test_end = '2025-02-07'
    model_dir = "models"
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # User menu
    while True:
        print("\n===== AI STOCK TRADING SYSTEM =====")
        print("1. Train models for all symbols")
        print("2. Train model for specific symbol")
        print("3. Train with curriculum learning (recommended)")  # Add this new option
        print("4. Backtest models")
        print("5. Run paper trading simulation")
        print("6. Run automated live trading")
        print("7. Setup system service")
        print("8. Exit")  # Changed from 7 to 8

        choice = input("\nEnter your choice (1-8): ")  # Update range


        if choice == '1':
            # Train models for all symbols
            for symbol in symbols:
                train_model(symbol, training_start, training_end, model_dir=model_dir)
                
        elif choice == '2':
            # Train for specific symbol
            symbol = input("Enter stock symbol to train model for: ").upper()
            if symbol:
                train_model(symbol, training_start, training_end, model_dir=model_dir)
        # Add the new option handling
        elif choice == '3':
            # Train using curriculum learning
            symbol = input("Enter stock symbol for curriculum training: ").upper()
            if symbol:
                timesteps = int(input("Enter training timesteps (default: 50000): ") or 50000)
                curriculum_training(symbol, training_start, training_end, timesteps, model_dir=model_dir)

        elif choice == '4':
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
            
        elif choice == '5':
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
                
        elif choice == '6':
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
                
        elif choice == '7':
            # Setup system service
            print("\nSetting up system service for automated trading...")
            setup_system_service()
            
        elif choice == '8':
            print("Exiting program. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")

# Add this at the end of your existing file
if __name__ == "__main__":
    # Set warnings filter
    # warnings.filterwarnings("ignore", category=FutureWarning)
    
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