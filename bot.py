import gymnasium as gym
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import os
from finta import TA
from stable_baselines3 import PPO
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
        observations = observations.type(torch.float64)
        # Add batch dimension if not present
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(0)
        features, _ = self.lstm(observations)
        return features

class StockTradingEnv(gym.Env):
    def __init__(self, symbol, start_date, end_date, timeframe='1d'):
        super(StockTradingEnv, self).__init__()
        
        self.symbol = symbol
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
            
            # Add more robust technical indicators - force all data to float64 for consistency
            self.df = self.df.astype('float64')  # Ensure consistent data type
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
            
            # Replace scaled values in dataframe and ensure they're float64
            for i, col in enumerate(self.feature_columns):
                self.df[col] = scaled_features[:, i].astype(np.float64)
            
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
        self.action_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float64)
        
        # Observation space with strictly float64 type
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(len(self.feature_columns),), 
            dtype=np.float64
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
            
            # Ensure the output is float64 as expected by gym
            obs = obs.astype(np.float64)
            
            # Explicitly replace any non-finite values
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
            
            return obs
            
        except Exception as e:
            print(f"Error getting observation for {self.symbol}: {e}")
            # Return safe default observation
            return np.zeros(len(self.feature_columns), dtype=np.float64)

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

class MultiSymbolTrainer:
    def __init__(self, symbols, start_date_train, end_date_train, start_date_test, end_date_test):
        self.symbols = symbols
        self.start_date_train = start_date_train
        self.end_date_train = end_date_train
        self.start_date_test = start_date_test
        self.end_date_test = end_date_test
        self.models = {}
        self.results = {}
        
    def train_all(self, timesteps=10000):
        """Train models for all symbols"""
        for symbol in self.symbols:
            symbol_with_suffix = f"{symbol}.NS"  # Add .NS suffix for Indian stocks
            model_path = f"models/ppo_{symbol}"
            
            # Ensure directory exists
            os.makedirs("models", exist_ok=True)
            
            try:
                print(f"\n{'='*50}")
                print(f"Training model for {symbol}")
                print(f"{'='*50}")
                
                # Create environment
                env = StockTradingEnv(
                    symbol=symbol_with_suffix, 
                    start_date=self.start_date_train, 
                    end_date=self.end_date_train
                )
                
                # Create model
                model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    learning_rate=0.0001,
                    gamma=0.95,
                    n_steps=1024,
                    batch_size=64,
                    ent_coef=0.01,
                    clip_range=0.2,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(
                        net_arch=dict(pi=[64, 64], vf=[64, 64]),
                        activation_fn=nn.ReLU
                    )
                )
                
                # Train model
                model.learn(total_timesteps=timesteps)
                
                # Save model
                model.save(model_path)
                print(f"Model for {symbol} saved to {model_path}")
                
                # Store model reference
                self.models[symbol] = model
                
            except Exception as e:
                import traceback
                print(f"Error training model for {symbol}: {e}")
                traceback.print_exc()
                
    def test_all(self):
        """Test all trained models"""
        summary_results = []
        
        for symbol in self.symbols:
            symbol_with_suffix = f"{symbol}.NS"
            model_path = f"models/ppo_{symbol}"
            
            try:
                print(f"\n{'='*50}")
                print(f"Testing model for {symbol}")
                print(f"{'='*50}")
                
                # Check if model exists
                if not os.path.exists(f"{model_path}.zip"):
                    print(f"Model for {symbol} not found. Skipping...")
                    continue
                    
                # Create test environment
                env = StockTradingEnv(
                    symbol=symbol_with_suffix, 
                    start_date=self.start_date_test, 
                    end_date=self.end_date_test
                )
                
                # Load model
                model = PPO.load(model_path)
                
                # Test model
                obs, _ = env.reset()
                done = False
                total_reward = 0.0
                
                # Results tracking
                net_worths = [env.net_worth]
                actions = []
                
                # Run simulation
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    
                    buy_fraction, stop_loss_percent = action
                    action_str = f"Buy {buy_fraction:.2f}, SL {stop_loss_percent:.2f}"
                    actions.append(action_str)
                    net_worths.append(env.net_worth)
                
                # Calculate metrics
                initial_balance = env.initial_balance
                final_worth = env.net_worth
                percent_return = (final_worth - initial_balance) / initial_balance * 100
                
                # Store results
                result = {
                    'symbol': symbol,
                    'initial_balance': initial_balance,
                    'final_worth': final_worth,
                    'total_reward': total_reward,
                    'percent_return': percent_return,
                    'net_worths': net_worths,
                    'actions': actions
                }
                
                self.results[symbol] = result
                summary_results.append({
                    'symbol': symbol,
                    'initial_balance': initial_balance,
                    'final_worth': final_worth,
                    'percent_return': percent_return
                })
                
                # Print summary
                print(f"\nResults for {symbol}:")
                print(f"Initial Balance: ₹{initial_balance:.2f}")
                print(f"Final Net Worth: ₹{final_worth:.2f}")
                print(f"Return: {percent_return:.2f}%")
                
            except Exception as e:
                import traceback
                print(f"Error testing model for {symbol}: {e}")
                traceback.print_exc()
                
        # Create summary dataframe
        if summary_results:
            summary_df = pd.DataFrame(summary_results)
            summary_df = summary_df.sort_values('percent_return', ascending=False)
            
            print("\n==== OVERALL PERFORMANCE SUMMARY ====")
            print(summary_df)
            
            # Save results to CSV
            summary_df.to_csv("trading_results_summary.csv", index=False)
            print("Results saved to trading_results_summary.csv")
            
            # Plot top performers
            self.plot_top_performers(summary_df, top_n=5)
            
    def plot_top_performers(self, summary_df, top_n=5):
        """Plot the performance of top N performers"""
        top_symbols = summary_df.nlargest(top_n, 'percent_return')['symbol'].tolist()
        
        plt.figure(figsize=(15, 10))
        
        for symbol in top_symbols:
            if symbol in self.results:
                result = self.results[symbol]
                # Convert to percentage gain for better comparison
                normalized_worth = [100 * (nw / result['initial_balance']) for nw in result['net_worths']]
                plt.plot(normalized_worth, label=f"{symbol} ({result['percent_return']:.2f}%)")
        
        plt.title(f'Top {top_n} Performing Stocks (% Return)')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig("top_performers.png")
        plt.show()

if __name__ == "__main__":
    # Define the Nifty 100 stocks
    nifty100_stocks = ["ABB", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "AMBUJACEM", 
                      "APOLLOHOSP", "ASIANPAINT", "DMART", "AXISBANK","BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
                      "BAJAJHLDNG", "BAJAJHFL", "BANKBARODA", "BEL", "BPCL", "BHARTIARTL", "BOSCHLTD", "BRITANNIA",
                      "CGPOWER", "CANBK", "CHOLAFIN", "CIPLA", "COALINDIA", "DLF", "DABUR", "DIVISLAB", "DRREDDY", 
                      "EICHERMOT", "ETERNAL", "GAIL", "GODREJCP", "GRASIM", "HCLTECH", "HDFCBANK","HDFCLIFE", 
                      "HAVELLS", "HEROMOTOCO", "HINDALCO", "HAL", "HINDUNILVR", "HYUNDAI", "ICICIBANK","ICICIGI", 
                      "ICICIPRULI", "ITC", "INDHOTEL", "IOC", "IRFC", "INDUSINDBK", "NAUKRI", "INFY","INDIGO", 
                      "JSWENERGY", "JSWSTEEL", "JINDALSTEL", "JIOFIN", "KOTAKBANK", "LTIM", "LT", "LICI","LODHA", 
                      "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC", "PIDILITIND", "PFC", "POWERGRID", "PNB",
                      "RECLTD", "RELIANCE", "SBILIFE", "MOTHERSON", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SBIN",
                      "SUNPHARMA", "SWIGGY", "TVSMOTOR", "TCS", "TATACONSUM", "TATAMOTORS", "TATAPOWER", "TATASTEEL",
                      "TECHM", "TITAN", "TORNTPHARM", "TRENT", "ULTRACEMCO", "UNITDSPR", "VBL", "VEDL", "WIPRO", 
                      "ZYDUSLIFE"]
    
    # For demonstration and testing, use a subset of stocks
    # Uncomment this when you want to run with just a few stocks initially
    test_stocks = ["TCS", "RELIANCE", "INFY", "HDFCBANK", "ICICIBANK"]
    
    print("Stock Trading RL Agent for Multiple Symbols")
    print("1. Train models for all stocks")
    print("2. Test models for all stocks")
    print("3. Train and test models for all stocks")
    print("4. Use subset of stocks for quick testing")
    
    choice = input("Enter your choice (1-4): ")
    
    # Set date ranges
    train_start = '2020-01-01'
    train_end = '2023-12-31'
    test_start = '2024-01-01'
    test_end = '2025-04-16'
    
    if choice == "1":
        print("\nStarting training for all Nifty 100 stocks...")
        trainer = MultiSymbolTrainer(nifty100_stocks, train_start, train_end, test_start, test_end)
        trainer.train_all(timesteps=10000)
    elif choice == "2":
        print("\nTesting models for all Nifty 100 stocks...")
        trainer = MultiSymbolTrainer(nifty100_stocks, train_start, train_end, test_start, test_end)
        trainer.test_all()
    elif choice == "3":
        print("\nTraining and testing for all Nifty 100 stocks...")
        trainer = MultiSymbolTrainer(nifty100_stocks, train_start, train_end, test_start, test_end)
        trainer.train_all(timesteps=10000)
        trainer.test_all()
    elif choice == "4":
        print("\nRunning quick test with subset of stocks...")
        trainer = MultiSymbolTrainer(test_stocks, train_start, train_end, test_start, test_end)
        trainer.train_all(timesteps=5000)  # Reduced timesteps for quick testing
        trainer.test_all()
    else:
        print("Invalid choice. Please run the program again.")