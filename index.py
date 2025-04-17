import gymnasium as gym
import pandas as pd
import numpy as np
import yfinance as yf
from finta import TA
from stable_baselines3 import PPO
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
warnings.filterwarnings('ignore')  # Suppress warnings

class StockTradingEnv(gym.Env):
    def __init__(self, symbol, start_date, end_date, timeframe='1d'):
        super(StockTradingEnv, self).__init__()
        
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        
        # Download stock data using yfinance
        print(f"Downloading data for {symbol} from {start_date} to {end_date}...")
        self.df = yf.download(self.symbol, start=self.start_date, end=self.end_date, interval=self.timeframe)
        print(f"Downloaded {len(self.df)} rows of data")
        
        # Check if we have enough data
        if len(self.df) < 30:  # Minimum data needed for indicators
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
        
        try:
            # Calculate robust indicators with error handling
            self.df['rsi'] = TA.RSI(self.df, period=14)
            self.df['atr'] = TA.ATR(self.df, period=14)
            self.df['ma20'] = TA.SMA(self.df, period=20)
            self.df['ma50'] = TA.SMA(self.df, period=50)
            macd_data = TA.MACD(self.df)
            self.df['macd'] = macd_data['MACD']
            self.df['signal'] = macd_data['SIGNAL']
        except Exception as e:
            print(f"Error calculating indicators with finta: {e}")
            # Fallback to manual calculation
            self.df['rsi'] = self.df['close'].pct_change().rolling(window=14).mean()
            self.df['atr'] = (self.df['high'] - self.df['low']).rolling(window=14).mean()
            self.df['ma20'] = self.df['close'].rolling(window=20).mean()
            self.df['ma50'] = self.df['close'].rolling(window=50).mean()
            self.df['macd'] = self.df['close'].ewm(span=12).mean() - self.df['close'].ewm(span=26).mean()
            self.df['signal'] = self.df['macd'].ewm(span=9).mean()
        
        # Fill NaN values
        self.df.fillna(method='ffill', inplace=True)  # Forward fill
        self.df.fillna(method='bfill', inplace=True)  # Backward fill
        self.df.fillna(0, inplace=True)  # Replace any remaining NaNs with 0
        
        # Clip extreme values to prevent issues with scaling
        for col in self.df.columns:
            if col != 'volume':  # Skip volume as it can have legitimately high values
                q_low = self.df[col].quantile(0.01)
                q_high = self.df[col].quantile(0.99)
                self.df[col] = self.df[col].clip(q_low, q_high)
        
        # Define features and ensure they exist
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'atr', 'ma20', 'ma50', 'macd', 'signal']
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
            print("Warning: Non-finite values found after preprocessing. Replacing with zeros.")
            for col in self.feature_columns:
                self.df[col] = np.nan_to_num(self.df[col], nan=0.0, posinf=1.0, neginf=0.0)
        
        # Action space: Buy (0), Sell (1), Hold (2)
        self.action_space = spaces.Discrete(3)
        
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

    def reset(self, *, seed=None, options=None):
        # Updated reset method to match new Gymnasium API
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.done = False
        
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
            print(f"Error getting observation: {e}")
            # Return safe default observation
            return np.zeros(len(self.feature_columns), dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, self.done, False, {}
        
        # Safely get current price
        try:
            current_price = float(self.df.iloc[self.current_step]['close'])
        except (IndexError, KeyError) as e:
            print(f"Error getting price: {e}")
            current_price = 0.0
        
        prev_net_worth = self.net_worth
        
        # Record state before action for reward calculation
        prev_position = self.position
        prev_balance = self.balance
        
        # Execute action with proper error handling
        try:
            if action == 0:  # Buy
                if self.balance > 0:
                    shares_bought = self.balance / current_price if current_price > 0 else 0
                    self.position += shares_bought
                    self.balance = 0.0
            elif action == 1:  # Sell
                if self.position > 0:
                    self.balance += self.position * current_price
                    self.position = 0.0
            # action == 2 is Hold, do nothing
        except Exception as e:
            print(f"Error executing action: {e}")
        
        # Move to next step
        self.current_step += 1
        
        # Calculate values with error handling
        try:
            if self.current_step < len(self.df):
                current_price = float(self.df.iloc[self.current_step]['close'])
                
            self.net_worth = self.balance + (self.position * current_price)
            self.max_net_worth = max(self.max_net_worth, self.net_worth)
        except Exception as e:
            print(f"Error calculating net worth: {e}")
        
        # Check if done
        self.done = self.current_step >= len(self.df) - 1
        
        # Calculate reward with safe operations
        try:
            reward = (self.net_worth - prev_net_worth) / self.initial_balance
            
            # Add penalty for excessive trading to encourage holding
            trade_penalty = 0.0005  # 0.05% trading fee/penalty
            if (action == 0 and prev_balance > 0) or (action == 1 and prev_position > 0):
                reward -= trade_penalty
                
            # Clip reward to prevent numerical issues
            reward = np.clip(reward, -1.0, 1.0)
            reward = float(reward)  # Ensure it's a Python float
        except Exception as e:
            print(f"Error calculating reward: {e}")
            reward = 0.0
        
        # Get observation with safety checks
        obs = self._get_obs()
        
        # Return with updated Gymnasium API
        info = {
            'net_worth': float(self.net_worth),
            'balance': float(self.balance),
            'position': float(self.position),
            'step': int(self.current_step)
        }
        
        return obs, reward, self.done, False, info

# Define the model path
MODEL_PATH = "ppo_stock_trading_model"

# Define a function to train the RL agent with proper error handling
def train_agent():
    try:
        # Create environment
        env = StockTradingEnv(symbol='TCS.NS', start_date='2020-01-01', end_date='2021-01-01', timeframe='1d')
        
        # Check obs space and action space
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test a sample observation
        obs, _ = env.reset()
        print(f"Sample observation shape: {obs.shape}, dtype: {obs.dtype}")
        print(f"Sample observation: {obs}")
        
        # Create model with proper type checking
        model = PPO(
            'MlpPolicy', 
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
                net_arch=[64, 64]
            )
        )
        
        print("Starting model training...")
        model.learn(total_timesteps=10000)
        
        # Save the model
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        
        # Verify the model file exists
        if os.path.exists(f"{MODEL_PATH}.zip"):
            print(f"Confirmed: Model file {MODEL_PATH}.zip exists")
        else:
            print(f"Warning: Model file {MODEL_PATH}.zip was not found after saving")
        
    except Exception as e:
        import traceback
        print(f"Error in training: {e}")
        traceback.print_exc()

# Define a function to test the RL agent
def test_agent():
    try:
        # Check if model file exists before attempting to load
        model_file = f"{MODEL_PATH}.zip"
        if not os.path.exists(model_file):
            print(f"Error: Model file {model_file} not found. You need to train the model first.")
            return
            
        env = StockTradingEnv(symbol='TCS.NS', start_date='2021-01-01', end_date='2021-06-01', timeframe='1d')
        
        print(f"Loading model from {MODEL_PATH}...")
        model = PPO.load(MODEL_PATH)
        print("Model loaded successfully")
        
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        
        print("Starting evaluation...")
        
        # Record results for plotting
        net_worths = [env.net_worth]
        actions_taken = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            action_name = ["Buy", "Sell", "Hold"][action]
            actions_taken.append(action_name)
            net_worths.append(env.net_worth)
            
            print(f"Step: {env.current_step}, Action: {action_name}, Reward: {reward:.4f}, Net worth: {env.net_worth:.2f}")
        
        # Print summary results
        print("\nTrading Summary:")
        print(f"Initial Balance: ₹{env.initial_balance:.2f}")
        print(f"Final Net Worth: ₹{env.net_worth:.2f}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Return: {(env.net_worth - env.initial_balance) / env.initial_balance * 100:.2f}%")
        
        # Count actions
        buy_count = actions_taken.count("Buy")
        sell_count = actions_taken.count("Sell")
        hold_count = actions_taken.count("Hold")
        
        print(f"\nAction Distribution:")
        print(f"Buy actions: {buy_count} ({buy_count/len(actions_taken)*100:.1f}%)")
        print(f"Sell actions: {sell_count} ({sell_count/len(actions_taken)*100:.1f}%)")
        print(f"Hold actions: {hold_count} ({hold_count/len(actions_taken)*100:.1f}%)")
        
    except Exception as e:
        import traceback
        print(f"Error in testing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Stock Trading RL Agent")
    print("1. Train model")
    print("2. Test model")
    print("3. Train and test model")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        print("\nStarting training...")
        train_agent()
    elif choice == "2":
        print("\nStarting testing...")
        test_agent()
    elif choice == "3":
        print("\nStarting training...")
        train_agent()
        print("\nStarting testing...")
        test_agent()
    else:
        print("Invalid choice. Please run the program again.")