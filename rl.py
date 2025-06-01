import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch as th # For consistency with SB3
from torch import nn
from sb3_contrib import RecurrentPPO
# For MlpLstmPolicy, RecurrentPPO will resolve "MlpLstmPolicy" string
# to sb3_contrib.common.recurrent.policies.RecurrentActorCriticLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# --- 1. Load and prepare your data ---
df_full = pd.read_csv('PFE.csv', parse_dates=['Date']).set_index('Date')
df_full['MA10'] = df_full['Close'].rolling(window=10).mean()
df_full['MA50'] = df_full['Close'].rolling(window=50).mean()
delta = df_full['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean() + 1e-8
rs = gain / loss
df_full['RSI'] = 100 - (100 / (1 + rs))
feature_cols = ['Open','High','Low','Close','Volume','Adj Close', 'MA10','MA50','RSI']
df_full = df_full[feature_cols].dropna()
ENV_WINDOW_SIZE = 100
ENV_COST = 0.0003

# --- Define the split date ---
split_date = pd.Timestamp('2010-01-01')

# --- Split the DataFrame based on the date ---
train_df = df_full[df_full.index < split_date].copy()
test_df  = df_full[df_full.index >= split_date].copy()

# --- Ensure that there is data in both train and test sets ---
if train_df.empty:
    raise ValueError("No training data before 2010-01-01. Please check your data or split date.")
if test_df.empty:
    raise ValueError("No testing data on or after 2010-01-01. Please check your data or split date.")

scaler = StandardScaler()
scaler.fit(train_df[feature_cols].values)

# --- 2. Custom always-in trading environment ---
class AlwaysInTradingEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, df, feature_columns, obs_scaler, window=ENV_WINDOW_SIZE, cost=ENV_COST, start_at_beginning=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.scaler = obs_scaler
        self.window = window
        self.cost = cost
        self.start_at_beginning = start_at_beginning
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window, len(self.feature_columns)), dtype=np.float32
        )
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.start_at_beginning:
            self.idx = self.window
        else:
            max_start_idx = len(self.df) - 2
            if max_start_idx < self.window:
                 self.idx = self.window
            else:
                 self.idx = np.random.randint(self.window, max_start_idx + 1)
        self.last_action = 1 # Start long
        self.done = False
        return self._obs(), {}
    def _obs(self):
        obs_raw = self.df[self.feature_columns].iloc[self.idx-self.window:self.idx].values
        obs_scaled = self.scaler.transform(obs_raw)
        return obs_scaled.astype(np.float32)
    def step(self, action):
        assert action in [0, 1], f"Invalid action: {action}"
        old_price = self.df['Adj Close'].iloc[self.idx-1]
        new_price = self.df['Adj Close'].iloc[self.idx]
        held_position = 1 if self.last_action == 1 else -1
        pnl = held_position * (new_price - old_price)
        transaction_cost = self.cost * abs(new_price) if action != self.last_action else 0
        reward = pnl - transaction_cost
        self.idx += 1
        self.last_action = action
        done = (self.idx >= len(self.df) -1)
        info = {"position_taken": 1 if action == 1 else -1, "pnl": pnl, "cost": transaction_cost, "current_price_for_eval": new_price}
        return self._obs(), reward, done, False, info

# --- 3. Environment wrappers ---
def make_train_env_fn():
    return AlwaysInTradingEnv(train_df, feature_cols, scaler, window=ENV_WINDOW_SIZE, cost=ENV_COST)
def make_eval_env_fn():
    return AlwaysInTradingEnv(test_df, feature_cols, scaler, window=ENV_WINDOW_SIZE, cost=ENV_COST, start_at_beginning=False)
def make_final_eval_env_fn():
    return AlwaysInTradingEnv(test_df, feature_cols, scaler, window=ENV_WINDOW_SIZE, cost=ENV_COST, start_at_beginning=True)
train_env = DummyVecEnv([make_train_env_fn])
eval_callback_env = DummyVecEnv([make_eval_env_fn])
final_eval_env = DummyVecEnv([make_final_eval_env_fn])

# --- 4. LSTM Implementation using SB3-Contrib ---
# This file uses the standard RecurrentPPO with MlpLstmPolicy from SB3-Contrib.
# No custom LSTM implementation is needed.

# --- 5. Training and evaluation logic ---
if __name__ == "__main__":
    print("--- Training LSTM Agent ---")

    # These are the default RecurrentPPO LSTM parameters
    LSTM_HIDDEN_SIZE = 256 
    N_LSTM_LAYERS = 1

    model = RecurrentPPO(
        policy="MlpLstmPolicy", # Use standard MlpLstmPolicy
        env=train_env,
        n_steps=256, 
        batch_size=64, 
        n_epochs=10, 
        learning_rate=3e-4, 
        ent_coef=0.01, 
        gamma=0.99, 
        gae_lambda=0.95, 
        clip_range=0.2, 
        verbose=1,
        tensorboard_log="./lstm_trading_pfe_1M/",
        policy_kwargs=dict(
            lstm_hidden_size=LSTM_HIDDEN_SIZE,
            n_lstm_layers=N_LSTM_LAYERS,
            enable_critic_lstm=True, 
            # Use empty net_arch to have LSTM directly connected to output layers
            net_arch=[], 
            # activation_fn=th.nn.ReLU # Example, default is Tanh
        )
    )

    eval_callback = EvalCallback(eval_callback_env,
                                 best_model_save_path="./best_model_lstm_pfe_1M/",
                                 log_path="./logs_lstm_pfe_1M/",
                                 eval_freq=max(5000 // train_env.num_envs, 1),
                                 deterministic=True,
                                 render=False)

    print("Training LSTM model...")
    model.learn(total_timesteps=1_000_000, callback=eval_callback) # 1 Million timesteps training
    model.save("ppo_lstm_trading_pfe_final_1M")
    print("Training complete.")
    
    # Short training for demonstration (optional):
    # model.learn(total_timesteps=1000, callback=eval_callback)
    # model.save("ppo_lstm_trading_pfe_temp")
    # print("Short training complete. Model saved to ppo_lstm_trading_pfe_temp")

    # --- Load pre-trained model ---
    model_loaded = False
    best_model_path = "./best_model_lstm_pfe_1M/best_model"
    final_model_path = "ppo_lstm_trading_pfe_final_1M"

    try:
        model = RecurrentPPO.load(best_model_path, env=final_eval_env)
        print(f"Loaded best LSTM model from {best_model_path}.")
        model_loaded = True
    except Exception as e1:
        print(f"Could not load {best_model_path}: {e1}")
        try:
            model = RecurrentPPO.load(final_model_path, env=final_eval_env)
            print(f"Loaded final LSTM model from {final_model_path}.zip (or without .zip if specified).")
            model_loaded = True
        except Exception as e2:
            print(f"Could not load {final_model_path} (with/without .zip): {e2}")
            print("No pre-trained LSTM model found. Using the freshly initialized (untrained) model for evaluation.")

    if not model_loaded and 'model' not in locals():
        print("CRITICAL: LSTM Model was not initialized before attempting to load. This should not happen.")
        # Fallback: re-initialize
        model = RecurrentPPO(
            policy="MlpLstmPolicy", env=train_env, n_steps=256, batch_size=64, n_epochs=10,
            learning_rate=3e-4, ent_coef=0.01, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            verbose=0, tensorboard_log="./lstm_trading_pfe_1M/",
            policy_kwargs=dict(
                lstm_hidden_size=LSTM_HIDDEN_SIZE,
                n_lstm_layers=N_LSTM_LAYERS,
                enable_critic_lstm=True,
                net_arch=[]
            )
        )
        print("Re-initialized a new LSTM model as a fallback.")
    elif not model_loaded:
        print("Warning: Proceeding with an untrained or freshly initialized LSTM model.")

    # --- Evaluation ---
    print("\n--- Evaluating LSTM AGENT on test set ---")
    obs_eval = final_eval_env.reset() 
    lstm_states_eval = None 
    episode_starts_eval = np.ones((final_eval_env.num_envs,), dtype=bool)
    done_eval_loop = np.array([False] * final_eval_env.num_envs)

    agent_actions_hist, agent_rewards_hist, agent_pnl_hist, agent_cost_hist = [], [], [], []
    
    num_steps_eval = 0
    max_steps_eval = len(test_df) - ENV_WINDOW_SIZE -1 

    while not done_eval_loop[0] and num_steps_eval < max_steps_eval:
        action_pred, lstm_states_eval = model.predict(
            obs_eval,
            state=lstm_states_eval,
            episode_start=episode_starts_eval,
            deterministic=True
        )
        obs_eval, reward_eval, dones_eval_vec, info_eval_vec = final_eval_env.step(action_pred)
        
        done_eval_loop = dones_eval_vec
        episode_starts_eval = dones_eval_vec 

        action_taken = action_pred[0]
        reward_val = reward_eval[0]
        info = info_eval_vec[0]

        agent_actions_hist.append(action_taken)
        agent_rewards_hist.append(reward_val)
        agent_pnl_hist.append(info.get('pnl',0))
        agent_cost_hist.append(info.get('cost',0))
        
        num_steps_eval += 1

    print(f"Agent evaluation finished after {num_steps_eval} steps.")
    total_agent_reward = sum(agent_rewards_hist)
    print(f"Total Agent Reward: {total_agent_reward:.2f}")
    print(f"Average Agent Reward per step: {np.mean(agent_rewards_hist) if agent_rewards_hist else 0:.4f}")

    # --- 6. Calculate "ALWAYS LONG" BASELINE performance on test set ---
    print("\n--- Calculating ALWAYS LONG (Buy and Hold) BASELINE on test set ---")
    baseline_rewards = []
    baseline_last_action = 1 

    start_idx_new_price = ENV_WINDOW_SIZE
    end_idx_new_price = len(test_df) - 2 

    if start_idx_new_price > end_idx_new_price or len(test_df) <= ENV_WINDOW_SIZE :
        print("Test data too short for baseline calculation.")
        total_baseline_reward = 0
        num_baseline_steps = 0
    else:
        num_baseline_steps = 0
        for current_idx_for_new_price in range(start_idx_new_price, end_idx_new_price + 1):
            old_price = test_df['Adj Close'].iloc[current_idx_for_new_price - 1]
            new_price = test_df['Adj Close'].iloc[current_idx_for_new_price]

            held_position = 1 
            pnl = held_position * (new_price - old_price)

            current_baseline_action = 1 
            transaction_cost = ENV_COST * abs(new_price) if current_baseline_action != baseline_last_action else 0
            
            reward_step = pnl - transaction_cost
            baseline_rewards.append(reward_step)
            
            baseline_last_action = current_baseline_action 
            num_baseline_steps +=1

        total_baseline_reward = sum(baseline_rewards)

    print(f"Baseline evaluation finished after {num_baseline_steps} steps.")
    print(f"Total Baseline (Always Long) Reward: {total_baseline_reward:.2f}")
    print(f"Average Baseline Reward per step: {np.mean(baseline_rewards) if baseline_rewards else 0:.4f}")

    # --- 7. Comparison and Plotting ---
    print("\n--- COMPARISON ---")
    print(f"Agent Total Reward: {total_agent_reward:.2f}")
    print(f"Baseline (Always Long) Total Reward: {total_baseline_reward:.2f}")
    if total_baseline_reward != 0:
        print(f"Agent performance vs Baseline: {(total_agent_reward/total_baseline_reward)*100:.2f}%")
    else:
        if total_agent_reward > 0:
            print("Agent performance vs Baseline: Agent profitable, Baseline zero.")
        elif total_agent_reward < 0:
            print("Agent performance vs Baseline: Agent loss, Baseline zero.")
        else:
            print("Agent performance vs Baseline: Both Agent and Baseline zero.")

    if agent_rewards_hist:
        plt.figure(figsize=(14, 7))
        plt.plot(np.cumsum(agent_rewards_hist), label='LSTM Agent Cumulative P&L')
        if baseline_rewards:
            plt.plot(np.cumsum(baseline_rewards), label='Baseline (Always Long) Cumulative P&L', linestyle='--')
        plt.title('LSTM Agent vs Baseline Cumulative P&L on Test Set')
        plt.xlabel('Time Step in Test Set')
        plt.ylabel('Cumulative Reward/P&L')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(14, 4))
        plt.plot(agent_actions_hist,'.', label='Agent Actions (0=Short, 1=Long)')
        plt.title('LSTM Agent Actions Taken During Test')
        plt.xlabel('Time Step')
        plt.ylabel('Action')
        plt.yticks([0,1])
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No agent rewards recorded for plotting.")