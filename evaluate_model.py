import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch as th # For consistency with SB3
from torch import nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
# EvalCallback is not needed for evaluation-only script
import matplotlib.pyplot as plt
import argparse # For command-line arguments
from stable_baselines3.common.policies import ActorCriticPolicy # For MlpLstmPolicy
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# Conditional xLSTM class definitions - only if policy_type is 'xlstm'
# These will be defined inside main() if needed.

# --- AlwaysInTradingEnv Definition (remains the same) ---
class AlwaysInTradingEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, df, feature_columns, obs_scaler, window, cost, start_at_beginning=False): # Added window, cost as direct args
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.scaler = obs_scaler
        self.window = window # Use passed window
        self.cost = cost       # Use passed cost
        self.start_at_beginning = start_at_beginning
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window, len(self.feature_columns)), dtype=np.float32
        )
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # For deterministic evaluation, always start from the beginning of the test set
        self.idx = self.window
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

# Global definitions for xLSTM components (if needed by MlpXLstmPolicy)
# These will be properly defined within main if policy_type == 'xlstm'
xLSTMCellWrapper = None
xLSTM = None
MlpXLstmPolicy = None

def main():
    global xLSTMCellWrapper, xLSTM, MlpXLstmPolicy # Allow modification of globals

    parser = argparse.ArgumentParser(description="Evaluate a trained trading agent.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model (.zip file).")
    parser.add_argument("--policy_type", type=str, required=True, default="lstm", choices=["lstm", "xlstm"], help="Type of policy the model uses ('lstm' or 'xlstm').")
    parser.add_argument("--env_window_size", type=int, default=100, help="Observation window size for the environment.")
    parser.add_argument("--env_cost", type=float, default=0.0003, help="Transaction cost for the environment.")
    parser.add_argument("--data_path", type=str, default="PFE.csv", help="Path to the financial data CSV file.")
    parser.add_argument("--split_date", type=str, default="2010-01-01", help="Date to split train/test data (YYYY-MM-DD).")

    args = parser.parse_args()

    # --- xLSTM Definitions (only if policy_type is 'xlstm') ---
    if args.policy_type == "xlstm":
        class xLSTMCellWrapper_class(nn.Module):
            def __init__(self, input_size, hidden_size, bias=True):
                super().__init__()
                self.mixing_layer = nn.Linear(input_size, hidden_size, bias=bias)
                self.mix_activation = nn.Tanh()
            def forward(self, lstm_output: th.Tensor) -> th.Tensor:
                mixed_output = self.mixing_layer(lstm_output)
                activated_output = self.mix_activation(mixed_output)
                return activated_output
        xLSTMCellWrapper = xLSTMCellWrapper_class

        class xLSTM_class(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, n_layers: int = 1, bias: bool = True, batch_first: bool = False):
                super().__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.n_layers = n_layers
                self.bias = bias
                self.batch_first = batch_first
                self.lstm = nn.LSTM(
                    input_size=input_size, hidden_size=hidden_size,
                    num_layers=n_layers, bias=bias, batch_first=batch_first
                )
                self.mixer = xLSTMCellWrapper(hidden_size, hidden_size, bias=bias) # type: ignore
            def forward(self, x: th.Tensor, hidden_states: Optional[Tuple[th.Tensor, th.Tensor]] = None) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
                lstm_out, (h_n, c_n) = self.lstm(x, hidden_states)
                mixed_output = self.mixer(lstm_out)
                return mixed_output, (h_n, c_n)
        xLSTM = xLSTM_class

        class MlpXLstmPolicy_class(RecurrentActorCriticPolicy):
            def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, lr_schedule: Any,
                         net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None, activation_fn: Type[nn.Module] = nn.Tanh,
                         features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor, features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam, optimizer_kwargs: Optional[Dict[str, Any]] = None,
                         lstm_hidden_size: int = 256, n_lstm_layers: int = 1, shared_lstm: bool = False, enable_critic_lstm: bool = True, **kwargs):
                if net_arch is None: net_arch = []
                self.xlstm_hidden_size = lstm_hidden_size
                self.xlstm_n_layers = n_lstm_layers
                self.xlstm_shared = shared_lstm
                self.xlstm_enable_critic = enable_critic_lstm
                super().__init__(observation_space=observation_space, action_space=action_space, lr_schedule=lr_schedule, net_arch=net_arch,
                                 activation_fn=activation_fn, features_extractor_class=features_extractor_class, features_extractor_kwargs=features_extractor_kwargs,
                                 optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs, lstm_hidden_size=lstm_hidden_size,
                                 n_lstm_layers=n_lstm_layers, shared_lstm=shared_lstm, enable_critic_lstm=enable_critic_lstm, **kwargs)

            def _setup_model(self) -> None:
                super()._setup_model()
                def replace_lstm_with_xlstm(original_lstm_module, module_name):
                    if original_lstm_module is None:
                        print(f"{module_name} is None, skipping replacement.")
                        return None
                    original_input_size = original_lstm_module.input_size
                    print(f"Replacing {module_name} LSTM with xLSTM (input_size={original_input_size}, hidden_size={self.xlstm_hidden_size}, n_layers={self.xlstm_n_layers})")
                    return xLSTM(input_size=original_input_size, hidden_size=self.xlstm_hidden_size, # type: ignore
                                   n_layers=self.xlstm_n_layers, batch_first=False)
                if hasattr(self, 'lstm_actor'):
                    self.lstm_actor = replace_lstm_with_xlstm(self.lstm_actor, "actor")
                if self.xlstm_enable_critic:
                    if self.xlstm_shared:
                        self.lstm_critic = self.lstm_actor
                    elif hasattr(self, 'lstm_critic'):
                        self.lstm_critic = replace_lstm_with_xlstm(self.lstm_critic, "critic")
        MlpXLstmPolicy = MlpXLstmPolicy_class


    # --- 1. Load and prepare data ---
    print(f"Loading data from {args.data_path}...")
    try:
        df_full = pd.read_csv(args.data_path, parse_dates=['Date']).set_index('Date')
    except FileNotFoundError:
        print(f"Error: Data file '{args.data_path}' not found.")
        exit(1)

    df_full['MA10'] = df_full['Close'].rolling(window=10).mean()
    df_full['MA50'] = df_full['Close'].rolling(window=50).mean()
    delta = df_full['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean() + 1e-8
    rs = gain / loss
    df_full['RSI'] = 100 - (100 / (1 + rs))
    feature_cols = ['Open','High','Low','Close','Volume','Adj Close', 'MA10','MA50','RSI']
    df_full = df_full[feature_cols].dropna()
    
    # Use args for ENV_WINDOW_SIZE and ENV_COST
    ENV_WINDOW_SIZE = args.env_window_size
    ENV_COST = args.env_cost

    split_dt = pd.Timestamp(args.split_date)
    train_df = df_full[df_full.index < split_dt].copy()
    test_df  = df_full[df_full.index >= split_dt].copy()

    if train_df.empty:
        print(f"Error: No training data before {args.split_date} for scaler. Check data or split date.")
        exit(1)
    if test_df.empty:
        print(f"Error: No testing data on or after {args.split_date}. Check data or split date.")
        exit(1)

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    # --- 2. Environment Setup ---
    def make_eval_env_fn():
        # Pass ENV_WINDOW_SIZE and ENV_COST to the constructor
        return AlwaysInTradingEnv(test_df, feature_cols, scaler, 
                                  window=ENV_WINDOW_SIZE, cost=ENV_COST, 
                                  start_at_beginning=True)
    
    eval_env = DummyVecEnv([make_eval_env_fn])

    # --- 3. Load Model ---
    print(f"Attempting to load model from {args.model_path} with policy type '{args.policy_type}'...")
    
    custom_objects = {}
    if args.policy_type == "xlstm":
        if MlpXLstmPolicy is None: # Should have been defined above
            print("Error: MlpXLstmPolicy not defined for xlstm type. This is an internal script error.")
            exit(1)
        custom_objects = {"policy_class": MlpXLstmPolicy}
        # For SB3 Contrib RecurrentPPO, when loading a custom policy,
        # it often expects 'policy_kwargs' to be passed if they were used during saving
        # and are essential for policy reconstruction (like lstm_hidden_size for xLSTM).
        # However, SB3 usually saves these. If issues arise, one might need to pass
        # policy_kwargs to .load() if the model was saved without them embedded.
        # For now, assume SB3 handles this well.

    try:
        # For RecurrentPPO, custom_objects might need to specify the policy itself
        # if it was a custom class not automatically discoverable by SB3 by name.
        # If MlpXLstmPolicy is registered with SB3 or if SB3 saves enough info,
        # this might not be strictly needed for 'policy_class' but doesn't hurt.
        model = RecurrentPPO.load(args.model_path, env=eval_env, custom_objects=custom_objects)
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"CRITICAL: Could not load model from {args.model_path}. Error: {e}")
        print("Ensure the model_path is correct and the model was saved with a compatible SB3 version.")
        if args.policy_type == "xlstm":
            print("If it's an xLSTM model, ensure it was saved correctly and policy_kwargs were handled.")
        exit(1)

    # --- 4. Evaluation ---
    print(f"\n--- Evaluating AGENT ({args.policy_type.upper()}) on test set ---")
    obs_eval = eval_env.reset()
    lstm_states_eval = None 
    episode_starts_eval = np.ones((eval_env.num_envs,), dtype=bool)
    
    agent_actions_hist, agent_rewards_hist, agent_pnl_hist, agent_cost_hist = [], [], [], []
    
    num_steps_eval = 0
    max_steps_eval = len(test_df) - ENV_WINDOW_SIZE -1 
    if max_steps_eval <=0:
        print(f"Test data too short for evaluation. Needs at least {ENV_WINDOW_SIZE + 2} days. Found {len(test_df)} days in test set.")
        exit(1)

    done_eval_loop = np.array([False] * eval_env.num_envs) # Ensure correct shape for DummyVecEnv
    while not done_eval_loop[0] and num_steps_eval < max_steps_eval:
        action_pred, lstm_states_eval = model.predict(
            obs_eval,
            state=lstm_states_eval,
            episode_start=episode_starts_eval,
            deterministic=True
        )
        obs_eval, reward_eval, dones_eval_vec, info_eval_vec = eval_env.step(action_pred)
        
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

    # --- 5. Calculate "ALWAYS LONG" BASELINE performance on test set ---
    print("\n--- Calculating ALWAYS LONG (Buy and Hold) BASELINE on test set ---")
    baseline_rewards = []
    baseline_last_action = 1 

    start_idx_new_price = ENV_WINDOW_SIZE # Relative to test_df
    end_idx_new_price = len(test_df) - 2 

    if start_idx_new_price > end_idx_new_price or len(test_df) <= ENV_WINDOW_SIZE :
        print("Test data too short for baseline calculation.")
        total_baseline_reward = 0
        num_baseline_steps = 0
    else:
        num_baseline_steps = 0
        # Ensure test_df has 'Adj Close' and is indexed properly for iloc
        # The AlwaysInTradingEnv resets its internal df index, so test_df here should be the original slice
        temp_test_df_for_baseline = test_df.reset_index(drop=True)

        for current_idx_for_new_price in range(start_idx_new_price, end_idx_new_price + 1):
            if current_idx_for_new_price >= len(temp_test_df_for_baseline):
                print(f"Warning: Baseline calculation trying to access index {current_idx_for_new_price} beyond test_df length {len(temp_test_df_for_baseline)}.")
                break
            old_price = temp_test_df_for_baseline['Adj Close'].iloc[current_idx_for_new_price - 1]
            new_price = temp_test_df_for_baseline['Adj Close'].iloc[current_idx_for_new_price]
            
            held_position = 1 
            pnl = held_position * (new_price - old_price)
            current_baseline_action = 1 
            transaction_cost = ENV_COST * abs(new_price) if current_baseline_action != baseline_last_action else 0
            reward_step = pnl - transaction_cost
            baseline_rewards.append(reward_step)
            baseline_last_action = current_baseline_action # Stays 1 for "always long" after first step
            num_baseline_steps +=1
        total_baseline_reward = sum(baseline_rewards)

    print(f"Baseline evaluation finished after {num_baseline_steps} steps.")
    print(f"Total Baseline (Always Long) Reward: {total_baseline_reward:.2f}")
    print(f"Average Baseline Reward per step: {np.mean(baseline_rewards) if baseline_rewards else 0:.4f}")

    # --- 6. Comparison and Plotting ---
    print("\n--- COMPARISON ---")
    print(f"Agent Total Reward: {total_agent_reward:.2f}")
    print(f"Baseline (Always Long) Total Reward: {total_baseline_reward:.2f}")
    if total_baseline_reward != 0:
        print(f"Agent performance vs Baseline: {(total_agent_reward/total_baseline_reward)*100:.2f}%")
    else:
        if total_agent_reward > 0: print("Agent performance vs Baseline: Agent profitable, Baseline zero.")
        elif total_agent_reward < 0: print("Agent performance vs Baseline: Agent loss, Baseline zero.")
        else: print("Agent performance vs Baseline: Both Agent and Baseline zero.")

    if agent_rewards_hist:
        plt.figure(figsize=(14, 7))
        plt.plot(np.cumsum(agent_rewards_hist), label=f'{args.policy_type.upper()} Agent Cumulative P&L')
        if baseline_rewards:
            plt.plot(np.cumsum(baseline_rewards), label='Baseline (Always Long) Cumulative P&L', linestyle='--')
        plt.title(f'{args.policy_type.upper()} Agent vs Baseline Cumulative P&L on Test Set (Window: {ENV_WINDOW_SIZE}, Cost: {ENV_COST})')
        plt.xlabel('Time Step in Test Set')
        plt.ylabel('Cumulative Reward/P&L')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(14, 4))
        plt.plot(agent_actions_hist,'.', label=f'Agent Actions (0=Short, 1=Long) - {args.policy_type.upper()}')
        plt.title(f'{args.policy_type.upper()} Agent Actions Taken During Test')
        plt.xlabel('Time Step')
        plt.ylabel('Action')
        plt.yticks([0,1])
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No agent rewards recorded for plotting.")

if __name__ == "__main__":
    main() 