# Pfizer-Trader: LSTM/xLSTM Reinforcement Learning for Stock Trading

## Overview
This repository implements a reinforcement learning (RL) agent for trading Pfizer (PFE) stock using LSTM and xLSTM-based policies. The agent is trained and evaluated on historical PFE data, with a focus on realistic trading environments and transaction costs.

- **RL Training**: `rl.py` (standard LSTM) and optionally `rl_xlstm.py` (xLSTM variant, if present)
- **Evaluation**: `evaluate_model.py` (supports both LSTM and xLSTM models)
- **CLI Tool**: `main_cli.py` for interactive training and evaluation
- **Model**: Trained model weights (e.g., `ppo_lstm_trading_pfe_final.zip`)
- **Data**: `PFE.csv` (Pfizer historical stock data)

---

## Project Structure

- `rl.py` — Main script for training an RL agent with LSTM policy using SB3-Contrib's RecurrentPPO.
- `evaluate_model.py` — Script to evaluate a trained model (LSTM or xLSTM) on the test set.
- `main_cli.py` — Command-line interface for training and evaluation workflows.
- `ppo_lstm_trading_pfe_final.zip` — Example of a trained model file.
- `PFE.csv` — Historical stock data for Pfizer (used for training/testing).
- `requirements.txt` — Python dependencies.
- `best_model_lstm_pfe_1M/`, `logs_lstm_pfe_1M/`, etc. — Model checkpoints and logs.

---

## Data
- **File**: `PFE.csv`
- **Columns**: `Date, Open, High, Low, Close, Adj Close, Volume`
- **Usage**: The RL environment uses rolling features (MA10, MA50, RSI) computed from this data.

---

## Training

### Standard LSTM Policy
To train an LSTM-based RL agent:
```bash
python rl.py
```
- The script will train a RecurrentPPO agent with an LSTM policy on the training portion of the data.
- Model checkpoints and logs are saved in `best_model_lstm_pfe_1M/` and `logs_lstm_pfe_1M/`.
- Final model is saved as `ppo_lstm_trading_pfe_final_1M.zip`.

### xLSTM Policy (if available)
If you have `rl_xlstm.py`, you can train an xLSTM variant:
```bash
python rl_xlstm.py
```

---

## Evaluation

To evaluate a trained model (LSTM or xLSTM):
```bash
python evaluate_model.py --model_path <path_to_model.zip> --policy_type <lstm|xlstm> --env_window_size 100 --env_cost 0.0003 --data_path PFE.csv --split_date 2010-01-01
```
- `--model_path`: Path to the trained model zip file
- `--policy_type`: `lstm` or `xlstm`
- `--env_window_size`: Observation window size (default: 100)
- `--env_cost`: Transaction cost per trade (default: 0.0003)
- `--data_path`: Path to the data CSV (default: PFE.csv)
- `--split_date`: Train/test split date (default: 2010-01-01)

---

## Command-Line Interface (CLI)

You can use the interactive CLI for training and evaluation:
```bash
python main_cli.py
```
- Choose to train a new model or evaluate an existing one.
- The CLI will guide you through the required parameters and workflow.

---

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```
**Note:** You will also need:
- `stable-baselines3`
- `sb3-contrib`
- `gymnasium`

Install with:
```bash
pip install stable-baselines3 sb3-contrib gymnasium
```

---

## Model Files
- Trained models are saved as `.zip` files (e.g., `ppo_lstm_trading_pfe_final.zip`).
- Best models during training are saved in `best_model_lstm_pfe_1M/`.

---

## Citation
If you use this codebase, please cite or reference this repository.

---

## License
MIT License (see LICENSE file if present) 