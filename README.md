# Pfizer-Trader RL Agent

## Overview
This repository implements a reinforcement learning (RL) agent for trading Pfizer (PFE) stock using an LSTM-based policy. The code is designed for training and evaluating the agent on historical data, with a focus on realistic trading environments and transaction costs.

---

## Repository Contents

- `rl.py` — Main script for training the RL agent using SB3-Contrib's RecurrentPPO with an LSTM policy.
- `evaluate_model.py` — Script to evaluate a trained model on a test set.
- `environment.yml` — Conda environment file listing all dependencies required to run the code.
- `best_model_lstm_pfe/` — Directory containing the best trained model (`best_model.zip`).

---

## Setup

### 1. Create the Conda Environment
Install all dependencies using the provided environment file:
```bash
conda env create -f environment.yml
conda activate pfizer-analysis-env
```

### 2. (Alternative) Install with pip
If you prefer pip, install the main dependencies manually:
```bash
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn torch torchvision gymnasium stable-baselines3 sb3-contrib tensorboard opencv-python
```

---

## Training

To train the RL agent on your data, run:
```bash
python rl.py
```
- The script will load the data (make sure the required CSV is present and referenced in the script), train the agent, and save the best model to `best_model_lstm_pfe/best_model.zip`.

---

## Evaluation

To evaluate a trained model:
```bash
python evaluate_model.py --model_path best_model_lstm_pfe/best_model.zip --policy_type lstm --env_window_size 100 --env_cost 0.0003 --data_path <your_data.csv> --split_date 2010-01-01
```
- Adjust `--data_path` to point to your data file (must match the format expected by the scripts).
- The script will print evaluation results and plot performance.

---

## Model Files
- The best trained model is saved as `best_model_lstm_pfe/best_model.zip`.

---

## Notes
- Make sure your data file (CSV) is present and matches the expected format in `rl.py` and `evaluate_model.py`.
- The scripts expect columns like `Date, Open, High, Low, Close, Adj Close, Volume` and will compute additional features (MA10, MA50, RSI).

---

## License
MIT License (see LICENSE file if present) 