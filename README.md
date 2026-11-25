# AI/ML Network Traffic Analysis

A complete, runnable project that classifies network flows as **Normal, DDoS, PortScan, BruteForce or Botnet**
using machine learning, and flags unusual flows with an **Isolation Forest** anomaly detector.
Includes a Streamlit dashboard and an optional live packet-capture module.

## Project structure

```
network-traffic-analysis/
├── app.py                 # Streamlit dashboard
├── run_all.py             # one-command pipeline (data -> train -> predict)
├── requirements.txt
├── src/
│   ├── config.py          # paths, feature list, class names
│   ├── generate_data.py   # synthetic flow generator (CICIDS-style features)
│   ├── train.py           # trains RandomForest / GradientBoosting / Neural Net + Isolation Forest
│   ├── detector.py        # inference wrapper (used by everything else)
│   ├── predict.py         # classify a CSV of flows
│   └── live_capture.py    # sniff real packets -> flows -> predictions (optional)
├── data/                  # generated dataset goes here
├── models/                # trained models are saved here
└── outputs/               # metrics.json, plots, predictions.csv
```

## Setup in VS Code

1. Install Python 3.9+ and the VS Code **Python** extension.
2. Open the `network-traffic-analysis` folder in VS Code (File > Open Folder).
3. Open a terminal (Ctrl+`) and create a virtual environment:

   **Windows**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   **macOS / Linux**
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Select the interpreter: Ctrl+Shift+P > "Python: Select Interpreter" > pick the `venv` one.

## Run it

```
python run_all.py          # generates data, trains models, makes sample predictions
streamlit run app.py       # opens the dashboard in your browser
```

Or run the steps individually:

```
python src/generate_data.py --samples 30000
python src/train.py
python src/predict.py                      # simulated sample
python src/predict.py --input my_flows.csv # your own CSV
```

You can also use the Run and Debug panel (F5) — launch configs for each step are in `.vscode/launch.json`.

## What you get

* `outputs/metrics.json` — accuracy, precision, recall, F1 for every model
* `outputs/confusion_matrix.png`, `feature_importance.png`, `model_comparison.png`, `class_distribution.png`
* `outputs/predictions.csv` — per-flow prediction, confidence, anomaly score
* Dashboard tabs: simulate traffic, upload CSV, inspect a single flow, model performance

## Use real data (CICIDS2017 / UNSW-NB15 etc.)

1. Download a flow dataset and map its columns to the names in `FEATURES` (`src/config.py`),
   or edit `FEATURES` / `CLASSES` to match the dataset's own columns and labels.
2. Save it as `data/traffic_flows.csv` with a `label` column.
3. Run `python src/train.py`. Everything else (dashboard, predict) adapts automatically.

## Live capture (optional)

```
# Linux / macOS
sudo python src/live_capture.py --iface eth0 --duration 60
# Windows: install Npcap, run the terminal as Administrator
python src/live_capture.py --duration 60
```

It groups packets into flows, computes the same features, and runs the trained model.
Note: models trained on the synthetic data are a demo of the pipeline — for real networks,
train on real labelled data. Only capture traffic on networks you own or are authorised to monitor.

## How it works

1. **Features**: per-flow statistics (duration, packet/byte counts, rates, inter-arrival times, TCP flags).
2. **Supervised classification**: three models are compared on a held-out test set; the best by macro-F1 is saved.
3. **Anomaly detection**: an Isolation Forest trained only on normal traffic scores how unusual each flow is,
   which can catch attacks the classifier was never trained on.
4. **Dashboard**: runs both models on simulated or uploaded flows and visualises results.

## Ideas to extend

* Add more attack classes (SQL injection, DNS tunnelling) in `generate_data.py`
* Add an LSTM / autoencoder for sequence-based detection
* Stream live capture into the dashboard
* Add SHAP explanations for each alert
