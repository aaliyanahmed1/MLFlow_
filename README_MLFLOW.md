# Minimal MLflow Script

A single, memory-efficient script for model tracking, comparison, and drift detection using MLflow.

## Requirements

```
mlflow>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The script supports three modes of operation:

### 1. Train and Track a Model

```bash
python mlflow_minimal.py --mode train --experiment my_experiment
```

### 2. Compare Model Performance

```bash
python mlflow_minimal.py --mode compare --experiment my_experiment
```

### 3. Check for Model Drift

```bash
python mlflow_minimal.py --mode drift --experiment my_experiment
```

### 4. Run All Operations

```bash
python mlflow_minimal.py --mode all --experiment my_experiment
```

## Additional Options

- `--run-name`: Specify a name for the training run
- `--drift-factor`: Control the amount of simulated drift (0-1, default: 0.5)

## View Results

After running the script, view the results in the MLflow UI:

```bash
mlflow ui
```

Then open http://127.0.0.1:5000 in your browser.

## Quick Start

For convenience, you can use the provided batch file:

```bash
run_minimal.bat
```

This will set up a virtual environment, install dependencies, run the script in 'all' mode, and launch the MLflow UI.