# Minimal MLflow Script for Model Tracking, Comparison and Drift Detection
# Combines essential functionality in a memory-efficient implementation

import argparse
import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt


def train_and_track(experiment_name="model_tracking", run_name="training_run"):
    """Train a model and track metrics with MLflow."""
    # Set tracking URI and experiment
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)
    
    # Load sample data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        params = {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 5}
        mlflow.log_params(params)
        
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Calculate metrics
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_test)
        
        metrics = {
            "train_accuracy": accuracy_score(y_train, train_preds),
            "val_accuracy": accuracy_score(y_test, val_preds),
            "val_f1": f1_score(y_test, val_preds, average='macro')
        }
        
        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        print(f"Model trained and logged to MLflow")
        
        return model, run_id, X_test, y_test


def compare_models(experiment_name="model_tracking", metric_name="val_accuracy", max_runs=5):
    """Compare performance of different model runs."""
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Get experiment
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return None
    
    # Get runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_runs,
        order_by=[f"metrics.{metric_name} DESC"]
    )
    
    if not runs:
        print(f"No runs found for experiment '{experiment_name}'.")
        return None
    
    # Extract run data
    run_data = []
    for run in runs:
        run_info = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", "Unknown"),
            "model_type": run.data.params.get("model_type", "Unknown")
        }
        
        # Add metrics
        for metric in ["val_accuracy", "val_f1"]:
            if metric in run.data.metrics:
                run_info[metric] = run.data.metrics[metric]
        
        run_data.append(run_info)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(run_data)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Get best model
    best_run = runs[0]
    print(f"\nBest Model (based on {metric_name}):")
    print(f"Run ID: {best_run.info.run_id}")
    print(f"{metric_name}: {best_run.data.metrics.get(metric_name, 'Unknown')}")
    
    return best_run.info.run_id


def check_drift(run_id, drift_factor=0.5, experiment_name="model_drift"):
    """Check for model drift using a production model."""
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)
    
    # Load model
    try:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        print(f"Loaded model from run ID: {run_id}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Generate data
    X, y = load_iris(return_X_y=True)
    X_test, _, y_test, _ = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Create drifted dataset
    noise = np.random.normal(0, drift_factor, X_test.shape)
    X_drifted = X_test + noise
    
    # Start MLflow run
    with mlflow.start_run(run_name="drift_check"):
        # Calculate metrics on original data
        original_preds = model.predict(X_test)
        original_acc = accuracy_score(y_test, original_preds)
        original_f1 = f1_score(y_test, original_preds, average='macro')
        
        # Calculate metrics on drifted data
        drifted_preds = model.predict(X_drifted)
        drifted_acc = accuracy_score(y_test, drifted_preds)
        drifted_f1 = f1_score(y_test, drifted_preds, average='macro')
        
        # Calculate drift
        acc_drift = original_acc - drifted_acc
        f1_drift = original_f1 - drifted_f1
        
        # Log metrics
        mlflow.log_metrics({
            "original_accuracy": original_acc,
            "original_f1": original_f1,
            "drifted_accuracy": drifted_acc,
            "drifted_f1": drifted_f1,
            "accuracy_drift": acc_drift,
            "f1_drift": f1_drift
        })
        
        # Create visualization
        plt.figure(figsize=(8, 4))
        metrics = ["Accuracy", "F1 Score"]
        original_values = [original_acc, original_f1]
        drifted_values = [drifted_acc, drifted_f1]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, original_values, width, label='Original')
        plt.bar(x + width/2, drifted_values, width, label='Drifted')
        plt.title('Model Performance: Original vs. Drifted Data')
        plt.xticks(x, metrics)
        plt.ylim(0, 1.0)
        plt.legend()
        
        # Save and log figure
        plt.savefig("drift_analysis.png")
        mlflow.log_artifact("drift_analysis.png")
        
        # Determine if drift is significant
        is_significant = acc_drift > 0.1  # 10% threshold
        mlflow.log_param("significant_drift", is_significant)
        
        print(f"\nDrift Analysis:")
        print(f"Original - Accuracy: {original_acc:.4f}, F1: {original_f1:.4f}")
        print(f"Drifted  - Accuracy: {drifted_acc:.4f}, F1: {drifted_f1:.4f}")
        print(f"Drift    - Accuracy: {acc_drift:.4f}, F1: {f1_drift:.4f}")
        print(f"Significant Drift: {is_significant}")
        
        if is_significant:
            print("WARNING: Significant model drift detected!")


def main():
    parser = argparse.ArgumentParser(description="Minimal MLflow script for model tracking and evaluation")
    parser.add_argument("--mode", type=str, default="all", choices=["train", "compare", "drift", "all"],
                        help="Operation mode: train, compare, drift, or all")
    parser.add_argument("--experiment", type=str, default="minimal_mlflow",
                        help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for the training run")
    parser.add_argument("--drift-factor", type=float, default=0.5,
                        help="Factor to control the amount of simulated drift (0-1)")
    args = parser.parse_args()
    
    run_id = None
    
    if args.mode in ["train", "all"]:
        print("\n=== Training Model and Tracking Metrics ===")
        _, run_id, _, _ = train_and_track(args.experiment, args.run_name)
    
    if args.mode in ["compare", "all"]:
        print("\n=== Comparing Model Performance ===")
        best_run_id = compare_models(args.experiment)
        if best_run_id and args.mode == "compare":
            run_id = best_run_id
    
    if args.mode in ["drift", "all"]:
        print("\n=== Checking for Model Drift ===")
        if run_id:
            check_drift(run_id, args.drift_factor, args.experiment + "_drift")
        else:
            print("No run ID available for drift checking. Please train a model first.")
    
    print("\nTo view results in MLflow UI, run: mlflow ui")


if __name__ == "__main__":
    main()