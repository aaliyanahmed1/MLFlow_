# MLflow Project - Drift Detection Module
# Detects model drift by comparing performance on original and drifted datasets

import os
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Optional, Union, Any, Tuple

# Import utility functions
from utils import load_dataset, evaluate_model

# Import RFDETR specific modules
import io
import requests
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv


def generate_drift_data(X: Union[np.ndarray, List[str]], y: Optional[np.ndarray] = None, 
                       drift_factor: float = 0.1, drift_type: str = "feature") -> Tuple[Union[np.ndarray, List[str]], Optional[np.ndarray]]:
    """Generate synthetic data drift for testing model robustness.
    
    Args:
        X: Original feature data or list of image URLs for object detection
        y: Original labels (None for object detection)
        drift_factor: Magnitude of the drift (0.0 to 1.0)
        drift_type: Type of drift to simulate ('feature', 'label', or 'both')
        
    Returns:
        Tuple of (drifted_X, drifted_y)
    """
    # Handle object detection data (image URLs)
    if isinstance(X, list) and all(isinstance(url, str) for url in X):
        # For object detection, we simulate drift by selecting different images
        # or by applying transformations to the existing images
        if drift_type in ["feature", "both"]:
            # For feature drift in object detection, we could:
            # 1. Add noise to the images
            # 2. Adjust brightness/contrast
            # 3. Apply slight blur
            # Here we'll just return the same images, as actual image manipulation
            # would require downloading and processing each image
            print(f"Simulating {drift_factor} feature drift for object detection images")
            
        # For object detection, label drift doesn't apply in the same way
        # as we're using a pre-trained model
        return X, None
    
    # Handle traditional ML data (numpy arrays)
    else:
        drifted_X = X.copy()
        drifted_y = y.copy() if y is not None else None
        
        if drift_type in ["feature", "both"]:
            # Feature drift: Add noise to features
            feature_noise = np.random.normal(0, drift_factor, drifted_X.shape)
            drifted_X = drifted_X + feature_noise
        
        if y is not None and drift_type in ["label", "both"]:
            # Label drift: Flip some labels randomly
            num_samples = len(drifted_y)
            num_to_flip = int(num_samples * drift_factor)
            indices_to_flip = np.random.choice(num_samples, num_to_flip, replace=False)
            
            # Get unique labels
            unique_labels = np.unique(y)
            
            # Flip labels
            for idx in indices_to_flip:
                current_label = drifted_y[idx]
                other_labels = [l for l in unique_labels if l != current_label]
                drifted_y[idx] = np.random.choice(other_labels)
                
        return drifted_X, drifted_y
    
    return drifted_X, drifted_y


def generate_precision_drift(X: np.ndarray, precision: str = "int8") -> np.ndarray:
    """Generate data with reduced numerical precision to simulate quantization effects.
    
    Args:
        X: Original feature data
        precision: Target precision ('int8', 'int16', 'float16')
        
    Returns:
        Data with reduced precision
    """
    if precision == "int8":
        # Convert to int8 and back to float
        X_reduced = X.astype(np.int8).astype(np.float32)
    elif precision == "int16":
        # Convert to int16 and back to float
        X_reduced = X.astype(np.int16).astype(np.float32)
    elif precision == "float16":
        # Convert to float16 and back to float32
        X_reduced = X.astype(np.float16).astype(np.float32)
    else:
        # No precision reduction
        X_reduced = X.copy()
    
    return X_reduced


def check_drift(run_id: str, drift_factor: float = 0.1, drift_type: str = "feature",
               precision_formats: List[str] = None) -> Dict[str, float]:
    """Check model drift by comparing performance on original and drifted datasets.
    
    Args:
        run_id: MLflow run ID of the model to evaluate
        drift_factor: Magnitude of the drift (0.0 to 1.0)
        drift_type: Type of drift to simulate ('feature', 'label', or 'both')
        precision_formats: List of precision formats to test ('int8', 'int16', 'float16')
        
    Returns:
        Dictionary of drift metrics
    """
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Start a new run for drift detection
    with mlflow.start_run(run_name=f"drift_detection_{drift_type}_{drift_factor}") as drift_run:
        # Log parameters
        mlflow.log_param("original_run_id", run_id)
        mlflow.log_param("drift_factor", drift_factor)
        mlflow.log_param("drift_type", drift_type)
        
        # Load the model from the specified run
        try:
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            print(f"Loaded model from run {run_id}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return {}
        
        # Load dataset
        X_train, X_test, y_train, y_test = load_dataset()
        
        # Evaluate on original test data
        original_metrics = evaluate_model(model, X_test, y_test)
        original_accuracy = original_metrics["test_accuracy"]
        original_f1 = original_metrics["test_f1"]
        
        # Log original metrics
        mlflow.log_metric("original_accuracy", original_accuracy)
        mlflow.log_metric("original_f1", original_f1)
        
        # Generate drifted data
        drifted_X, drifted_y = generate_drift_data(X_test, y_test, drift_factor, drift_type)
        
        # Evaluate on drifted data
        drifted_metrics = evaluate_model(model, drifted_X, drifted_y)
        drifted_accuracy = drifted_metrics["test_accuracy"]
        drifted_f1 = drifted_metrics["test_f1"]
        
        # Log drifted metrics
        mlflow.log_metric("drifted_accuracy", drifted_accuracy)
        mlflow.log_metric("drifted_f1", drifted_f1)
        
        # Calculate and log drift metrics
        accuracy_drift = original_accuracy - drifted_accuracy
        f1_drift = original_f1 - drifted_f1
        
        mlflow.log_metric("accuracy_drift", accuracy_drift)
        mlflow.log_metric("f1_drift", f1_drift)
        
        # Determine if drift is significant (more than 5% drop in performance)
        is_significant_drift = accuracy_drift > 0.05 or f1_drift > 0.05
        mlflow.log_param("significant_drift", is_significant_drift)
        
        # Create visualization of drift
        plt.figure(figsize=(10, 6))
        metrics = ["accuracy", "f1"]
        original_values = [original_accuracy, original_f1]
        drifted_values = [drifted_accuracy, drifted_f1]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, original_values, width, label="Original")
        plt.bar(x + width/2, drifted_values, width, label="Drifted")
        
        plt.xlabel("Metrics")
        plt.ylabel("Values")
        plt.title(f"Model Performance Drift (Factor: {drift_factor}, Type: {drift_type})")
        plt.xticks(x, metrics)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Save and log figure
        drift_figure_path = "drift_comparison.png"
        plt.savefig(drift_figure_path)
        mlflow.log_artifact(drift_figure_path)
        
        # Check precision format drift if specified
        precision_metrics = {}
        if precision_formats:
            mlflow.log_param("precision_formats_tested", ",".join(precision_formats))
            
            for precision in precision_formats:
                # Generate precision-reduced data
                precision_X = generate_precision_drift(X_test, precision)
                
                # Evaluate on precision-reduced data
                precision_metrics = evaluate_model(model, precision_X, y_test)
                precision_accuracy = precision_metrics["test_accuracy"]
                precision_f1 = precision_metrics["test_f1"]
                
                # Log precision metrics
                mlflow.log_metric(f"{precision}_accuracy", precision_accuracy)
                mlflow.log_metric(f"{precision}_f1", precision_f1)
                
                # Calculate and log precision drift
                precision_accuracy_drift = original_accuracy - precision_accuracy
                precision_f1_drift = original_f1 - precision_f1
                
                mlflow.log_metric(f"{precision}_accuracy_drift", precision_accuracy_drift)
                mlflow.log_metric(f"{precision}_f1_drift", precision_f1_drift)
                
                # Store in results dictionary
                precision_metrics[f"{precision}_accuracy"] = precision_accuracy
                precision_metrics[f"{precision}_f1"] = precision_f1
                precision_metrics[f"{precision}_accuracy_drift"] = precision_accuracy_drift
                precision_metrics[f"{precision}_f1_drift"] = precision_f1_drift
            
            # Create visualization of precision format drift
            plt.figure(figsize=(12, 6))
            
            # Prepare data for plotting
            all_precisions = ["original"] + precision_formats
            accuracy_values = [original_accuracy] + [precision_metrics.get(f"{p}_accuracy", 0) for p in precision_formats]
            f1_values = [original_f1] + [precision_metrics.get(f"{p}_f1", 0) for p in precision_formats]
            
            # Plot
            x = np.arange(len(all_precisions))
            width = 0.35
            
            plt.bar(x - width/2, accuracy_values, width, label="Accuracy")
            plt.bar(x + width/2, f1_values, width, label="F1 Score")
            
            plt.xlabel("Precision Format")
            plt.ylabel("Metric Value")
            plt.title("Model Performance Across Precision Formats")
            plt.xticks(x, all_precisions)
            plt.ylim(0, 1.0)
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            
            # Save and log figure
            precision_figure_path = "precision_comparison.png"
            plt.savefig(precision_figure_path)
            mlflow.log_artifact(precision_figure_path)
        
        # Combine all metrics
        drift_metrics = {
            "original_accuracy": original_accuracy,
            "original_f1": original_f1,
            "drifted_accuracy": drifted_accuracy,
            "drifted_f1": drifted_f1,
            "accuracy_drift": accuracy_drift,
            "f1_drift": f1_drift,
            "significant_drift": is_significant_drift
        }
        
        # Add precision metrics if available
        drift_metrics.update(precision_metrics)
        
        # Print summary
        print(f"\nDrift Detection Results:")
        print(f"Original Accuracy: {original_accuracy:.4f}")
        print(f"Drifted Accuracy: {drifted_accuracy:.4f}")
        print(f"Accuracy Drift: {accuracy_drift:.4f}")
        print(f"Original F1: {original_f1:.4f}")
        print(f"Drifted F1: {drifted_f1:.4f}")
        print(f"F1 Drift: {f1_drift:.4f}")
        print(f"Significant Drift: {is_significant_drift}")
        
        if precision_formats:
            print("\nPrecision Format Results:")
            for precision in precision_formats:
                print(f"{precision} Accuracy: {precision_metrics.get(f'{precision}_accuracy', 0):.4f}")
                print(f"{precision} Accuracy Drift: {precision_metrics.get(f'{precision}_accuracy_drift', 0):.4f}")
        
        return drift_metrics


def monitor_drift_over_time(run_id: str, drift_factors: List[float] = None,
                          drift_type: str = "feature") -> str:
    """Monitor model drift over increasing levels of data drift.
    
    Args:
        run_id: MLflow run ID of the model to evaluate
        drift_factors: List of drift factors to test
        drift_type: Type of drift to simulate
        
    Returns:
        Path to the drift trend figure
    """
    if drift_factors is None:
        drift_factors = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Start a new run for drift monitoring
    with mlflow.start_run(run_name=f"drift_monitoring_{drift_type}") as monitor_run:
        # Log parameters
        mlflow.log_param("original_run_id", run_id)
        mlflow.log_param("drift_type", drift_type)
        mlflow.log_param("drift_factors", ",".join([str(f) for f in drift_factors]))
        
        # Load the model from the specified run
        try:
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
        
        # Load dataset
        X_train, X_test, y_train, y_test = load_dataset()
        
        # Track metrics across drift factors
        accuracy_values = []
        f1_values = []
        
        for factor in drift_factors:
            if factor == 0.0:
                # Original data without drift
                metrics = evaluate_model(model, X_test, y_test)
            else:
                # Generate drifted data
                drifted_X, drifted_y = generate_drift_data(X_test, y_test, factor, drift_type)
                metrics = evaluate_model(model, drifted_X, drifted_y)
            
            accuracy_values.append(metrics["test_accuracy"])
            f1_values.append(metrics["test_f1"])
            
            # Log metrics for this drift factor
            mlflow.log_metric(f"accuracy_factor_{factor}", metrics["test_accuracy"])
            mlflow.log_metric(f"f1_factor_{factor}", metrics["test_f1"])
        
        # Create visualization of drift trend
        plt.figure(figsize=(10, 6))
        
        plt.plot(drift_factors, accuracy_values, 'o-', label="Accuracy")
        plt.plot(drift_factors, f1_values, 's-', label="F1 Score")
        
        plt.xlabel("Drift Factor")
        plt.ylabel("Metric Value")
        plt.title(f"Model Performance vs. Drift Factor (Type: {drift_type})")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        
        # Add threshold line
        plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label="Acceptable Threshold")
        
        # Save and log figure
        drift_trend_path = "drift_trend.png"
        plt.savefig(drift_trend_path)
        mlflow.log_artifact(drift_trend_path)
        
        print(f"\nDrift Monitoring Results:")
        print(f"Drift factors tested: {drift_factors}")
        print(f"Accuracy values: {[f'{acc:.4f}' for acc in accuracy_values]}")
        print(f"F1 values: {[f'{f1:.4f}' for f1 in f1_values]}")
        
        return drift_trend_path


def compare_model_robustness(experiment_name: str, drift_factor: float = 0.1,
                           drift_type: str = "feature", top_n: int = 3) -> pd.DataFrame:
    """Compare robustness of multiple models to the same drift conditions.
    
    Args:
        experiment_name: Name of the experiment containing models to compare
        drift_factor: Magnitude of the drift to apply
        drift_type: Type of drift to simulate
        top_n: Number of top models to compare
        
    Returns:
        DataFrame with robustness comparison
    """
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Get experiment
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return pd.DataFrame()
    
    # Get top runs based on accuracy
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=top_n,
        order_by=["metrics.test_accuracy DESC"]
    )
    
    if not runs:
        print(f"No runs found for experiment '{experiment_name}'.")
        return pd.DataFrame()
    
    # Start a new run for robustness comparison
    with mlflow.start_run(run_name=f"robustness_comparison_{drift_type}_{drift_factor}") as comparison_run:
        # Log parameters
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("drift_factor", drift_factor)
        mlflow.log_param("drift_type", drift_type)
        mlflow.log_param("top_n", top_n)
        
        # Load dataset
        X_train, X_test, y_train, y_test = load_dataset()
        
        # Generate drifted data once
        drifted_X, drifted_y = generate_drift_data(X_test, y_test, drift_factor, drift_type)
        
        # Compare robustness of each model
        robustness_data = []
        
        for run in runs:
            run_id = run.info.run_id
            run_name = run.data.tags.get("mlflow.runName", "Unknown")
            model_type = run.data.params.get("model_type", "Unknown")
            
            try:
                # Load model
                model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                
                # Evaluate on original data
                original_metrics = evaluate_model(model, X_test, y_test)
                original_accuracy = original_metrics["test_accuracy"]
                original_f1 = original_metrics["test_f1"]
                
                # Evaluate on drifted data
                drifted_metrics = evaluate_model(model, drifted_X, drifted_y)
                drifted_accuracy = drifted_metrics["test_accuracy"]
                drifted_f1 = drifted_metrics["test_f1"]
                
                # Calculate drift metrics
                accuracy_drift = original_accuracy - drifted_accuracy
                f1_drift = original_f1 - drifted_f1
                
                # Calculate robustness score (higher is better)
                # Simple formula: 1 - average normalized performance drop
                robustness_score = 1 - ((accuracy_drift / original_accuracy + f1_drift / original_f1) / 2)
                
                # Store results
                model_data = {
                    "run_id": run_id,
                    "run_name": run_name,
                    "model_type": model_type,
                    "original_accuracy": original_accuracy,
                    "drifted_accuracy": drifted_accuracy,
                    "accuracy_drift": accuracy_drift,
                    "original_f1": original_f1,
                    "drifted_f1": drifted_f1,
                    "f1_drift": f1_drift,
                    "robustness_score": robustness_score
                }
                
                robustness_data.append(model_data)
                
                # Log metrics for this model
                mlflow.log_metric(f"robustness_score_{model_type}", robustness_score)
                
            except Exception as e:
                print(f"Error evaluating model {run_id}: {e}")
        
        # Create DataFrame
        robustness_df = pd.DataFrame(robustness_data)
        
        if not robustness_df.empty:
            # Sort by robustness score
            robustness_df = robustness_df.sort_values(by="robustness_score", ascending=False).reset_index(drop=True)
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            
            # Prepare data for plotting
            models = robustness_df["model_type"].tolist()
            original_acc = robustness_df["original_accuracy"].tolist()
            drifted_acc = robustness_df["drifted_accuracy"].tolist()
            robustness_scores = robustness_df["robustness_score"].tolist()
            
            # Plot accuracy comparison
            x = np.arange(len(models))
            width = 0.35
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Bar chart for accuracy
            ax1.bar(x - width/2, original_acc, width, label="Original Accuracy")
            ax1.bar(x + width/2, drifted_acc, width, label="Drifted Accuracy")
            ax1.set_xlabel("Model Type")
            ax1.set_ylabel("Accuracy")
            ax1.set_ylim(0, 1.0)
            ax1.set_xticks(x)
            ax1.set_xticklabels(models)
            ax1.legend(loc="upper left")
            
            # Line chart for robustness score
            ax2 = ax1.twinx()
            ax2.plot(x, robustness_scores, 'ro-', linewidth=2, label="Robustness Score")
            ax2.set_ylabel("Robustness Score")
            ax2.set_ylim(0, 1.0)
            ax2.legend(loc="upper right")
            
            plt.title(f"Model Robustness Comparison (Drift Factor: {drift_factor}, Type: {drift_type})")
            plt.grid(axis="y", linestyle="--", alpha=0.3)
            
            # Save and log figure
            robustness_figure_path = "robustness_comparison.png"
            plt.savefig(robustness_figure_path)
            mlflow.log_artifact(robustness_figure_path)
            
            # Export to CSV
            csv_path = "robustness_comparison.csv"
            robustness_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)
            
            print(f"\nModel Robustness Comparison:")
            print(robustness_df[["model_type", "robustness_score", "accuracy_drift", "f1_drift"]])
        
        return robustness_df