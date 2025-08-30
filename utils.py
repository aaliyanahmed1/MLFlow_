# MLflow Project - Common Utility Functions
# Contains shared functionality for data loading, model evaluation, and optimization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import mlflow
import torch
import time
import os
import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
from typing import Dict, Tuple, List, Any, Optional, Union

# Dictionary of available datasets
AVAILABLE_DATASETS = {
    "iris": load_iris,
    "digits": load_digits,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "coco": None  # COCO dataset for object detection with RFDETR
}


def load_dataset(dataset_name: str = "coco", test_size: float = 0.2, random_state: int = 42, image_urls: List[str] = None) -> Tuple:
    """Load and split a dataset for model training and evaluation.
    
    Args:
        dataset_name: Name of the dataset to load (iris, digits, wine, breast_cancer, coco)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        image_urls: List of image URLs for COCO dataset (only used if dataset_name is 'coco')
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not available. Choose from: {list(AVAILABLE_DATASETS.keys())}")
    
    if dataset_name == "coco":
        # For COCO dataset, we use image URLs
        if image_urls is None:
            # Default image URLs if none provided
            image_urls = [
                "https://media.roboflow.com/notebooks/examples/dog-2.jpeg",
                "https://media.roboflow.com/notebooks/examples/dog.jpeg",
                "https://media.roboflow.com/notebooks/examples/pexels-photo-1108099.jpeg",
                "https://media.roboflow.com/notebooks/examples/bus.jpg",
                "https://media.roboflow.com/notebooks/examples/car-road.jpg"
            ]
        
        # Load images
        images = []
        for url in image_urls:
            try:
                img = Image.open(io.BytesIO(requests.get(url).content))
                images.append(img)
            except Exception as e:
                print(f"Error loading image from {url}: {e}")
        
        # Split into train and test
        n_test = max(1, int(len(images) * test_size))
        X_train = images[:-n_test]
        X_test = images[-n_test:]
        
        # For RFDETR, we don't have explicit labels as y_train and y_test
        # Instead, we'll use None as placeholders
        y_train = None
        y_test = None
        
        print(f"Loaded COCO dataset: {len(images)} images")
        print(f"Training set: {len(X_train)} images, Test set: {len(X_test)} images")
    else:
        # Load traditional dataset
        data_loader = AVAILABLE_DATASETS[dataset_name]
        X, y = data_loader(return_X_y=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        print(f"Loaded {dataset_name} dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def evaluate_model(model: Any, X_test: Union[np.ndarray, List[Image.Image]], y_test: Optional[np.ndarray] = None, prefix: str = "") -> Dict[str, float]:
    """Evaluate a model and return performance metrics.
    
    Args:
        model: Trained model with predict method
        X_test: Test features or images
        y_test: Test labels (optional, not used for RFDETR)
        prefix: Optional prefix for metric names
        
    Returns:
        Dictionary of metrics
    """
    # Check if model is RFDETR
    if isinstance(model, RFDETRBase):
        # For RFDETR, evaluate object detection metrics
        metrics = {}
        total_detections = 0
        avg_confidence = 0.0
        
        for i, img in enumerate(X_test):
            # Run inference
            detections = model.predict(img)
            
            # Count detections and calculate average confidence
            num_detections = len(detections)
            total_detections += num_detections
            
            if num_detections > 0:
                avg_confidence += sum(detections.confidence)
            
            # Log individual image metrics
            metrics[f"{prefix}detections_image_{i}"] = num_detections
        
        # Calculate average metrics
        metrics[f"{prefix}avg_detections"] = total_detections / len(X_test) if len(X_test) > 0 else 0
        metrics[f"{prefix}avg_confidence"] = avg_confidence / total_detections if total_detections > 0 else 0
        
        return metrics
    else:
        # For traditional models
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            f"{prefix}accuracy": accuracy_score(y_test, y_pred),
            f"{prefix}f1": f1_score(y_test, y_pred, average='macro'),
            f"{prefix}precision": precision_score(y_test, y_pred, average='macro'),
            f"{prefix}recall": recall_score(y_test, y_pred, average='macro')
        }
        
        return metrics


def log_model_metrics(metrics: Dict[str, float]) -> None:
    """Log metrics to MLflow.
    
    Args:
        metrics: Dictionary of metrics to log
    """
    for name, value in metrics.items():
        mlflow.log_metric(name, value)


def measure_inference_time(model: Any, X: Union[np.ndarray, List[Image.Image]], n_runs: int = 10) -> Dict[str, float]:
    """Measure inference time for a model.
    
    Args:
        model: Trained model with predict method
        X: Input data for inference (numpy array or list of images)
        n_runs: Number of runs to average over
        
    Returns:
        Dictionary with timing metrics
    """
    # Check if model is RFDETR
    if isinstance(model, RFDETRBase):
        # Warm-up run for RFDETR
        if len(X) > 0:
            model.predict(X[0])
    else:
        # Warm-up run for traditional models
        model.predict(X[:10])
    
    # Measure batch inference time
    start_time = time.time()
    for _ in range(n_runs):
        model.predict(X)
    end_time = time.time()
    batch_time = (end_time - start_time) / n_runs
    
    # Measure per-sample inference time
    start_time = time.time()
    for i in range(n_runs):
        for j in range(min(100, len(X))):
            model.predict(X[j:j+1])
    end_time = time.time()
    sample_time = (end_time - start_time) / (n_runs * min(100, len(X)))
    
    return {
        "batch_inference_time": batch_time,
        "sample_inference_time": sample_time,
        "samples_per_second": 1.0 / sample_time if sample_time > 0 else 0
    }


def apply_quantization(model: Any, precision: str = "int8") -> Any:
    """Apply quantization to a model (if supported).
    
    Args:
        model: Model to quantize
        precision: Precision format (int8, int16, float16)
        
    Returns:
        Quantized model or original model if quantization not supported
    """
    # This is a placeholder implementation
    # Actual implementation would depend on the model type and framework
    if hasattr(model, "_sklearn_is_fitted"):
        # For scikit-learn models, we'll just return the original model
        # with a note that quantization is simulated
        print(f"Note: Quantization to {precision} is simulated for scikit-learn models")
        return model
    
    if isinstance(model, torch.nn.Module):
        # For PyTorch models, we can use torch quantization
        try:
            import torch.quantization
            
            # Basic static quantization example
            if precision == "int8":
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                return quantized_model
            elif precision == "float16":
                # Convert to float16
                model.half()
                return model
        except (ImportError, AttributeError):
            print("PyTorch quantization not available")
    
    print(f"Quantization to {precision} not implemented for this model type")
    return model


def apply_pruning(model: Any, pruning_level: float = 0.5) -> Any:
    """Apply pruning to a model (if supported).
    
    Args:
        model: Model to prune
        pruning_level: Level of pruning (0.0 to 1.0)
        
    Returns:
        Pruned model or original model if pruning not supported
    """
    # This is a placeholder implementation
    # Actual implementation would depend on the model type and framework
    if hasattr(model, "_sklearn_is_fitted"):
        # For scikit-learn models, we'll just return the original model
        # with a note that pruning is simulated
        print(f"Note: Pruning at level {pruning_level} is simulated for scikit-learn models")
        return model
    
    if isinstance(model, torch.nn.Module):
        # For PyTorch models, we can use torch pruning
        try:
            import torch.nn.utils.prune as prune
            
            # Apply pruning to all linear layers
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=pruning_level)
            
            return model
        except (ImportError, AttributeError):
            print("PyTorch pruning not available")
    
    print(f"Pruning not implemented for this model type")
    return model


def plot_metrics_comparison(metrics_list: List[Dict[str, float]], labels: List[str], 
                           title: str = "Model Performance Comparison") -> str:
    """Plot comparison of metrics across different models or versions.
    
    Args:
        metrics_list: List of metric dictionaries
        labels: Labels for each set of metrics
        title: Plot title
        
    Returns:
        Path to saved figure
    """
    # Extract common metrics
    common_metrics = set.intersection(*[set(m.keys()) for m in metrics_list])
    common_metrics = [m for m in common_metrics if not m.startswith("batch_") and not m.startswith("sample_")]
    
    if not common_metrics:
        print("No common metrics found for comparison")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up bar positions
    n_metrics = len(common_metrics)
    n_models = len(metrics_list)
    width = 0.8 / n_models
    
    # Plot bars for each model
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        positions = np.arange(n_metrics) + (i - n_models/2 + 0.5) * width
        values = [metrics.get(metric, 0) for metric in common_metrics]
        ax.bar(positions, values, width, label=label)
    
    # Set labels and title
    ax.set_xticks(np.arange(n_metrics))
    ax.set_xticklabels(common_metrics)
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend()
    
    # Save figure
    figure_path = "metrics_comparison.png"
    plt.savefig(figure_path)
    plt.close()
    
    return figure_path


def plot_inference_time_comparison(timing_list: List[Dict[str, float]], labels: List[str],
                                 title: str = "Inference Time Comparison") -> str:
    """Plot comparison of inference times across different models or versions.
    
    Args:
        timing_list: List of timing dictionaries
        labels: Labels for each set of timings
        title: Plot title
        
    Returns:
        Path to saved figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract timing metrics
    batch_times = [timing.get("batch_inference_time", 0) for timing in timing_list]
    sample_times = [timing.get("sample_inference_time", 0) for timing in timing_list]
    
    # Plot batch inference times
    ax1.bar(labels, batch_times)
    ax1.set_title("Batch Inference Time (s)")
    ax1.set_ylabel("Time (seconds)")
    
    # Plot per-sample inference times
    ax2.bar(labels, sample_times)
    ax2.set_title("Per-Sample Inference Time (s)")
    ax2.set_ylabel("Time (seconds)")
    
    # Adjust layout
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save figure
    figure_path = "inference_time_comparison.png"
    plt.savefig(figure_path)
    plt.close()
    
    return figure_path