# MLflow Project - Model Training Module
# Handles training and logging of different model versions with MLflow

import os
import mlflow
import numpy as np
import pandas as pd
import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
from typing import Dict, Tuple, List, Any, Optional, Union
import time
import json
import torch

# Import utility functions
from utils import (
    load_dataset, evaluate_model, log_model_metrics, 
    measure_inference_time, apply_quantization, apply_pruning
)


def get_model(model_type: str, params: Dict[str, Any] = None) -> Any:
    """Create a model instance based on model type and parameters.
    
    Args:
        model_type: Type of model to create (rfdetr, rfdetr_large)
        params: Dictionary of model parameters
        
    Returns:
        Model instance
    """
    if params is None:
        params = {}
    
    if model_type.lower() == "rfdetr":
        model = RFDETRBase()
        # Apply any additional configuration from params
        if params.get("optimize_for_inference", False):
            model.optimize_for_inference()
        return model
    elif model_type.lower() == "rfdetr_large":
        # For demonstration purposes - in a real scenario, you might have different model variants
        model = RFDETRBase(model_type="large")
        if params.get("optimize_for_inference", False):
            model.optimize_for_inference()
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from: rfdetr, rfdetr_large")


def train_and_log_model(
    model_type: str,
    model_params: Dict[str, Any],
    dataset_name: str = "coco",
    experiment_name: str = "model_training",
    run_name: Optional[str] = None,
    optimization: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
    image_urls: List[str] = None
) -> Tuple[Any, str, Any, Any]:
    """Train a model and log metrics and artifacts to MLflow.
    
    Args:
        model_type: Type of model to train
        model_params: Model parameters
        dataset_name: Name of dataset to use
        experiment_name: MLflow experiment name
        run_name: MLflow run name (optional)
        optimization: Dictionary with optimization settings (optional)
            - precision: Quantization precision (int8, int16, float16)
            - pruning_level: Level of pruning (0.0 to 1.0)
        tags: Additional tags to log with the run
        
    Returns:
        Tuple of (model, run_id, X_test, y_test)
    """
    # Set tracking URI and experiment
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)
    
    # Generate run name if not provided
    if run_name is None:
        run_name = f"{model_type}_{int(time.time())}"
    
    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset(
        dataset_name, 
        image_urls=image_urls
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        
        # Log dataset info
        mlflow.log_param("dataset", dataset_name)
        if isinstance(X_train, list):
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
        else:
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
        
        # Log model info
        mlflow.log_param("model_type", model_type)
        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Create model
        start_time = time.time()
        model = get_model(model_type, model_params)
        # For RFDETR, we don't need to train the model as we're using a pre-trained model
        if not isinstance(model, RFDETRBase):
            model.fit(X_train, y_train)
        training_time = time.time() - start_time
        mlflow.log_metric("training_time", training_time)
        
        # Evaluate model
        train_metrics = evaluate_model(model, X_train, y_train, prefix="train_")
        test_metrics = evaluate_model(model, X_test, y_test, prefix="test_")
        
        # Log metrics
        log_model_metrics(train_metrics)
        log_model_metrics(test_metrics)
        
        # Measure and log inference time
        timing_metrics = measure_inference_time(model, X_test)
        log_model_metrics(timing_metrics)
        
        # For RFDETR, log sample detections as artifacts
        if isinstance(model, RFDETRBase) and isinstance(X_test, list) and len(X_test) > 0:
            # Create a directory for detection images
            os.makedirs("detections", exist_ok=True)
            
            # Process a few test images and save the detection results
            for i, img in enumerate(X_test[:3]):  # Process up to 3 images
                # Run inference
                detections = model.predict(img)
                
                # Create labels for visualization
                labels = [
                    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                    for class_id, confidence in zip(detections.class_id, detections.confidence)
                ]
                
                # Convert PIL Image to numpy array for supervision visualization
                image_np = sv.Image.from_pil(img)
                
                # Create box annotator
                box_annotator = sv.BoxAnnotator()
                
                # Annotate image
                annotated_image = box_annotator.annotate(
                    scene=image_np.copy(),
                    detections=detections,
                    labels=labels
                )
                
                # Save the annotated image
                output_path = f"detections/detection_{i}.jpg"
                sv.plot_image(annotated_image, output_path)
                
                # Log the image as an artifact
                mlflow.log_artifact(output_path)
        
        # Apply optimizations if specified
        optimized_model = model
        if optimization:
            # Log optimization parameters
            for opt_name, opt_value in optimization.items():
                mlflow.log_param(f"optimization_{opt_name}", opt_value)
            
            # Apply quantization if specified
            if "precision" in optimization and isinstance(model, RFDETRBase):
                precision = optimization["precision"]
                mlflow.log_param("quantization_precision", precision)
                
                # For RFDETR, we'll simulate quantization by logging it as a parameter
                # In a real scenario, you would apply actual quantization to the model
                mlflow.log_param("quantized_model", "simulated")
                
                # For demonstration, we'll log slightly different metrics to simulate quantization effects
                if isinstance(X_test, list) and len(X_test) > 0:
                    # Simulate quantized model evaluation
                    # In a real scenario, you would evaluate the actual quantized model
                    quantized_metrics = {}
                    for key, value in test_metrics.items():
                        # Simulate slight degradation in metrics due to quantization
                        if "confidence" in key:
                            quantized_metrics[f"quantized_{key}"] = value * 0.95  # Slightly lower confidence
                        elif "detections" in key:
                            quantized_metrics[f"quantized_{key}"] = value  # Same number of detections
                    
                    # Log the simulated metrics
                    log_model_metrics(quantized_metrics)
                    
                    # Simulate faster inference time for quantized model
                    quantized_timing = {}
                    for key, value in timing_metrics.items():
                        quantized_timing[f"quantized_{key}"] = value * 0.8  # Simulate 20% faster inference
                    
                    log_model_metrics(quantized_timing)
            
            # Apply pruning if specified
            if "pruning_level" in optimization and isinstance(model, RFDETRBase):
                pruning_level = optimization["pruning_level"]
                mlflow.log_param("pruning_level", pruning_level)
                
                # For RFDETR, we'll simulate pruning by logging it as a parameter
                # In a real scenario, you would apply actual pruning to the model
                mlflow.log_param("pruned_model", "simulated")
                
                # For demonstration, we'll log slightly different metrics to simulate pruning effects
                if isinstance(X_test, list) and len(X_test) > 0:
                    # Simulate pruned model evaluation
                    # In a real scenario, you would evaluate the actual pruned model
                    pruned_metrics = {}
                    for key, value in test_metrics.items():
                        # Simulate slight degradation in metrics due to pruning
                        if "confidence" in key:
                            pruned_metrics[f"pruned_{key}"] = value * 0.9  # Lower confidence due to pruning
                        elif "detections" in key:
                            pruned_metrics[f"pruned_{key}"] = value * 0.95  # Slightly fewer detections
                    
                    # Log the simulated metrics
                    log_model_metrics(pruned_metrics)
                    
                    # Simulate faster inference time for pruned model
                    pruned_timing = {}
                    for key, value in timing_metrics.items():
                        pruned_timing[f"pruned_{key}"] = value * 0.7  # Simulate 30% faster inference
                    
                    log_model_metrics(pruned_timing)
        
        # Log additional tags
        if tags:
            for tag_name, tag_value in tags.items():
                mlflow.set_tag(tag_name, tag_value)
        
        # Log model
        if isinstance(model, RFDETRBase):
            # For RFDETR, we can't use mlflow.sklearn.log_model
            # Instead, we'll log model information as parameters and artifacts
            mlflow.log_param("model_framework", "rfdetr")
            mlflow.log_param("model_type", model_type)
            
            # In a real scenario, you would save and log the actual model file
            # For demonstration, we'll just log a model info file
            model_info = {
                "model_type": model_type,
                "framework": "rfdetr",
                "optimized_for_inference": model_params.get("optimize_for_inference", False)
            }
            
            # Save model info to a JSON file
            with open("model_info.json", "w") as f:
                json.dump(model_info, f)
            
            # Log the model info file as an artifact
            mlflow.log_artifact("model_info.json")
        else:
            mlflow.sklearn.log_model(optimized_model, "model")
        
        # Log model summary
        model_info = {
            "model_type": model_type,
            "parameters": model_params,
            "dataset": dataset_name,
            "training_time": training_time,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "batch_inference_time": timing_metrics.get("batch_inference_time", 0),
            "optimizations": optimization or {}
        }
        with open("model_summary.json", "w") as f:
            json.dump(model_info, f, indent=2)
        mlflow.log_artifact("model_summary.json")
        
        print(f"Run ID: {run_id}")
        print(f"Model trained and logged to MLflow")
        print(f"Training accuracy: {train_metrics['train_accuracy']:.4f}")
        print(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")
        
        return optimized_model, run_id, X_test, y_test


def train_model_variants(
    base_model_type: str,
    param_variations: List[Dict[str, Any]],
    dataset_name: str = "coco",
    experiment_name: str = "model_variants",
    optimization: Optional[Dict[str, Any]] = None,
    image_urls: List[str] = None
) -> List[str]:
    """Train multiple variants of a model with different parameters.
    
    Args:
        base_model_type: Base model type
        param_variations: List of parameter dictionaries for different variants
        dataset_name: Name of dataset to use
        experiment_name: MLflow experiment name
        optimization: Dictionary with optimization settings (optional)
        
    Returns:
        List of run IDs for the trained models
    """
    run_ids = []
    
    for i, params in enumerate(param_variations):
        variant_name = f"{base_model_type}_variant_{i+1}"
        print(f"\nTraining {variant_name} with parameters: {params}")
        
        _, run_id, _, _ = train_and_log_model(
            model_type=base_model_type,
            model_params=params,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            run_name=variant_name,
            optimization=optimization,
            tags={"variant": str(i+1), "variant_group": base_model_type}
        )
        
        run_ids.append(run_id)
    
    return run_ids


def train_with_optimizations(
    model_type: str,
    model_params: Dict[str, Any],
    dataset_name: str = "coco",
    experiment_name: str = "model_optimizations",
    optimizations: List[Dict[str, Any]] = None,
    image_urls: List[str] = None
) -> List[str]:
    """Train a model with different optimization techniques.
    
    Args:
        model_type: Model type
        model_params: Model parameters
        dataset_name: Name of dataset to use
        experiment_name: MLflow experiment name
        optimizations: List of optimization settings to try
        
    Returns:
        List of run IDs for the trained models
    """
    if optimizations is None:
        optimizations = [
            {"precision": "int8"},
            {"precision": "float16"},
            {"pruning_level": 0.3},
            {"pruning_level": 0.5, "precision": "int8"}
        ]
    
    run_ids = []
    
    # First train the baseline model without optimizations
    print("\nTraining baseline model without optimizations")
    _, baseline_run_id, _, _ = train_and_log_model(
        model_type=model_type,
        model_params=model_params,
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        run_name=f"{model_type}_baseline",
        tags={"optimization": "none", "optimization_group": model_type}
    )
    run_ids.append(baseline_run_id)
    
    # Train with each optimization setting
    for i, opt in enumerate(optimizations):
        opt_name = "_".join([f"{k}_{v}" for k, v in opt.items()])
        print(f"\nTraining with optimization: {opt}")
        
        _, run_id, _, _ = train_and_log_model(
            model_type=model_type,
            model_params=model_params,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            run_name=f"{model_type}_{opt_name}",
            optimization=opt,
            tags={"optimization": opt_name, "optimization_group": model_type}
        )
        
        run_ids.append(run_id)
    
    return run_ids