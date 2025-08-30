# MLflow Project - Main Entry Point
# Parses arguments and orchestrates the execution of MLflow functions

import argparse
import os
import mlflow
import sys
from typing import Dict, List, Optional, Union, Any, Tuple

# Import functions from other modules
from train import train_and_log_model, train_model_variants, train_with_optimizations
from compare import compare_models, compare_model_variants, compare_optimizations, get_best_model, export_comparison_report
from drift import check_drift, monitor_drift_over_time, compare_model_robustness
from utils import load_dataset


def setup_mlflow(experiment_name: str) -> None:
    """Set up MLflow tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
    """
    # Set tracking URI to local mlruns directory
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Set or create experiment
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow tracking set up with experiment: {experiment_name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="MLflow Model Training, Comparison, and Drift Detection")
    
    # Operation mode
    parser.add_argument(
        "--mode", 
        type=str, 
        default="all", 
        choices=["train", "compare", "drift", "optimize", "all"],
        help="Operation mode: train, compare, drift, optimize, or all"
    )
    
    # MLflow settings
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default="model-comparison",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name", 
        type=str, 
        default=None,
        help="MLflow run name"
    )
    
    # Model settings
    parser.add_argument(
        "--model-type", 
        type=str, 
        default="rfdetr",
        choices=["rfdetr", "rfdetr_large"],
        help="Type of model to use (rfdetr, rfdetr_large)"
    )
    parser.add_argument(
        "--variant-group", 
        type=str, 
        default=None,
        help="Variant group name for model comparison"
    )
    parser.add_argument(
        "--optimize-for-inference", 
        action="store_true", 
        help="Optimize RFDETR model for inference"
    )
    
    # Optimization settings
    parser.add_argument(
        "--optimize", 
        action="store_true",
        help="Apply optimization techniques (quantization, pruning)"
    )
    parser.add_argument(
        "--optimization-group", 
        type=str, 
        default=None,
        help="Optimization group name for comparison"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="coco", 
        help="Dataset to use (coco, iris, digits, wine, breast_cancer)"
    )
    parser.add_argument(
        "--image-urls", 
        type=str, 
        nargs="+", 
        default=None, 
        help="List of image URLs for object detection (only used with coco dataset)"
    )
    
    # Drift settings
    parser.add_argument(
        "--drift-factor", 
        type=float, 
        default=0.1,
        help="Magnitude of drift to simulate (0.0 to 1.0)"
    )
    parser.add_argument(
        "--drift-type", 
        type=str, 
        default="feature",
        choices=["feature", "label", "both"],
        help="Type of drift to simulate"
    )
    parser.add_argument(
        "--precision-formats", 
        type=str, 
        default="int8,float16",
        help="Comma-separated list of precision formats to test"
    )
    
    # Comparison settings
    parser.add_argument(
        "--metric", 
        type=str, 
        default="test_accuracy",
        help="Primary metric for model comparison"
    )
    parser.add_argument(
        "--top-n", 
        type=int, 
        default=3,
        help="Number of top models to compare"
    )
    
    # Output settings
    parser.add_argument(
        "--export-report", 
        action="store_true",
        help="Export comparison report as HTML and CSV"
    )
    
    return parser.parse_args()


def main():
    """Main function to orchestrate MLflow operations."""
    # Parse arguments
    args = parse_args()
    
    # Set up MLflow tracking
    setup_mlflow(args.experiment_name)
    
    # Process precision formats
    precision_formats = args.precision_formats.split(",") if args.precision_formats else []
    
    # Execute operations based on mode
    if args.mode in ["train", "all"]:
        print("\n===== Training Models =====\n")
        
        # Train RFDETR model
        run_name = args.run_name or f"{args.model_type}_model"
        
        # Create model parameters dictionary
        model_params = {
            "optimize_for_inference": args.optimize_for_inference
        }
        
        # Train and log the model
        model_run_id = train_and_log_model(
            args.model_type, 
            model_params,
            args.dataset,
            args.experiment_name,
            run_name,
            args.optimize,
            args.image_urls
        )
            
            # Apply optimizations if requested
            if args.optimize:
                print("\n===== Applying Optimizations =====\n")
                optimization_group = args.optimization_group or f"{args.model_type}_optimizations"
                train_with_optimizations(
                    model_run_id, 
                    optimization_group,
                    args.dataset,
                    args.image_urls
                )
    
    if args.mode in ["compare", "all"]:
        print("\n===== Comparing Models =====\n")
        
        if args.variant_group:
            # Compare model variants
            comparison_df, _ = compare_model_variants(args.experiment_name, args.variant_group)
            
            # Export report if requested
            if args.export_report and not comparison_df.empty:
                export_comparison_report(comparison_df, f"variant_comparison_{args.variant_group}")
        
        elif args.optimization_group:
            # Compare optimization techniques
            comparison_df, _, _ = compare_optimizations(args.experiment_name, args.optimization_group)
            
            # Export report if requested
            if args.export_report and not comparison_df.empty:
                export_comparison_report(comparison_df, f"optimization_comparison_{args.optimization_group}")
        
        else:
            # General model comparison
            comparison_df = compare_models(args.experiment_name, args.metric, args.top_n)
            
            # Get best model
            best_run_id = get_best_model(args.experiment_name, args.metric)
            
            # Export report if requested
            if args.export_report and not comparison_df.empty:
                export_comparison_report(comparison_df, "model_comparison_report")
    
    if args.mode in ["drift", "all"]:
        print("\n===== Checking Model Drift =====\n")
        
        # Get best model if not in train mode
        if args.mode != "train" and not args.model_type:
            best_run_id = get_best_model(args.experiment_name, args.metric)
        else:
            # Use the model from training step
            best_run_id = model_run_id if 'model_run_id' in locals() else None
        
        if best_run_id:
            # Check drift for the best model
            drift_metrics = check_drift(
                best_run_id, 
                args.drift_factor, 
                args.drift_type,
                precision_formats
            )
            
            # Monitor drift over time
            monitor_drift_over_time(
                best_run_id, 
                [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
                args.drift_type
            )
            
            # Compare model robustness if multiple models are available
            if args.variant_group or args.model_type == "all":
                compare_model_robustness(
                    args.experiment_name, 
                    args.drift_factor,
                    args.drift_type,
                    args.top_n
                )
        else:
            print("No model available for drift checking. Please train a model first.")
    
    if args.mode == "optimize":
        print("\n===== Optimizing Models =====\n")
        
        # Get best model if not specified
        best_run_id = get_best_model(args.experiment_name, args.metric)
        
        if best_run_id:
            # Apply optimizations
            optimization_group = args.optimization_group or "model_optimizations"
            train_with_optimizations(best_run_id, optimization_group)
            
            # Compare optimizations
            comparison_df, _, _ = compare_optimizations(args.experiment_name, optimization_group)
            
            # Export report if requested
            if args.export_report and not comparison_df.empty:
                export_comparison_report(comparison_df, f"optimization_comparison_{optimization_group}")
        else:
            print("No model available for optimization. Please train a model first.")
    
    print("\n===== MLflow Operations Complete =====\n")
    print(f"View results in the MLflow UI: mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()