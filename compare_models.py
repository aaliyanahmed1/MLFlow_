"""RF-DETR Model Comparison Script.

This script loads all RF-DETR model variants (nano, small, base, large),
performs inference on an input image provided via CLI, and generates a comprehensive comparison
report of speed, accuracy, and other metrics.

Usage:
    python compare_rfdetr_models.py --image path/to/your/image.jpg [--confidence 0.5] [--no-mlflow]
"""

import os
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from PIL import Image

# Import Roboflow Inference SDK for RF-DETR models
try:
    from inference import get_model
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("Roboflow Inference SDK not found. Install with: pip install inference")

# Try to import utils for enhanced output
try:
    from utils import print_section, print_subsection, print_environment_info
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    
    # Define minimal versions of utility functions
    def print_section(title):
        """Print a section title."""
        print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    
    def print_subsection(title):
        """Print a subsection title."""
        print(f"\n{'-' * 40}\n{title}\n{'-' * 40}")
    
    def print_environment_info():
        """Print basic environment information."""
        print("Environment Information:")
        print(f"- Python: {os.sys.version.split()[0]}")
        print(f"- MLflow: {mlflow.__version__}")
        print(f"- Working Directory: {os.getcwd()}")

def load_rfdetr_model(model_variant="base"):
    """Load an RF-DETR model variant using Roboflow Inference SDK.
    
    Args:
        model_variant (str): RF-DETR variant to load ('nano', 'small', 'base', 'large')
        
    Returns:
        model: Loaded RF-DETR model
        dict: Model metadata including size and variant
    """
    variant_map = {
        "nano": "rfdetr-nano/1",
        "small": "rfdetr-small/1",
        "base": "rfdetr-base/1",
        "large": "rfdetr-large/1"
    }
    
    model_id = variant_map.get(model_variant, "rfdetr-base/1")
    
    t0 = time.perf_counter()
    model = get_model(model_id)
    load_time = (time.perf_counter() - t0) * 1000.0  # ms
    
    # Approximate model sizes in MB
    model_sizes = {
        "nano": 15,
        "small": 35,
        "base": 45,
        "large": 300
    }
    
    metadata = {
        "variant": model_variant,
        "model_id": model_id,
        "size_mb": model_sizes.get(model_variant, 0),
        "load_time_ms": load_time
    }
    
    return model, metadata

def evaluate_rfdetr_model(model, image_path, confidence_threshold=0.5):
    """Run inference with RF-DETR model and calculate metrics.
    
    Args:
        model: RF-DETR model from Roboflow Inference SDK
        image_path (str): Path to test image
        confidence_threshold (float): Detection confidence threshold
        
    Returns:
        dict: Metrics including inference time and detection counts
    """
    # Ensure image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Run inference and time it
    t0 = time.perf_counter()
    outputs = model.infer(image_path, confidence=confidence_threshold)
    infer_time = (time.perf_counter() - t0) * 1000.0  # ms
    
    # Process predictions
    if isinstance(outputs, list):
        data = outputs[0]
    else:
        data = outputs
    
    preds = data.get("predictions", data)
    if isinstance(preds, dict):
        preds = [preds]
    
    # Extract detection metrics
    num_detections = len(preds)
    confidences = [float(p.get("confidence", p.get("score", 0.0))) for p in preds]
    avg_confidence = sum(confidences) / max(1, len(confidences))
    
    # Calculate additional metrics
    # For object detection, we can calculate a simple performance score
    # Lower is better (combines speed and detection quality)
    performance_score = infer_time / (avg_confidence + 0.1)  # Avoid division by zero
    
    metrics = {
        "inference_time_ms": infer_time,
        "num_detections": num_detections,
        "avg_confidence": avg_confidence,
        "performance_score": performance_score,
        "image_width": image.width,
        "image_height": image.height
    }
    
    return metrics

def compare_models(image_path, confidence_threshold=0.5, log_to_mlflow=True):
    """Compare all RF-DETR model variants on the same image.
    
    Args:
        image_path (str): Path to test image
        confidence_threshold (float): Detection confidence threshold
        log_to_mlflow (bool): Whether to log results to MLflow
        
    Returns:
        DataFrame: Comparison results
    """
    if not RFDETR_AVAILABLE:
        print("Error: Roboflow Inference SDK not available. Install with: pip install inference")
        return None
    
    # RF-DETR variants to compare
    variants = ["nano", "small", "base", "large"]
    results = []
    
    # Set experiment if logging to MLflow
    if log_to_mlflow:
        mlflow.set_experiment("RF-DETR-Comparison")
    
    for variant in variants:
        print_subsection(f"Evaluating RF-DETR {variant}")
        
        try:
            # Start MLflow run if logging enabled
            if log_to_mlflow:
                run_context = mlflow.start_run(run_name=f"rfdetr_{variant}_comparison")
            else:
                # Use a dummy context manager if not logging to MLflow
                from contextlib import nullcontext
                run_context = nullcontext()
            
            with run_context:
                # Load model
                print(f"Loading RF-DETR {variant} model...")
                model, metadata = load_rfdetr_model(variant)
                print(f"Model loaded in {metadata['load_time_ms']:.2f} ms")
                
                # Evaluate model
                print(f"Running inference on {image_path}...")
                metrics = evaluate_rfdetr_model(model, image_path, confidence_threshold)
                
                # Log parameters and metrics to MLflow if enabled
                if log_to_mlflow:
                    # Log parameters
                    mlflow.log_params({
                        "model_variant": variant,
                        "model_id": metadata["model_id"],
                        "model_size_mb": metadata["size_mb"],
                        "confidence_threshold": confidence_threshold
                    })
                    
                    # Log metrics
                    mlflow.log_metrics({
                        "load_time_ms": metadata["load_time_ms"],
                        "inference_time_ms": metrics["inference_time_ms"],
                        "num_detections": metrics["num_detections"],
                        "avg_confidence": metrics["avg_confidence"],
                        "performance_score": metrics["performance_score"]
                    })
                
                # Combine metadata and metrics
                result = {
                    "variant": variant,
                    "size_mb": metadata["size_mb"],
                    "load_time_ms": metadata["load_time_ms"],
                    "inference_time_ms": metrics["inference_time_ms"],
                    "num_detections": metrics["num_detections"],
                    "avg_confidence": metrics["avg_confidence"],
                    "performance_score": metrics["performance_score"]
                }
                
                results.append(result)
                
                # Print results
                print(f"Results for RF-DETR {variant}:")
                print(f"  - Model Size: {metadata['size_mb']} MB")
                print(f"  - Load Time: {metadata['load_time_ms']:.2f} ms")
                print(f"  - Inference Time: {metrics['inference_time_ms']:.2f} ms")
                print(f"  - Detections: {metrics['num_detections']}")
                print(f"  - Avg Confidence: {metrics['avg_confidence']:.4f}")
                print(f"  - Performance Score: {metrics['performance_score']:.2f} (lower is better)")
                
        except Exception as e:
            print(f"Error evaluating RF-DETR {variant}: {str(e)}")
    
    # Create DataFrame from results
    if results:
        results_df = pd.DataFrame(results)
        return results_df
    else:
        return None

def plot_comparison_results(results_df):
    """Create visualizations comparing RF-DETR model variants.
    
    Args:
        results_df (DataFrame): DataFrame with comparison results
    """
    if results_df is None or len(results_df) == 0:
        print("No results to plot")
        return
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("RF-DETR Model Variant Comparison", fontsize=16)
    
    # Sort by model size (nano, small, base, large)
    variant_order = {"nano": 0, "small": 1, "base": 2, "large": 3}
    results_df["variant_order"] = results_df["variant"].map(variant_order)
    sorted_df = results_df.sort_values("variant_order")
    labels = sorted_df["variant"]
    
    # Plot 1: Model Size
    ax = axs[0, 0]
    bars = ax.bar(labels, sorted_df["size_mb"])
    ax.set_title("Model Size (MB)")
    ax.set_ylabel("Size (MB)")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    # Plot 2: Load Time
    ax = axs[0, 1]
    bars = ax.bar(labels, sorted_df["load_time_ms"])
    ax.set_title("Model Load Time (ms)")
    ax.set_ylabel("Time (ms)")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    # Plot 3: Inference Time
    ax = axs[1, 0]
    bars = ax.bar(labels, sorted_df["inference_time_ms"])
    ax.set_title("Inference Time (ms)")
    ax.set_ylabel("Time (ms)")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    # Plot 4: Average Confidence
    ax = axs[1, 1]
    bars = ax.bar(labels, sorted_df["avg_confidence"])
    ax.set_title("Average Confidence")
    ax.set_ylabel("Confidence")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the plot
    plt.savefig("rfdetr_comparison.png")
    print("Plot saved as rfdetr_comparison.png")
    
    # Create a second figure for performance score
    plt.figure(figsize=(10, 6))
    
    # Sort by performance score (lower is better)
    perf_sorted_df = results_df.sort_values("performance_score")
    
    # Create bar chart
    bars = plt.bar(perf_sorted_df["variant"], perf_sorted_df["performance_score"])
    plt.xlabel("RF-DETR Model Variant")
    plt.ylabel("Performance Score (lower is better)")
    plt.title("RF-DETR Performance Score Comparison")
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("rfdetr_performance_score.png")
    print("Performance score plot saved as rfdetr_performance_score.png")
    
    # Show plots
    plt.show()

def generate_comparison_table(results_df):
    """Generate a formatted comparison table of RF-DETR model variants.
    
    Args:
        results_df (DataFrame): DataFrame with comparison results
    """
    if results_df is None or len(results_df) == 0:
        print("No results to display")
        return
    
    # Sort by model size (nano, small, base, large)
    variant_order = {"nano": 0, "small": 1, "base": 2, "large": 3}
    results_df["variant_order"] = results_df["variant"].map(variant_order)
    sorted_df = results_df.sort_values("variant_order")
    
    # Format the table
    print_section("RF-DETR Model Comparison Results")
    
    # Print header
    print(f"{'Variant':<10} {'Size (MB)':<10} {'Load (ms)':<10} {'Infer (ms)':<10} {'Detections':<10} {'Confidence':<10} {'Perf Score':<10}")
    print("-" * 70)
    
    # Print rows
    for _, row in sorted_df.iterrows():
        print(f"{row['variant']:<10} {row['size_mb']:<10.1f} {row['load_time_ms']:<10.1f} {row['inference_time_ms']:<10.1f} {row['num_detections']:<10} {row['avg_confidence']:<10.4f} {row['performance_score']:<10.2f}")
    
    # Find best model for each metric
    best_size = sorted_df.loc[sorted_df['size_mb'].idxmin()]
    best_load = sorted_df.loc[sorted_df['load_time_ms'].idxmin()]
    best_infer = sorted_df.loc[sorted_df['inference_time_ms'].idxmin()]
    best_conf = sorted_df.loc[sorted_df['avg_confidence'].idxmax()]
    best_perf = sorted_df.loc[sorted_df['performance_score'].idxmin()]
    
    print("\nBest Models by Metric:")
    print(f"- Smallest Size: RF-DETR {best_size['variant']} ({best_size['size_mb']} MB)")
    print(f"- Fastest Loading: RF-DETR {best_load['variant']} ({best_load['load_time_ms']:.1f} ms)")
    print(f"- Fastest Inference: RF-DETR {best_infer['variant']} ({best_infer['inference_time_ms']:.1f} ms)")
    print(f"- Highest Confidence: RF-DETR {best_conf['variant']} ({best_conf['avg_confidence']:.4f})")
    print(f"- Best Overall Performance: RF-DETR {best_perf['variant']} (score: {best_perf['performance_score']:.2f})")

def main():
    """Run the RF-DETR model comparison."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare RF-DETR model variants")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image for inference")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable logging to MLflow")
    args = parser.parse_args()
    
    # Print header
    print_section("RF-DETR Model Comparison")
    
    # Print environment info if available
    if UTILS_AVAILABLE:
        print_environment_info()
    
    # Check if Roboflow Inference SDK is available
    if not RFDETR_AVAILABLE:
        print("Error: Roboflow Inference SDK not available. Install with: pip install inference")
        return
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        print("Please provide a valid image path with --image argument")
        return
    
    # Run comparison
    print_subsection(f"Running comparison on {args.image} with confidence threshold {args.confidence}")
    results_df = compare_models(args.image, args.confidence, not args.no_mlflow)
    
    if results_df is not None:
        # Generate comparison table
        generate_comparison_table(results_df)
        
        # Plot comparison results
        plot_comparison_results(results_df)
        
        print("\nComparison complete! Results have been saved as:")
        print("- rfdetr_comparison.png")
        print("- rfdetr_performance_score.png")
        
        if not args.no_mlflow:
            print("\nResults have also been logged to MLflow. To view them, run:")
            print("mlflow ui")
    
    print("\nFor more information about MLflow Model Registry, visit:")
    print("https://www.mlflow.org/docs/latest/model-registry.html")

if __name__ == "__main__":
    main()