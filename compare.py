# MLflow Project - Model Comparison Module
# Analyzes performance across different model versions and optimizations

import os
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional, Union, Any, Tuple
import json

# Import utility functions
from utils import plot_metrics_comparison, plot_inference_time_comparison

# Import RFDETR specific modules
import io
import requests
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv


def get_experiment_runs(experiment_name: str, max_runs: int = 10) -> List[mlflow.entities.Run]:
    """Get runs for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        max_runs: Maximum number of runs to retrieve
        
    Returns:
        List of MLflow Run objects
    """
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Get experiment
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return []
    
    # Get runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_runs
    )
    
    if not runs:
        print(f"No runs found for experiment '{experiment_name}'.")
        return []
    
    return runs


def compare_models(experiment_name: str, metric_name: str = "test_accuracy", 
                  max_runs: int = 10, filter_tags: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Compare performance of different model runs.
    
    Args:
        experiment_name: Name of the experiment
        metric_name: Primary metric for comparison
        max_runs: Maximum number of runs to compare
        filter_tags: Optional dictionary of tags to filter runs
        
    Returns:
        DataFrame with model comparison
    """
    # Get runs
    runs = get_experiment_runs(experiment_name, max_runs)
    
    # Filter runs by tags if specified
    if filter_tags and runs:
        filtered_runs = []
        for run in runs:
            include_run = True
            for tag_key, tag_value in filter_tags.items():
                if run.data.tags.get(tag_key) != tag_value:
                    include_run = False
                    break
            if include_run:
                filtered_runs.append(run)
        runs = filtered_runs
    
    if not runs:
        print("No runs available for comparison.")
        return pd.DataFrame()
    
    # Extract run data
    run_data = []
    for run in runs:
        run_info = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", "Unknown"),
            "model_type": run.data.params.get("model_type", "Unknown"),
            "dataset": run.data.params.get("dataset", "Unknown")
        }
        
        # Add parameters
        for param_key, param_value in run.data.params.items():
            if param_key not in ["model_type", "dataset"]:
                run_info[f"param_{param_key}"] = param_value
        
        # Add metrics
        for metric_key, metric_value in run.data.metrics.items():
            run_info[metric_key] = metric_value
        
        # Add tags
        for tag_key, tag_value in run.data.tags.items():
            if tag_key.startswith("user."):
                run_info[f"tag_{tag_key[5:]}"] = tag_value
        
        run_data.append(run_info)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(run_data)
    
    # Sort by primary metric if available
    if metric_name in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by=metric_name, ascending=False).reset_index(drop=True)
    
    return comparison_df


def compare_model_variants(experiment_name: str, variant_group: str, 
                          metrics: List[str] = None) -> Tuple[pd.DataFrame, str]:
    """Compare different variants of the same model type.
    
    Args:
        experiment_name: Name of the experiment
        variant_group: Value of the variant_group tag to filter by
        metrics: List of metrics to include in comparison
        
    Returns:
        Tuple of (comparison DataFrame, path to comparison figure)
    """
    if metrics is None:
        metrics = ["test_accuracy", "test_f1", "batch_inference_time"]
    
    # Get runs with the specified variant_group tag
    filter_tags = {"variant_group": variant_group}
    comparison_df = compare_models(experiment_name, "test_accuracy", 50, filter_tags)
    
    if comparison_df.empty:
        print(f"No variants found for group '{variant_group}'")
        return comparison_df, None
    
    # Extract metrics for visualization
    metrics_list = []
    labels = []
    
    for _, row in comparison_df.iterrows():
        metrics_dict = {metric: row.get(metric, 0) for metric in metrics if metric in row}
        if metrics_dict:
            metrics_list.append(metrics_dict)
            labels.append(row.get("run_name", f"Run {row.get('run_id', '')}"))
    
    # Create comparison plot
    if metrics_list:
        figure_path = plot_metrics_comparison(
            metrics_list, labels, 
            title=f"Model Variant Comparison - {variant_group}"
        )
        mlflow.log_artifact(figure_path)
    else:
        figure_path = None
    
    return comparison_df, figure_path


def compare_optimizations(experiment_name: str, optimization_group: str) -> Tuple[pd.DataFrame, str, str]:
    """Compare different optimization techniques for the same model type.
    
    Args:
        experiment_name: Name of the experiment
        optimization_group: Value of the optimization_group tag to filter by
        
    Returns:
        Tuple of (comparison DataFrame, path to metrics figure, path to timing figure)
    """
    # Get runs with the specified optimization_group tag
    filter_tags = {"optimization_group": optimization_group}
    comparison_df = compare_models(experiment_name, "test_accuracy", 50, filter_tags)
    
    if comparison_df.empty:
        print(f"No optimization runs found for group '{optimization_group}'")
        return comparison_df, None, None
    
    # Extract metrics for visualization
    metrics_list = []
    timing_list = []
    labels = []
    
    for _, row in comparison_df.iterrows():
        # Performance metrics
        metrics_dict = {
            "test_accuracy": row.get("test_accuracy", 0),
            "test_f1": row.get("test_f1", 0),
            "quantized_accuracy": row.get("quantized_accuracy", row.get("test_accuracy", 0)),
            "pruned_accuracy": row.get("pruned_accuracy", row.get("test_accuracy", 0))
        }
        metrics_list.append(metrics_dict)
        
        # Timing metrics
        timing_dict = {
            "batch_inference_time": row.get("batch_inference_time", 0),
            "sample_inference_time": row.get("sample_inference_time", 0),
            "quantized_batch_inference_time": row.get("quantized_batch_inference_time", row.get("batch_inference_time", 0)),
            "pruned_batch_inference_time": row.get("pruned_batch_inference_time", row.get("batch_inference_time", 0))
        }
        timing_list.append(timing_dict)
        
        # Label
        opt_tag = row.get("tag_optimization", "baseline")
        labels.append(f"{opt_tag}")
    
    # Create comparison plots
    metrics_figure = None
    timing_figure = None
    
    if metrics_list:
        metrics_figure = plot_metrics_comparison(
            metrics_list, labels, 
            title=f"Optimization Performance Comparison - {optimization_group}"
        )
        mlflow.log_artifact(metrics_figure)
    
    if timing_list:
        timing_figure = plot_inference_time_comparison(
            timing_list, labels,
            title=f"Optimization Timing Comparison - {optimization_group}"
        )
        mlflow.log_artifact(timing_figure)
    
    return comparison_df, metrics_figure, timing_figure


def get_best_model(experiment_name: str, metric_name: str = "test_accuracy",
                  filter_tags: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Get the run ID of the best performing model based on a metric.
    
    Args:
        experiment_name: Name of the experiment
        metric_name: Metric to use for ranking
        filter_tags: Optional dictionary of tags to filter runs
        
    Returns:
        Run ID of the best model or None if no models found
    """
    comparison_df = compare_models(experiment_name, metric_name, 50, filter_tags)
    
    if comparison_df.empty or metric_name not in comparison_df.columns:
        return None
    
    best_run = comparison_df.iloc[0]
    best_run_id = best_run["run_id"]
    best_metric = best_run[metric_name]
    
    print(f"\nBest Model (based on {metric_name}):\n")
    print(f"Run ID: {best_run_id}")
    print(f"Run Name: {best_run.get('run_name', 'Unknown')}")
    print(f"Model Type: {best_run.get('model_type', 'Unknown')}")
    print(f"{metric_name}: {best_metric:.4f}")
    
    return best_run_id


def compare_across_experiments(experiment_names: List[str], metric_name: str = "test_accuracy",
                              max_runs_per_exp: int = 3) -> pd.DataFrame:
    """Compare top models across different experiments.
    
    Args:
        experiment_names: List of experiment names to compare
        metric_name: Metric to use for ranking
        max_runs_per_exp: Maximum number of top runs to include from each experiment
        
    Returns:
        DataFrame with cross-experiment comparison
    """
    all_top_runs = []
    
    for exp_name in experiment_names:
        print(f"\nGetting top runs from experiment: {exp_name}")
        comparison_df = compare_models(exp_name, metric_name, max_runs_per_exp)
        
        if not comparison_df.empty:
            # Add experiment name column
            comparison_df["experiment"] = exp_name
            all_top_runs.append(comparison_df)
    
    if not all_top_runs:
        print("No runs found across the specified experiments.")
        return pd.DataFrame()
    
    # Combine all top runs
    combined_df = pd.concat(all_top_runs, ignore_index=True)
    
    # Sort by the specified metric
    if metric_name in combined_df.columns:
        combined_df = combined_df.sort_values(by=metric_name, ascending=False).reset_index(drop=True)
    
    return combined_df


def export_comparison_report(comparison_df: pd.DataFrame, report_name: str = "model_comparison_report") -> str:
    """Export a model comparison report as HTML and CSV.
    
    Args:
        comparison_df: DataFrame with model comparison data
        report_name: Base name for the report files
        
    Returns:
        Path to the HTML report
    """
    if comparison_df.empty:
        print("No data available for report generation.")
        return None
    
    # Save CSV version
    csv_path = f"{report_name}.csv"
    comparison_df.to_csv(csv_path, index=False)
    
    # Create HTML report
    html_path = f"{report_name}.html"
    
    # Basic styling for the HTML report
    html_content = f"""
    <html>
    <head>
        <title>MLflow Model Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #0066cc; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric-cell {{ font-weight: bold; color: #0066cc; }}
        </style>
    </head>
    <body>
        <h1>MLflow Model Comparison Report</h1>
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <table>
    """
    
    # Add table header
    html_content += "<tr>"
    for col in comparison_df.columns:
        html_content += f"<th>{col}</th>"
    html_content += "</tr>"
    
    # Add table rows
    for _, row in comparison_df.iterrows():
        html_content += "<tr>"
        for col in comparison_df.columns:
            value = row.get(col, "")
            
            # Format numeric values
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if col.startswith(("test_", "train_", "val_", "quantized_", "pruned_")):
                    cell_class = "metric-cell"
                    value = f"{value:.4f}"
                else:
                    cell_class = ""
                    value = f"{value}"
            else:
                cell_class = ""
            
            html_content += f"<td class='{cell_class}'>{value}</td>"
        html_content += "</tr>"
    
    # Close HTML tags
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(html_path, "w") as f:
        f.write(html_content)
    
    # Log artifacts to MLflow
    mlflow.log_artifact(csv_path)
    mlflow.log_artifact(html_path)
    
    print(f"\nComparison report exported to:")
    print(f"- CSV: {csv_path}")
    print(f"- HTML: {html_path}")
    
    return html_path