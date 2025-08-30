# MLflow RFDETR Object Detection Project

A comprehensive MLflow-based framework for object detection using RFDETR models. This project helps data scientists and ML engineers to track experiments, compare model performance, apply optimization techniques, and monitor model drift for object detection tasks.

## Project Structure

```
│── main.py               # Entry point (parse args, call functions)
│── train.py              # Training + logging to MLflow
│── compare.py            # Compare models across runs
│── drift.py              # Drift detection & monitoring
│── utils.py              # Common helper functions
│── requirements.txt      # Dependencies
│── README.md             # Instructions
│── mlruns/               # Auto-created by MLflow (stores runs)
```

## Features

- **Object Detection**: Use RFDETR models for object detection tasks
- **Model Variants**: Compare different RFDETR model variants
- **Model Optimization**: Apply quantization and pruning techniques to RFDETR models
- **Performance Comparison**: Compare detection performance across different runs
- **Drift Detection**: Monitor model performance under different drift conditions
- **Precision Format Testing**: Evaluate model performance with different numerical precisions (int8, int16, float16)
- **Visualization**: Generate detection visualizations and performance reports

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Create a virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
# Use RFDETR model for object detection
python main.py --mode train --model-type rfdetr --image-urls https://example.com/image1.jpg https://example.com/image2.jpg

# Use RFDETR large model with inference optimization
python main.py --mode train --model-type rfdetr_large --optimize-for-inference

# Compare models
python main.py --mode compare

# Check model drift
python main.py --mode drift --drift-factor 0.1

# Run all operations
python main.py --mode all
```

### Advanced Usage

#### Using Different RFDETR Model Variants

```bash
# Compare different RFDETR model variants
python main.py --mode train --model-type rfdetr --variant-group rfdetr_variants
python main.py --mode train --model-type rfdetr_large --variant-group rfdetr_variants
```

#### RFDETR Model Optimization

```bash
# Use RFDETR model with optimization techniques
python main.py --mode train --model-type rfdetr --optimize --optimize-for-inference

# Apply optimization to existing RFDETR models
python main.py --mode optimize --optimization-group rfdetr_optimizations
```

#### Drift Detection with Different Precision Formats

```bash
# Check drift with specific precision formats
python main.py --mode drift --precision-formats int8,int16,float16

# Monitor drift over time with different drift factors
python main.py --mode drift --drift-type feature
```

#### Comparing Model Robustness

```bash
# Compare robustness of top models
python main.py --mode drift --top-n 5
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|--------|
| `--mode` | Operation mode: train, compare, drift, optimize, or all | all |
| `--experiment-name` | MLflow experiment name | model-comparison |
| `--run-name` | MLflow run name | None |
| `--model-type` | Model type to train | random_forest |
| `--variant-group` | Variant group name for model comparison | None |
| `--optimize` | Apply optimization techniques | False |
| `--optimization-group` | Optimization group name for comparison | None |
| `--drift-factor` | Magnitude of drift to simulate (0.0 to 1.0) | 0.1 |
| `--drift-type` | Type of drift to simulate | feature |
| `--precision-formats` | Comma-separated list of precision formats to test | int8,float16 |
| `--metric` | Primary metric for model comparison | test_accuracy |
| `--top-n` | Number of top models to compare | 3 |
| `--export-report` | Export comparison report as HTML and CSV | False |

## Viewing Results

Start the MLflow UI to view experiment results:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open your browser and navigate to http://localhost:5000

## Examples

### Example 1: Complete Model Lifecycle

```bash
# Train multiple model variants
python main.py --mode train --model-type all --variant-group model_variants_v1

# Compare model variants
python main.py --mode compare --variant-group model_variants_v1 --export-report

# Apply optimizations to the best model
python main.py --mode optimize --optimization-group opt_variants_v1

# Check drift for optimized models
python main.py --mode drift --optimization-group opt_variants_v1 --precision-formats int8,int16,float16
```

### Example 2: Evaluating Model Robustness

```bash
# Train models
python main.py --mode train --model-type all

# Compare robustness across different drift factors
python main.py --mode drift --drift-factor 0.05
python main.py --mode drift --drift-factor 0.1
python main.py --mode drift --drift-factor 0.2
```

## Extending the Project

### Adding New Model Types

To add a new model type, update the `get_model` function in `train.py` with your new model implementation.

### Adding New Optimization Techniques

Implement new optimization functions in `utils.py` and update the `train_with_optimizations` function in `train.py`.

### Custom Datasets

Modify the `load_dataset` function in `utils.py` to work with your custom datasets.

## License

This project is licensed under the MIT License - see the LICENSE file for details.