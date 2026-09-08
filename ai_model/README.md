# Demand Sensing ML Models

This directory contains the machine learning pipelines for predicting short-term retail demand (specifically 2-hour ahead horizons).

## Architecture

We use a standard `src/` layout to isolate operational packages:

```
ai_model/
├── tests/                  # All unit and regression tests
├── src/                    # Isolated execution source code folder
│   └── demand_sensing/     # Package root directory
│       ├── __init__.py
│       ├── features.py     # Causal lag, rolling metrics, and stochastic scenario injection
│       ├── train.py        # LightGBM model tuning via Optuna and registration via MLflow
│       └── export.py       # ONNX graph compilation tools and model export
├── pyproject.toml          # Centralized dependency config (LightGBM, scikit-learn, ONNX)
└── README.md
```

## Features and Data Pipeline
- **Stochastic Target Injection**: Generates robust synthetic demand logic for specific scenarios to learn robust elasticity, including:
  - Weather events (Heat waves, Typhoons, Heavy rain)
  - Calendar events (Payday, Fiestas, Holy Week, Back-to-school)
  - Time-of-day features (Rush hour spikes)
  - Local operational factors (Brownouts)
  - Seasonal factors (Flu season)
- **Causal Features**: Calculates dynamic spatiotemporal metrics such as rolling means (3h, 6h, 24h), rolling standard deviation (6h), lag variables, and expanding historical averages per store and product.
- **Strict Chronological Splitting**: Data is split into 80% train and 20% test subsets based on strict timestamp sequencing, ensuring no future data leakage.
- **Time Series Cross Validation**: The training uses Optuna for hyperparameter optimization coupled with `TimeSeriesSplit` (3 splits) cross-validation, using Mean Absolute Error (MAE) and WAPE for evaluation.
- **ONNX Native Bindings**: Converts both preprocessors (like `OrdinalEncoder`) and LightGBM models into unified structural ONNX graphs (`agent_demand_2h.onnx`) for optimized inference.

## Execution

### 1. Setup Environment
Before running the pipelines, set up your Python environment and install the required dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r output/requirements.txt
```

### 2. Run Pipeline
Run the training pipeline (trains the LightGBM model using Optuna, tracks experiments with MLflow, and saves `.pkl` outputs):
```bash
python -m src.demand_sensing.train
```

Export the ONNX asset (compiles the LightGBM model and OrdinalEncoder into a single unified `.onnx` file):
```bash
python -m src.demand_sensing.export
```

Run the tests:
```bash
# Run all test suites
python -m pytest tests/

# Or run the model test directly
python tests/test_model.py
```

### 3. Local Demand Forecasting Inference (ONNX)
The exported `.onnx` model (`output/agent_demand_2h.onnx`) can be used for local, edge-based inference without requiring LightGBM or Scikit-Learn dependencies. You can run it using the `onnxruntime` engine in Python (or `onnxruntime-node` in TypeScript/JavaScript).

**Python Example:**
```python
import onnxruntime as ort
import numpy as np

# Load the compiled ONNX graph
session = ort.InferenceSession("output/agent_demand_2h.onnx")

# Inputs must map exactly to the feature column names and expected types (2D arrays)
# Categorical features are strings, numeric features are floats.
inputs = {
    "store_id": np.array([["007113ac-1c31-4b8f-9225-d1d9950d10b5"]], dtype=object),
    "price": np.array([[50.0]], dtype=np.float32),
    "inventory": np.array([[100.0]], dtype=np.float32),
    # ... include all required feature columns
}

# Run inference
predicted_demand = session.run(None, inputs)
print(f"Predicted 2-Hour Demand: {max(0, predicted_demand[0][0][0])}")
```

For integration into the TypeScript Edge Agent, see the `ai_agent/predictor/` module in this repository.
