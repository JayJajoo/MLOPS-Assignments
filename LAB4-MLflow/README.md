# MLflow Nested Runs: Iris Classification

Comprehensive MLflow experiment demonstrating nested runs, hyperparameter tuning, and model registry for multi-class classification on the Iris dataset.

## 📋 Overview

This project trains **3 model types** with **4 configurations each** (12 total runs) using MLflow's nested runs architecture:

- **RandomForest** (4 hyperparameter configs)
- **SVC** (4 hyperparameter configs)  
- **LogisticRegression** (4 hyperparameter configs)

The best-performing model is automatically registered to MLflow Model Registry and promoted to **Production** stage.

## 🚀 Quick Start

```bash
# Install dependencies
pip install mlflow scikit-learn pandas numpy

# Train all models with nested runs
python train.py

# Run inference with registered model
python inference.py

# View results in MLflow UI
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

## 📂 Project Structure

```
├── train.py          # Main training pipeline with nested runs
├── inference.py      # Load registered model and predict
├── mlruns/           # MLflow tracking data (auto-generated)
└── README.md         # This file
```

## 🔧 What It Does

### `train.py` - Training Pipeline

1. **Loads Iris dataset** (150 samples, 4 features, 3 classes)
2. **Trains 12 models** organized as:
   - 3 parent runs (one per model type)
   - 12 child runs (4 configs per parent)
3. **Logs comprehensive artifacts**:
   - Training/test datasets
   - Confusion matrices
   - Classification reports
   - Predictions with probabilities
   - Feature importance (RandomForest)
   - Hyperparameters (JSON)
   - Metrics summary
4. **Compares all models** and identifies best performer
5. **Registers best model** to Model Registry as `iris_best_model`
6. **Promotes to Production** stage

### `inference.py` - Model Inference

1. **Loads model** from registry (Production stage)
2. **Makes predictions** on sample input
3. **Displays results** with class probabilities

Sample input: `[5.1, 3.5, 1.4, 0.2]` (sepal length, sepal width, petal length, petal width)

## 📊 Features

**Nested run hierarchy** (parent-child structure)  
**Hyperparameter comparison** across multiple configs  
**Model registry integration** with versioning  
**Comprehensive artifact logging** (8 artifact types per run)  
**Automated best model selection**  
**Production deployment** with stage transitions  
**Detailed metrics** (accuracy, precision, recall, F1)  

## 🎯 Expected Output

### Training Results
- **3 parent runs** (RandomForest, SVC, LogisticRegression)
- **12 child runs** (hyperparameter configurations)
- **Comparison tables** (best per model + detailed view)
- **Registered model** in Production stage

### Model Performance
- Test accuracy: ~0.95-1.00 (varies by configuration)
- Metrics tracked: accuracy, precision, recall, F1 score
- 8 artifacts logged per run

### Inference Output
```
Predicted Class: setosa
Class Probabilities:
  setosa         : 0.9800 (98.00%)
  versicolor     : 0.0150 (1.50%)
  virginica      : 0.0050 (0.50%)
```

## 📈 MLflow UI Navigation

After running `mlflow ui`, navigate to:

1. **Experiments** → `iris_classification_nested_runs`
2. View **parent runs** (3 model families)
3. Expand to see **child runs** (12 configurations)
4. **Models** → `iris_best_model` (registered model)
5. Check **Production** stage for deployed model

## 🔍 Key Concepts Demonstrated

- **Nested Runs**: Parent-child hierarchy for organized experiment tracking
- **Model Registry**: Centralized model versioning and stage management
- **Artifact Logging**: Comprehensive data, metrics, and analysis storage
- **Hyperparameter Tuning**: Systematic comparison of configurations
- **Production Deployment**: Model staging workflow (None → Production)

## 📝 Configuration

Modify these variables in `train.py` to customize:

```python
EXPERIMENT_NAME = "iris_classification_nested_runs"
MODEL_REGISTRY_NAME = "iris_best_model"
test_size = 0.2
random_state = 42
```
---

**Made with MLflow** 🚀 | **Dataset**: Iris (scikit-learn) | **Models**: RandomForest, SVC, LogisticRegression