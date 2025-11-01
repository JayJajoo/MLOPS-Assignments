"""
MLflow Nested Runs: Iris Classification
Models: RandomForest, SVC, LogisticRegression
Each model tested with multiple hyperparameter configurations
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import pandas as pd
import numpy as np
import json
import os
import shutil

# ==========================================
# CONFIGURATION
# ==========================================

mlflow.set_tracking_uri("mlruns")
EXPERIMENT_NAME = "iris_classification_nested_runs"
mlflow.set_experiment(EXPERIMENT_NAME)
MODEL_REGISTRY_NAME = "iris_best_model"

# ==========================================
# DATA PREPARATION
# ==========================================

def prepare_iris_data():
    """Load and split Iris dataset"""
    print("\n" + "="*70)
    print("Loading Iris Dataset")
    print("="*70)
    
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create DataFrames
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test
    
    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {feature_names}")
    print(f"  Classes: {list(target_names)}")
    
    return X_train, X_test, y_train, y_test, train_df, test_df, feature_names, target_names


# ==========================================
# MODEL CONFIGURATIONS
# ==========================================

def get_model_configurations():
    """
    Define 3 models with different hyperparameter sets
    """
    
    configurations = {
        
        "RandomForest": [
            {
                "model": RandomForestClassifier(
                    n_estimators=50,
                    max_depth=5,
                    min_samples_split=2,
                    random_state=42
                ),
                "params": {
                    "n_estimators": 50,
                    "max_depth": 5,
                    "min_samples_split": 2,
                    "criterion": "gini"
                }
            },
            {
                "model": RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=4,
                    random_state=42
                ),
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 4,
                    "criterion": "gini"
                }
            },
            {
                "model": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=2,
                    random_state=42
                ),
                "params": {
                    "n_estimators": 200,
                    "max_depth": 15,
                    "min_samples_split": 2,
                    "criterion": "gini"
                }
            },
            {
                "model": RandomForestClassifier(
                    n_estimators=150,
                    max_depth=None,
                    min_samples_split=5,
                    criterion="entropy",
                    random_state=42
                ),
                "params": {
                    "n_estimators": 150,
                    "max_depth": None,
                    "min_samples_split": 5,
                    "criterion": "entropy"
                }
            },
        ],
        
        "SVC": [
            {
                "model": SVC(
                    C=0.1,
                    kernel='linear',
                    random_state=42,
                    probability=True
                ),
                "params": {
                    "C": 0.1,
                    "kernel": "linear",
                    "gamma": "scale"
                }
            },
            {
                "model": SVC(
                    C=1.0,
                    kernel='rbf',
                    random_state=42,
                    probability=True
                ),
                "params": {
                    "C": 1.0,
                    "kernel": "rbf",
                    "gamma": "scale"
                }
            },
            {
                "model": SVC(
                    C=10.0,
                    kernel='rbf',
                    gamma='auto',
                    random_state=42,
                    probability=True
                ),
                "params": {
                    "C": 10.0,
                    "kernel": "rbf",
                    "gamma": "auto"
                }
            },
            {
                "model": SVC(
                    C=5.0,
                    kernel='poly',
                    degree=3,
                    random_state=42,
                    probability=True
                ),
                "params": {
                    "C": 5.0,
                    "kernel": "poly",
                    "degree": 3,
                    "gamma": "scale"
                }
            },
        ],
        
        "LogisticRegression": [
            {
                "model": LogisticRegression(
                    C=0.1,
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42
                ),
                "params": {
                    "C": 0.1,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "max_iter": 1000
                }
            },
            {
                "model": LogisticRegression(
                    C=1.0,
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42
                ),
                "params": {
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "max_iter": 1000
                }
            },
            {
                "model": LogisticRegression(
                    C=10.0,
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42
                ),
                "params": {
                    "C": 10.0,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "max_iter": 1000
                }
            },
            {
                "model": LogisticRegression(
                    C=1.0,
                    penalty='l1',
                    solver='saga',
                    max_iter=1000,
                    random_state=42
                ),
                "params": {
                    "C": 1.0,
                    "penalty": "l1",
                    "solver": "saga",
                    "max_iter": 1000
                }
            },
        ],
    }
    
    return configurations


# ==========================================
# TRAINING WITH NESTED RUNS
# ==========================================

def train_child_run(model, model_name, config_idx, hyperparams,
                    X_train, X_test, y_train, y_test,
                    train_df, test_df, target_names):
    """
    Train a single hyperparameter configuration (CHILD RUN)
    """
    
    run_name = f"{model_name}_config_{config_idx+1}"
    
    with mlflow.start_run(run_name=run_name, nested=True) as child_run:
        
        # Enable autologging
        mlflow.sklearn.autolog(log_models=False, silent=True)
        
        # Log hyperparameters
        mlflow.log_params(hyperparams)
        mlflow.log_param("config_index", config_idx + 1)
        
        # Set tags
        mlflow.set_tags({
            "model_type": model_name,
            "config": f"config_{config_idx+1}",
            "dataset": "iris"
        })
        
        # Train model
        print(f"    Training {run_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        # Log metrics
        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        }
        mlflow.log_metrics(metrics)
        
        # Log artifacts
        os.makedirs("temp_artifacts", exist_ok=True)
        
        # 1. Save training and test datasets
        train_path = "temp_artifacts/train_data.csv"
        test_path = "temp_artifacts/test_data.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        mlflow.log_artifact(train_path, "datasets")
        mlflow.log_artifact(test_path, "datasets")
        
        # 2. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        cm_df = pd.DataFrame(
            cm,
            index=[f"True_{name}" for name in target_names],
            columns=[f"Pred_{name}" for name in target_names]
        )
        cm_path = "temp_artifacts/confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(cm_path, "metrics")
        
        # 3. Classification Report
        class_report = classification_report(
            y_test, y_pred_test,
            target_names=target_names,
            output_dict=True
        )
        report_path = "temp_artifacts/classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(class_report, f, indent=4)
        mlflow.log_artifact(report_path, "metrics")
        
        # 4. Predictions with probabilities
        pred_df = pd.DataFrame(y_pred_proba, columns=[f"prob_{name}" for name in target_names])
        pred_df['actual'] = y_test
        pred_df['predicted'] = y_pred_test
        pred_df['correct'] = y_test == y_pred_test
        pred_path = "temp_artifacts/predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path, "predictions")
        
        # 5. Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            fi_df = pd.DataFrame({
                'feature': train_df.columns[:-1],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            fi_path = "temp_artifacts/feature_importance.csv"
            fi_df.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path, "analysis")
        
        # 6. Hyperparameters as JSON
        params_path = "temp_artifacts/hyperparameters.json"
        with open(params_path, 'w') as f:
            json.dump(hyperparams, f, indent=4)
        mlflow.log_artifact(params_path, "config")
        
        # 7. Metrics summary as JSON
        metrics_path = "temp_artifacts/metrics_summary.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(metrics_path, "metrics")
        
        # Log the model
        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=X_train[:5],
            signature=mlflow.models.infer_signature(X_train, y_train)
        )
        
        # Clean up
        if os.path.exists("temp_artifacts"):
            shutil.rmtree("temp_artifacts")
        
        print(f"      ✓ Accuracy: {test_accuracy:.4f} | F1: {test_f1:.4f}")
        
        return {
            "run_id": child_run.info.run_id,
            "run_name": run_name,
            "model_type": model_name,
            "config_idx": config_idx,
            "metrics": metrics,
            "model": model
        }


def train_parent_run(model_name, configurations,
                     X_train, X_test, y_train, y_test,
                     train_df, test_df, target_names):
    """
    Train all configurations for a model (PARENT RUN with CHILD RUNS)
    """
    
    print(f"\n{'='*70}")
    print(f"Training {model_name} with {len(configurations)} hyperparameter sets")
    print(f"{'='*70}")
    
    with mlflow.start_run(run_name=f"{model_name}_Experiments", nested=False) as parent_run:
        
        # Log parent metadata
        mlflow.set_tags({
            "model_family": model_name,
            "num_configurations": len(configurations),
            "experiment_type": "hyperparameter_tuning"
        })
        
        # Store child results
        child_results = []
        
        # Train each configuration
        for idx, config in enumerate(configurations):
            result = train_child_run(
                model=config["model"],
                model_name=model_name,
                config_idx=idx,
                hyperparams=config["params"],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                train_df=train_df,
                test_df=test_df,
                target_names=target_names
            )
            child_results.append(result)
        
        # Find best configuration
        best_config = max(child_results, key=lambda x: x["metrics"]["test_accuracy"])
        
        # Log parent metrics (aggregated)
        mlflow.log_metrics({
            "best_test_accuracy": best_config["metrics"]["test_accuracy"],
            "best_test_f1": best_config["metrics"]["test_f1"],
            "best_test_precision": best_config["metrics"]["test_precision"],
            "best_test_recall": best_config["metrics"]["test_recall"],
            "avg_test_accuracy": np.mean([r["metrics"]["test_accuracy"] for r in child_results]),
            "std_test_accuracy": np.std([r["metrics"]["test_accuracy"] for r in child_results]),
            "min_test_accuracy": np.min([r["metrics"]["test_accuracy"] for r in child_results]),
            "max_test_accuracy": np.max([r["metrics"]["test_accuracy"] for r in child_results])
        })
        
        mlflow.log_params({
            "best_config_idx": best_config["config_idx"] + 1,
            "best_run_id": best_config["run_id"]
        })
        
        print(f"\nSummary for {model_name}:")
        print(f"    Best Config: #{best_config['config_idx']+1}")
        print(f"    Best Accuracy: {best_config['metrics']['test_accuracy']:.4f}")
        print(f"    Avg Accuracy: {np.mean([r['metrics']['test_accuracy'] for r in child_results]):.4f}")
        print(f"    Std Accuracy: {np.std([r['metrics']['test_accuracy'] for r in child_results]):.4f}")
        
        return {
            "parent_run_id": parent_run.info.run_id,
            "model_name": model_name,
            "child_results": child_results,
            "best_config": best_config
        }


# ==========================================
# COMPARISON AND ANALYSIS
# ==========================================

def create_comparison_table(all_results):
    """Compare best configuration from each model type"""
    
    comparison_data = []
    
    for result in all_results:
        best = result["best_config"]
        comparison_data.append({
            "Model": result["model_name"],
            "Best Config": f"#{best['config_idx']+1}",
            "Test Accuracy": f"{best['metrics']['test_accuracy']:.4f}",
            "Test F1": f"{best['metrics']['test_f1']:.4f}",
            "Precision": f"{best['metrics']['test_precision']:.4f}",
            "Recall": f"{best['metrics']['test_recall']:.4f}",
            "Run ID": best['run_id'][:8]
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values("Test Accuracy", ascending=False)
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON - BEST FROM EACH TYPE")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print()
    
    return df


def create_detailed_comparison(all_results):
    """Show all configurations from all models"""
    
    detailed_data = []
    
    for result in all_results:
        for child in result["child_results"]:
            detailed_data.append({
                "Model": result["model_name"],
                "Config": f"#{child['config_idx']+1}",
                "Test Accuracy": f"{child['metrics']['test_accuracy']:.4f}",
                "Test F1": f"{child['metrics']['test_f1']:.4f}",
                "Run ID": child['run_id'][:8]
            })
    
    df = pd.DataFrame(detailed_data)
    df_sorted = df.sort_values("Test Accuracy", ascending=False)
    
    print(f"\n{'='*70}")
    print("DETAILED COMPARISON - ALL CONFIGURATIONS")
    print(f"{'='*70}")
    print(df_sorted.to_string(index=False))
    print()
    
    return df_sorted


# ==========================================
# MODEL REGISTRY
# ==========================================

def register_best_model(best_overall):
    """Register the best model to MLflow Model Registry"""
    
    print(f"\n{'='*70}")
    print(f"Registering Best Model to Model Registry")
    print(f"{'='*70}")
    
    model_uri = f"runs:/{best_overall['run_id']}/model"
    
    # Register model
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_REGISTRY_NAME
    )
    
    # Add description
    client = MlflowClient()
    client.update_model_version(
        name=MODEL_REGISTRY_NAME,
        version=registered_model.version,
        description=f"Best {best_overall['model_type']} model - Config #{best_overall['config_idx']+1} - Accuracy: {best_overall['metrics']['test_accuracy']:.4f}"
    )
    
    # Promote to Production
    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=registered_model.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"    Model Type: {best_overall['model_type']}")
    print(f"    Configuration: #{best_overall['config_idx']+1}")
    print(f"    Registry Name: {MODEL_REGISTRY_NAME}")
    print(f"    Version: {registered_model.version}")
    print(f"    Stage: Production")
    print(f"    Accuracy: {best_overall['metrics']['test_accuracy']:.4f}")
    print(f"    Run ID: {best_overall['run_id']}")
    
    return registered_model


# ==========================================
# INFERENCE
# ==========================================

def load_and_predict(X_test, y_test, target_names, stage="Production"):
    """Load model from registry and make predictions"""
    
    print(f"\n{'='*70}")
    print(f"INFERENCE - Loading {stage} Model")
    print(f"{'='*70}")
    
    # Load model
    model_uri = f"models:/{MODEL_REGISTRY_NAME}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Get version info
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=[stage])
    version_info = versions[0] if versions else None
    
    print(f"  Model Name: {MODEL_REGISTRY_NAME}")
    print(f"  Version: {version_info.version if version_info else 'Unknown'}")
    print(f"  Stage: {stage}")
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    
    print(f"\nResults:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    F1 Score: {f1:.4f}")
    
    print(f"\nSample Predictions:")
    print(f"  {'-'*65}")
    
    for i in range(min(10, len(predictions))):
        true_class = target_names[y_test[i]]
        pred_class = target_names[predictions[i]]
        confidence = probabilities[i][predictions[i]]
        status = "✓" if y_test[i] == predictions[i] else "✗"
        print(f"  {status} Sample {i+1:2d}: True={true_class:15s} | Pred={pred_class:15s} | Conf={confidence:.3f}")
    
    return predictions, probabilities


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """Main pipeline execution"""
    
    print("="*70)
    print("MLflow Nested Runs: Iris Classification")
    print("   Models: RandomForest, SVC, LogisticRegression")
    print("="*70)
    
    # Prepare data
    X_train, X_test, y_train, y_test, train_df, test_df, feature_names, target_names = prepare_iris_data()
    
    # Get model configurations
    configurations = get_model_configurations()
    
    total_configs = sum(len(configs) for configs in configurations.values())
    print(f"\nTotal Configurations: {total_configs}")
    print(f"  RandomForest: {len(configurations['RandomForest'])} configs")
    print(f"  SVC: {len(configurations['SVC'])} configs")
    print(f"  LogisticRegression: {len(configurations['LogisticRegression'])} configs")
    
    # Train all models with nested runs
    all_results = []
    
    for model_name, model_configs in configurations.items():
        result = train_parent_run(
            model_name=model_name,
            configurations=model_configs,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_df=train_df,
            test_df=test_df,
            target_names=target_names
        )
        all_results.append(result)
    
    # Create comparison tables
    summary_df = create_comparison_table(all_results)
    detailed_df = create_detailed_comparison(all_results)
    
    # Find overall best model
    best_overall = max(
        [r["best_config"] for r in all_results],
        key=lambda x: x["metrics"]["test_accuracy"]
    )
    
    print(f"\n{'='*70}")
    print(f"OVERALL BEST MODEL")
    print(f"{'='*70}")
    print(f"  Model Type: {best_overall['model_type']}")
    print(f"  Configuration: #{best_overall['config_idx']+1}")
    print(f"  Test Accuracy: {best_overall['metrics']['test_accuracy']:.4f}")
    print(f"  Test F1 Score: {best_overall['metrics']['test_f1']:.4f}")
    print(f"  Test Precision: {best_overall['metrics']['test_precision']:.4f}")
    print(f"  Test Recall: {best_overall['metrics']['test_recall']:.4f}")
    print(f"  Run ID: {best_overall['run_id']}")
    
    # Register best model
    registered_model = register_best_model(best_overall)
    
    # Perform inference
    load_and_predict(X_test, y_test, target_names, stage="Production")
    
    # Display MLflow UI info
    print(f"\n{'='*70}")
    print("View Results in MLflow UI")
    print(f"{'='*70}")
    print(f"\nRun this command:")
    print(f"    mlflow ui --backend-store-uri mlruns")
    print(f"\n  Then open: http://localhost:5000")
    print(f"\n  Experiment: {EXPERIMENT_NAME}")
    print(f"  Registered Model: {MODEL_REGISTRY_NAME}")
    print(f"  Total Runs: {total_configs + len(configurations)} ({len(configurations)} parent + {total_configs} child)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()