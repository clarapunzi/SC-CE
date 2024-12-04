"""
This module contains functions for performing inner cross-validation
 for hyperparameter optimization.
"""
import time
from typing import Dict, Any, Tuple, List
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import wandb
#from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

def create_pipeline(model: Any) -> Pipeline:
    """Create a pipeline with scaling and the model."""
    if isinstance(model, Pipeline):
        steps = [('scaler', StandardScaler())] + model.steps
        return Pipeline(steps)
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

def perform_single_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_info: Dict[str, Any],
    dataset_name: str,
    n_folds: int = 5
) -> Tuple[Any, List[Dict[str, float]], List[Dict[str, Any]]]:
    """
    Perform cross-validation for hyperparameter optimization, then train a final model.
    Returns the final model, validation metrics, and best parameters.
    """
    if isinstance(X, pd.DataFrame):
        X = np.array(X.values)
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = np.array(y.values)
    y = y.ravel()

    # Create base pipeline
    pipeline = create_pipeline(model_info['model'])

    # Perform hyperparameter search using k-fold CV
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42,)

    random_search = RandomizedSearchCV(
        pipeline,
        model_info['params'],
        cv=cv,
        scoring={k:k for k in ['f1_weighted', 'balanced_accuracy', 'accuracy']},
        n_iter=15,
        random_state=42,
        n_jobs=-1,
        refit='f1_weighted'
    )

    start_time = time.time()
    random_search.fit(X, y)
    train_time = time.time() - start_time

    # Log validation metrics for each fold
    fold_metrics = []
    print(random_search.cv_results_.keys())
    

    best_idx = random_search.best_index_
    for fold_idx in range(n_folds):
        val_metric_f1 = random_search.cv_results_["split"+str(fold_idx)+"_test_f1_weighted"]
        val_metric_b_acc = random_search.cv_results_["split"+str(fold_idx)+"_test_balanced_accuracy"]
        val_metric_acc = random_search.cv_results_["split"+str(fold_idx)+"_test_accuracy"]
        print(f"Fold {fold_idx} - F1: {val_metric_f1}, B.Acc: {val_metric_b_acc}, Acc: {val_metric_acc}")
        metrics = {
            'fold': fold_idx,
            'f1_weighted': val_metric_f1,
            'balanced_accuracy': val_metric_b_acc,
            'accuracy': val_metric_acc,
            'train_time': train_time
        }

        # Log metrics for this validation fold
        wandb.log({
            "fold": fold_idx,
            f"{dataset_name}/f1_weighted_{fold_idx}": val_metric_f1[best_idx],
            f"{dataset_name}/balanced_accuracy_{fold_idx}": val_metric_b_acc[best_idx],
            f"{dataset_name}/accuracy_{fold_idx}": val_metric_acc[best_idx],
        })
        fold_metrics.append(metrics)
    wandb.log({
            f"{dataset_name}/validation_time": train_time
        })


    # Get best hyperparameters
    best_params = random_search.best_params_

    # Train final model with best hyperparameters on full dataset
    final_model = create_pipeline(model_info['model'])
    final_model.set_params(**best_params)
    final_model.fit(X, y)

    return final_model, fold_metrics, best_params
def log_cv_results(
    model_name: str,
    fold_metrics: List[Dict[str, float]],
    dataset_name: str
):
    """Log cross-validation results to W&B."""
    # Convert metrics to pandas for easier analysis
    metrics_df = pd.DataFrame(fold_metrics)

    # Create box plots
    fig_b_acc = px.box(metrics_df, y='balanced_accuracy',
                     title=f'{model_name} Balanced Accuracy Distribution')
    fig_f1 = px.box(metrics_df, y='f1_weighted',
                    title=f'{model_name} F1 Score Distribution')
    fig_acc = px.box(metrics_df, y='accuracy',
                     title=f'{model_name} Accuracy Distribution')

    wandb.log({
        f"{dataset_name}/figure_balanced_accuracy_distribution": fig_b_acc,
        f"{dataset_name}/figure_f1_distribution": fig_f1,
        f"{dataset_name}/figure_accuracy_distribution": fig_acc,
    })
