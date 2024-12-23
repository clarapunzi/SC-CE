"""
This module contains the ModelTrainer class, 
which is responsible for training and managing multiple models, 
as well as saving and loading them to/from disk.
"""
from typing import Dict, Any
import os
import pickle
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
import wandb
from .inner_cross_val import perform_single_cv, log_cv_results

class ModelTrainer:
    """Handles training and management of multiple models."""

    def __init__(self, config: Dict):
        self.config = config


        # Define model configurations
        self.model_configs = {
            "random_forest": {
                "model": RandomForestClassifier(),
                "params": self.config['models']['rf_params']
            },
            "mlp": {
                "model": MLPClassifier(),
                "params": self.config['models']['mlp_params']
            },
            "xgboost": {
                "model": xgb.XGBClassifier(),
                "params": self.config['models']['xgb_params']
            },
        "lgbm": {
                "model": lgb.LGBMClassifier(),
                "params": self.config['models']['lgbm_params']
            }
        }
        '''
        '''


    def _get_model_path(self, model_name: str, dataset_name: str) -> str:
        """Get path for model saving/loading."""
        return os.path.join(
            self.config['paths']['models'],
            dataset_name,
            f"{model_name}.pkl"
        )

    def _save_model(self, model: Any, model_name: str, dataset_name: str):
        """Save trained model to disk."""
        path = self._get_model_path(model_name, dataset_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {model_name} model to {path}")

    def _load_model(self, model_name: str, dataset_name: str) -> Any:
        """Load trained model from disk."""
        path = self._get_model_path(model_name, dataset_name)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded {model_name} model from {path}")
            return model
        print("No existing model found at", path)
        return None

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str,
        n_folds: int = 10,
        config: Dict = None,
        which_models = "all"
    ) -> Dict[str, Any]:
        """Train all models using nested CV or load if already trained."""
        trained_models = {}
        if which_models != "all":
            self.model_configs = {k: v for k, v in self.model_configs.items() if k in which_models}

        for model_name, model_info in self.model_configs.items():
            print(f"Processing {model_name} for {dataset_name}")

            # Try to load existing model
            model = self._load_model(model_name, dataset_name)

            if model is None:
                print(f"Training new {model_name} model")
                run = wandb.init(
                    project=self.config['wandb']['project_name'],
                    config={"dataset": dataset_name, "model": model_name},
                    )

                # Perform nested cross-validation
                best_model, fold_metrics, fold_params = perform_single_cv(X=X_train,
                                                                          y=y_train,
                                                                          model_info=model_info,
                                                                          dataset_name=dataset_name,
                                                                          n_folds=n_folds)
                run.config.update(fold_params)
                # Log cross-validation results
                log_cv_results(model_name=model_name,
                               fold_metrics=fold_metrics,
                               dataset_name=dataset_name)

                # Evaluate final model on held-out test set
                self._evaluate_on_test_set(model=best_model,
                                           X_test=X_test,
                                           y_test=y_test,
                                           run=run)
                self._save_model(best_model, model_name, dataset_name)
                run.finish()
                # Save the model
                trained_models[model_name] = best_model
            else:
                print(f"Loaded existing {model_name} model")
                trained_models[model_name] = model

        return trained_models

    def _evaluate_on_test_set(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        # model_name: str,
        # dataset_name: str,
        run = None
    ) -> Dict[str, float]:
        """Evaluate the final model on the held-out test set."""
        y_pred = model.predict(X_test)
        metrics = {
            'test_balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'test_f1_score_w': f1_score(y_test, y_pred, average='weighted'),
            'test_accuracy': accuracy_score(y_test, y_pred),
        }
        if run is not None:
            run.log(metrics)

        return metrics
