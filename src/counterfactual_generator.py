"""
Handles counterfactual generation for different models using DiCE, LoRE and other methods.
"""
import os
from typing import Dict, Any, List, Union
import warnings
import pickle
from datetime import datetime
from tqdm import tqdm
import dice_ml
#from dice_ml.utils import helpers
import pandas as pd
import numpy as np
#import wandb

warnings.filterwarnings("ignore",
    message="X has feature names, but StandardScaler was fitted without feature names")

class CounterfactualGenerator:
    """Handles counterfactual generation for different models using DiCE."""

    def __init__(self, config: Dict):
        self.config = config
        self.base_path = os.path.join(config['paths']['counterfactuals'],
                                      "cf_generators",
                                      'dice')
        os.makedirs(self.base_path, exist_ok=True)

        # Paths for DiCE components
        self.paths = {
            'data': os.path.join(self.base_path, 'dice_data.pkl'),
            'model': os.path.join(self.base_path, 'dice_model.pkl'),
            'explainer': os.path.join(self.base_path, 'dice_explainer.pkl')
        }

        self._dice_data = None
        self._dice_models = {}  # Dictionary to store model-specific DiCE models
        self._explainers = {}   # Dictionary to store model-specific explainers

    def _save_component(self, component: Any, path: str) -> None:
        """Save a DiCE component to disk."""
        with open(path, 'wb') as f:
            pickle.dump(component, f)
        print(f"Saved component to {path}")

    def _load_component(self, path: str) -> Any:
        """Load a DiCE component from disk."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                component = pickle.load(f)
            print(f"Loaded component from {path}")
            return component
        return None

    def setup_dice(self,
                  reference_data: pd.DataFrame,
                  # feature_names: List[str],
                  continuous_features: List[str],
                  categorical_features: List[str],
                  target_name: str = 'target') -> None:
        """Setup DiCE data object and save it."""
        # Try to load existing DiCE data
        self._dice_data = self._load_component(self.paths['data'])
        print(reference_data.columns)
        
        if self._dice_data is None:
            try:
                # Create new data object for DiCE
                self._dice_data = dice_ml.Data(
                    dataframe=reference_data,
                    continuous_features=continuous_features,
                    categorical_features=categorical_features,
                    outcome_name=target_name
                )
                # Save the data object
                self._save_component(self._dice_data, self.paths['data'])

            except Exception as e:
                print(f"Error setting up DiCE data: {str(e)}")
                raise

    def _setup_model_components(self, model: Any, model_name: str) -> None:
        """Setup and save DiCE model and explainer for a specific model."""
        # Create model-specific paths
        model_path =     " "#os.path.join(self.base_path, f'dice_model_{model_name}.pkl')
        explainer_path = " "#os.path.join(self.base_path, f'dice_explainer_{model_name}.pkl')

        # Try to load existing components
        self._dice_models[model_name] = self._load_component(model_path)
        self._explainers[model_name] = self._load_component(explainer_path)

        if self._dice_models[model_name] is None or self._explainers[model_name] is None:
            method = self.config.get('dice', {}).get('method', 'random')
            print(f"Setting up DiCE components for {model_name} using method: {method}")
            try:
                # Create new components
                self._dice_models[model_name] = dice_ml.Model(model=model, backend="sklearn")
                self._explainers[model_name] = dice_ml.Dice(
                    self._dice_data,
                    self._dice_models[model_name],
                    method=method
                )

                # Save components
                self._save_component(self._dice_models[model_name], model_path)
                self._save_component(self._explainers[model_name], explainer_path)

            except Exception as e:
                print(f"Error setting up DiCE components for {model_name}: {str(e)}")
                raise

    def generate_counterfactuals(self,
                               models: Dict[str, Any],
                               X_calibration: Union[pd.DataFrame, np.ndarray],
                               y_calibration: Union[pd.Series, np.ndarray],
                               cf_method: str,
                               num_cf: int = 32,
                               dt_name = 'dataset_name') -> Dict[str, List[Dict[str, Any]]]:
        """Generate counterfactuals for all models and calibration samples."""
        if self._dice_data is None:
            raise ValueError("DiCE not initialized. Call setup_dice first.")

        # Ensure X_calibration is a DataFrame
        if isinstance(X_calibration, np.ndarray):
            X_calibration = pd.DataFrame(
                X_calibration,
                columns=self._dice_data.feature_names
            )

        results = {}
        # Process each model
        for model_name, model in models.items():
            print(f"Generating counterfactuals for model: {model_name}")

            # Setup model-specific components if not already done
            self._setup_model_components(model, model_name)

            model_results = []

            # Process each instance in the calibration set
            for idx in tqdm(range(len(X_calibration)), desc=f"Processing {model_name}"):
                instance = X_calibration.iloc[idx:idx+1]
                true_class = y_calibration.iloc[idx]
                #try:
                # Generate counterfactuals
                cf_result = self._explainers[model_name].generate_counterfactuals(
                    instance,
                    total_CFs=num_cf,
                    desired_class="opposite"
                )

                # Extract and store results
                result = {
                    'instance_idx': idx,
                    'original_instance': instance,
                    'true_class': true_class,
                    'counterfactuals': cf_result.cf_examples_list[0].final_cfs_df,
                    'metadata': {
                        'success': len(cf_result.cf_examples_list[0].final_cfs_df) > 0,
                        'num_generated': len(cf_result.cf_examples_list[0].final_cfs_df)
                    }
                }

                model_results.append(result)

                # Log to W&B periodically
                #if idx % 10 == 0:
                #    self._log_progress(model_name, model_results[-10:], idx)

            # Store results for this model
            results[model_name] = model_results

            # Save results locally
            self._save_results(model_results, model_name+"_"+cf_method, dt_name=dt_name)

            # Log final metrics to W&B
            # self._log_final_metrics(model_name, model_results)

        return results

    def _save_results(self, results: List[Dict[str, Any]], model_name: str, dt_name:str) -> None:
        """Save counterfactual results locally."""
        save_dir = os.path.join(self.config['paths']['counterfactuals'], model_name,dt_name)
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"cf_results_{timestamp}")

        np.savez(filepath, cfs=results, allow_pickle=True)
        print(f"Saved results to {filepath}")
