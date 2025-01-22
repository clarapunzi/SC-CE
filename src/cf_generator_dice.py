"""
Handles counterfactual generation for different models using DiCE, LoRE and other methods.
"""
import os
import time
from typing import Dict, Any, List, Union
import warnings
import pickle
import dice_ml
#from dice_ml.utils import helpers
import pandas as pd
import numpy as np
#import wandb
from .cf_generator_base import CFGeneratorBase
from src.utils import write_time

warnings.filterwarnings("ignore",
    message="X has feature names, but StandardScaler was fitted without feature names")

class DiceCFGenerator(CFGeneratorBase):
    """Handles counterfactual generation for different models using DiCE."""

    def __init__(self, config: Dict,dataset_name:str):
        super().__init__(config,dataset_name)

        # Paths for DiCE components
        self.paths = {
            'data': os.path.join(self.base_path, self.dataset_name+'_dice_data.pkl'),
            'model': os.path.join(self.base_path, 'dice_model.pkl'),
            'explainer': os.path.join(self.base_path, 'dice_explainer.pkl')
        }

        self._dice_data = None
        self._dice_models = {}  # Dictionary to store model-specific DiCE models
        self._explainers = {}   # Dictionary to store model-specific explainers
        self._method = config.get("counterfactuals",
                                     {}).get('dice',
                                              {}).get('method',
                                                      'genetic')
        print(f"Using method: {self._method}")
    def _get_method_name(self) -> str:
        """Return the name of the counterfactual generation method."""
        return 'dice'

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

    def setup(self,
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
        model_path =     os.path.join(self.base_path, 
                f'dice_model_{self.dataset_name}_{model_name}_{self._method}.pkl')
        explainer_path = os.path.join(self.base_path, 
            f'dice_explainer_{self.dataset_name}_{model_name}_{self._method}.pkl')
        print("trying to load explainer model from",explainer_path)
        # Try to load existing components
        self._dice_models[model_name] = self._load_component(model_path)
        self._explainers[model_name] = self._load_component(explainer_path)

        if self._dice_models[model_name] is None or self._explainers[model_name] is None:
            print(f"Setting up DiCE components for {model_name} using method: {self._method}")
            try:
                # Create new components
                self._dice_models[model_name] = dice_ml.Model(model=model, backend="sklearn")
                self._explainers[model_name] = dice_ml.Dice(
                    self._dice_data,
                    self._dice_models[model_name],
                    method=self._method
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
                               num_cf: int = 32,
                               dt_name = 'dataset_name',
                               set_name = "") -> Dict[str, List[Dict[str, Any]]]:
        """Generate counterfactuals for all models and calibration samples."""
        if self._dice_data is None:
            raise ValueError("DiCE not initialized. Call setup first.")

        # Ensure X_calibration is a DataFrame
        if isinstance(X_calibration, np.ndarray):
            X_calibration = pd.DataFrame(
                X_calibration,
                columns=self._dice_data.feature_names
            )

        results = {}
        # Process each model
        for model_name, model in models.items():
            print(f"Generating {num_cf} CFs with with method: ({self._method}) for model: {model_name}")

            # Setup model-specific components if not already done
            self._setup_model_components(model, model_name)

            model_results = []
            start = time.time()
            # Generate counterfactuals for entire dataset
            print(X_calibration)
            cf_result = self._explainers[model_name].generate_counterfactuals(
                X_calibration,
                total_CFs=num_cf,
                desired_class="opposite"
            )
            end = time.time()

            # Process results
            model_results = [{
                'instance_idx': idx,
                'original_instance': X_calibration.iloc[idx:idx+1],
                'true_class': y_calibration.iloc[idx],
                'counterfactuals': cf_result.cf_examples_list[idx].final_cfs_df,
                'metadata': {
                    'success': len(cf_result.cf_examples_list[idx].final_cfs_df) > 0,
                    'num_generated': len(cf_result.cf_examples_list[idx].final_cfs_df)
                }
            } for idx in range(len(X_calibration))]

            # for every instance in the calibration set check how many counterfactuals valid were generated over he num_cf
            for idx in range(len(X_calibration)):
                print("sample",idx,
                      np.round((model.predict(X_calibration.iloc[[idx]])!=model_results[idx]['counterfactuals']["target"].values)
                               .sum()/num_cf,2))
            fancy_time = write_time(end-start)
            print(f"Time to generate counterfactuals: {fancy_time} seconds")
            # append on file the time to generate the counterfactuals
            with open(os.path.join(self.base_path,"time_to_generate_dice.txt"),"a+",encoding='utf-8') as f:
                f.write(f"{model_name} {fancy_time} {num_cf} {self._method} {set_name}\n")
            # Store results for this model
            results[model_name] = model_results

            # Save results locally
            self.save_results(model_results, model_name,
                               dt_name=dt_name,
                               set_name=set_name)

            # Log final metrics to W&B
            # self._log_final_metrics(model_name, model_results)

        return results
