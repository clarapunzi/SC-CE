"""
This module contains the abstract base class for counterfactual generation implementations.
It defines the interface for counterfactual generation methods and provides a common setup for saving results.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
import os
from datetime import datetime

class BaseCounterfactualGenerator(ABC):
    """Abstract base class for counterfactual generation implementations."""
    
    def __init__(self, config: Dict):
        """
        Initialize the counterfactual generator.
        
        Args:
            config: Configuration dictionary containing paths and method-specific settings
        """
        self.config = config
        self.base_path = os.path.join(
            config['paths']['counterfactuals'],
            "cf_generators",
            self._get_method_name()
        )
        os.makedirs(self.base_path, exist_ok=True)

    @abstractmethod
    def _get_method_name(self) -> str:
        """Return the name of the counterfactual generation method."""
        pass

    @abstractmethod
    def setup(self,
             reference_data: pd.DataFrame,
             continuous_features: List[str],
             categorical_features: List[str],
             target_name: str = 'target') -> None:
        """
        Setup the counterfactual generator with reference data and feature information.
        
        Args:
            reference_data: Training data used as reference for generating counterfactuals
            continuous_features: List of continuous feature names
            categorical_features: List of categorical feature names
            target_name: Name of the target column
        """
        pass

    @abstractmethod
    def generate_counterfactuals(self,
                               models: Dict[str, Any],
                               X_calibration: Union[pd.DataFrame, np.ndarray],
                               y_calibration: Union[pd.Series, np.ndarray],
                               cf_method: str,
                               num_cf: int = 32,
                               dt_name: str = 'dataset_name') -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate counterfactuals for given models and instances.
        
        Args:
            models: Dictionary of model name to model object mappings
            X_calibration: Input instances to generate counterfactuals for
            y_calibration: True labels for the input instances
            cf_method: Method to use for counterfactual generation
            num_cf: Number of counterfactuals to generate per instance
            dt_name: Name of the dataset
            
        Returns:
            Dictionary mapping model names to lists of counterfactual results
        """
        pass

    def _save_results(self, results: List[Dict[str, Any]], model_name: str, dt_name: str) -> None:
        """
        Save generated counterfactuals to disk.
        
        Args:
            results: List of counterfactual results
            model_name: Name of the model
            dt_name: Name of the dataset
        """
        save_dir = os.path.join(self.config['paths']['counterfactuals'], model_name, dt_name)
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"cf_results_{timestamp}_{self._get_method_name()}")
        
        np.savez(filepath, cfs=results, allow_pickle=True)
        print(f"Saved results to {filepath}")

    @abstractmethod
    def _setup_model_components(self, model: Any, model_name: str) -> None:
        """
        Setup model-specific components needed for counterfactual generation.
        
        Args:
            model: The model object
            model_name: Name of the model
        """
        pass