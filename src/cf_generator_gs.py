"""
Handles counterfactual generation for different models using Growing Spheres.
"""
import os
import time
from typing import Dict, Any, List, Union
import warnings
import pickle
import pandas as pd
import numpy as np
from src.cf_generator_base import CFGeneratorBase
from src.utils import write_time
#import wandb
from src.growingspheres.growingspheres import counterfactuals as cf

warnings.filterwarnings("ignore",
    message="X has feature names, but StandardScaler was fitted without feature names")


class GrowingSpheresCFGenerator(CFGeneratorBase):
    """
    Handles counterfactual generation for different models using Growing Spheres.
    It implements the CFGeneratorBase class.
    """
    def __init__(self, config: Dict[str, Any],dataset_name:str):
        super().__init__(config,dataset_name)
        self.config = config
        self.n_in_layer = config["counterfactuals"]["growingspheres"].get("n_in_layer",200)
        self.first_radius = config["counterfactuals"]["growingspheres"].get("first_radius",1.1)
        self.dicrease_radius = config["counterfactuals"]["growingspheres"].get("dicrease_radius",2.0)
        self.sparse = config["counterfactuals"]["growingspheres"].get("sparse",True)
        self._method = "growingspheres"
    def _get_method_name(self) -> str:
        return "growingspheres"

    def setup(self,
              reference_data: pd.DataFrame,
              continuous_features: List[str],
              categorical_features: List[str],
              #ordinal_features: List[str],
              target_name: str = 'target') -> None:
        """
        Setup the counterfactual generator with reference data and feature information.

        Args:
            reference_data: Training data used as reference for generating counterfactuals
            continuous_features: List of continuous feature names
            categorical_features: List of categorical feature names
            target_name: Name of the target column
        """

        self.target_name = target_name
        self.continuous_features = continuous_features # are all the features that are not categorical, by default all that are not in the categorical_features nor in the ordinal
        self.categorical_features = categorical_features
        # reference_data[target_name] = reference_data[target_name].astype('category')


    def _setup_model_components(self, model, model_name,
                                X_calibration=None,y_calibration=None
                                )->None:
        """
        Setup the model components for growingspheres.
        """
        # growingspheres does not need model-specific components,
        # it is highly practical and this method is a no-op (but needed for implementation).
        pass

    def generate_counterfactuals(self,
                                 models: Dict[str, Any],
                                 X_calibration: Union[pd.DataFrame, np.ndarray],
                                 y_calibration: Union[pd.Series, np.ndarray],
                                 num_cf: int = 32,
                                 dt_name: str = 'dataset_name',
                                 set_name:str = '') -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate counterfactuals for given models and instances.
        """
        results = {}
        # Process each model
        for model_name, model in models.items():
            print(f"Generating CFs with with method: ({self._method}) for model: {model_name}")

            model_results = []
            start = time.time()
            # Generate counterfactuals for each instance
            for idx in range(len(X_calibration)):
                instance = X_calibration.iloc[idx].values.reshape(1, -1)
                CF = cf.CounterfactualExplanation(instance, model.predict, method='GS')
                CF.fit(n_in_layer=self.n_in_layer,
                       first_radius=self.first_radius,
                       dicrease_radius=self.dicrease_radius,
                       sparse=self.sparse,
                       verbose=False,
                       num_enemies=num_cf)

                # cf has the best form ever <3 a numpy array of enemies


                cf_cf_df = pd.DataFrame(CF.enemies)

                cf_cf_df['target'] = model.predict(cf_cf_df)

                res_dictionary = {'instance_idx': idx,
                    'original_instance': X_calibration.iloc[idx:idx+1],
                    'true_class': y_calibration.iloc[idx],
                    'counterfactuals': cf_cf_df,

                }
                model_results.append(res_dictionary)
            results[model_name] = model_results
            end = time.time()
            fancy_time = write_time(end-start)

            for idx in range(len(X_calibration)):
                print("sample",idx,
                      np.round((model.predict(X_calibration.iloc[[idx]])!=model_results[idx]['counterfactuals']["target"].values)
                               .sum()/num_cf,2))
            with open(os.path.join(self.base_path,"time_to_generate_lore.txt"),"a+",encoding='utf-8') as f:
                f.write(f"{model_name} {fancy_time} {num_cf} {self._method} {set_name} {dt_name} {len(X_calibration)}\n")

            # Save the trees, the rules and the counterfactuals
            self.save_results(results=model_results,
                                model_name= model_name,
                                dt_name=dt_name,
                                set_name=set_name)

        return results
