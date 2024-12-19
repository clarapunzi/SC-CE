"""
Handles counterfactual generation for different models using DiCE, LoRE and other methods.
"""
import os
from typing import Dict, Any, List, Union
import warnings
import pickle
from datetime import datetime
#from dice_ml.utils import helpers
import pandas as pd
import numpy as np
from src.cf_generator_base import CFGeneratorBase
#import wandb
#############################################
################# LORE imports ##############
#############################################
from src.LORE_sa.lore_sa import sklearn_classifier_bbox


from src.LORE_sa.lore_sa.dataset import TabularDataset
from src.LORE_sa.lore_sa.encoder_decoder import ColumnTransformerEnc
from src.LORE_sa.lore_sa.lore import (TabularRandomGeneratorLore,
                                        TabularGeneticGeneratorLore,
                                        TabularRandGenGeneratorLore)
# from src.LORE_sa.lore_sa.surrogate import DecisionTreeSurrogate

warnings.filterwarnings("ignore",
    message="X has feature names, but StandardScaler was fitted without feature names")

class LoreCFGenerator(CFGeneratorBase):
    """
    Handles counterfactual generation for different models using Lore.
    It implements the CFGeneratorBase class.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.dataset = None
        self.encoder = None
        self.surrogate = None
        self.generator = None
        self._explainers = {}
        self._method = config.get("counterfactuals",
                                     {}).get('lore',
                                              {}).get('method',
                                                      'random')

    def _get_method_name(self) -> str:
        return "lore"

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
        reference_data[target_name] = reference_data[target_name].astype('category')
        self.dataset = TabularDataset(reference_data,
                                      target_name,
                                      continuous_features,
                                      categorical_features)
        #self.encoder = ColumnTransformerEnc(self.dataset)
        #self.surrogate = DecisionTreeSurrogate()
        # To initialize the LORE object, as well as the RaondomGenerator
        # we need to pass the blackbox model
        # self.Lore = Lore(bbox=blackbox_model,...)
        # self.generator = TabularRandomGeneratorLore(self.dataset)

    def _setup_model_components(self, model, model_name):
        """
        Setup the model components for LORE.
        The Lore object and the blackbox model are initialized here.
        """
        if self._method == 'random':
            self._explainers[model_name] = TabularRandomGeneratorLore(
                        bbox=sklearn_classifier_bbox.sklearnBBox(model),
                         dataset=self.dataset)
                         # encoder=self.encoder,
                         # generator=self.generator,
                         # surrogate=self.surrogate)
        if self._method == 'genetic':
            self._explainers[model_name] = TabularGeneticGeneratorLore(
                        bbox=sklearn_classifier_bbox.sklearnBBox(model),
                         dataset=self.dataset)
        if self._method == 'random_gen':
            self._explainers[model_name] = TabularRandGenGeneratorLore(
                        bbox=sklearn_classifier_bbox.sklearnBBox(model),
                         dataset=self.dataset)

    def generate_counterfactuals(self,
                                 models: Dict[str, Any],
                                 X_calibration: Union[pd.DataFrame, np.ndarray],
                                 y_calibration: Union[pd.Series, np.ndarray],
                                 num_cf: int = 32,
                                 dt_name: str = 'dataset_name') -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate counterfactuals for given models and instances.
        """
        results = {}
        results_rt = {}
        # Process each model
        for model_name, model in models.items():
            print(f"Generating CFs with with method: ({self._method}) for model: {model_name}")
            # Setup model-specific components if not already done
            self._setup_model_components(model, model_name)
            lore = self._explainers[model_name]
            model_results = []
            # to save the rules and trees
            rules_and_trees = []
            # Generate counterfactuals for each instance
            for idx in range(len(X_calibration)):
                instance = X_calibration.iloc[idx]
                print(idx/len(X_calibration))
                cf_result = lore.explain(instance)
                print(cf_result)
                # cf has this form:
                # {'x': x, 'rule': rule, 'counterfactuals': self.crules, 'deltas': self.deltas}
                # x is the original instance
                # rule is the rule that was used to generate the counterfactuals
                # counterfactuals are the crules and deltas are the distances

                
                res_dictionary = {'instance_idx': idx,
                    'original_instance': X_calibration.iloc[idx:idx+1],
                    'true_class': y_calibration.iloc[idx],
                    'counterfactuals': cf_result["counterfactuals"],
                    'rule': cf_result["rule"],
                    'deltas': cf_result["deltas"],
                    'metadata': {
                        'success':len(cf_result["deltas"])
                    }
                }
                rules_and_trees.append({"instance_idx": idx,
                                      "original_instance": X_calibration.iloc[idx:idx+1],
                                     "tree": lore.surrogate,
                                      "rule": cf_result["rule"]})
                model_results.append(res_dictionary)
            results[model_name] = model_results
            results_rt[model_name] = rules_and_trees
            # Save the trees, the rules and the counterfactuals
            self._save_results(results=model_results,
                               results_rt=rules_and_trees,
                               model_name=model_name,
                               dt_name=dt_name)

        return results

    def _save_results(self, results,results_rt, model_name, dt_name):
        """
        Save the results of the counterfactual generation.
        """
        # Save the trees
        save_dir = os.path.join(self.config['paths']['counterfactuals'], model_name, dt_name)
        # os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"cf_results_{self._get_method_name()}_{self._method}_{timestamp}_")
        filepath = filepath + "_rules_and_trees"
        print(f"Saving results to {filepath}",type(results_rt))
        print("Saving the others also",type(results))
        # Save the counterfactuals
        #super()._save_results(results, model_name, dt_name)