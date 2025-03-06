"""
Handles counterfactual generation for different models using DiCE, LoRE and other methods.
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
    def __init__(self, config: Dict[str, Any],dataset_name:str):
        super().__init__(config,dataset_name)
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
        reference_data[target_name] = reference_data[target_name].astype('category')
        self.dataset = TabularDataset(data=reference_data,
                                      class_name = target_name,
                                      categorial_columns=categorical_features,
                                      ordinal_columns=None)
        #self.encoder = ColumnTransformerEnc(self.dataset)
        #self.surrogate = DecisionTreeSurrogate()
        # To initialize the LORE object, as well as the RaondomGenerator
        # we need to pass the blackbox model
        # self.Lore = Lore(bbox=blackbox_model,...)
        # self.generator = TabularRandomGeneratorLore(self.dataset)

    def _setup_model_components(self, model, model_name,X_calibration=None,y_calibration=None
                                )->None:
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
                                 dt_name: str = 'dataset_name',
                                 set_name:str = '') -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate counterfactuals for given models and instances.
        """
        results = {}
        results_rt = {}
        # Process each model
        for model_name, model in models.items():
            print(f"Generating CFs with with method: ({self._method}) for model: {model_name}")
            start = time.time()
            # Setup model-specific components if not already done
            self._setup_model_components(model, model_name)
            lore = self._explainers[model_name]
            model_results = []
            # to save the rules and trees
            rules_and_trees = []
            # Generate counterfactuals for each instance
            for idx in range(len(X_calibration)):
                instance = X_calibration.iloc[idx]
                cf_result = lore.explain(instance)
                
                # cf has this form:
                # {'x': x, 'rule': rule, 'counterfactuals': self.crules, 'deltas': self.deltas}
                # x is the original instance
                # rule is the rule that was used to generate the counterfactuals
                # counterfactuals are the crules and deltas are the differences with the original rule
                # counterfactuals_samples are the counterfactuals in the original space that are predicted as different from the original instance
                # counterfactuals_predictions

                cf_cf_df = pd.DataFrame(cf_result["counterfactual_samples"])
                cf_cf_df.columns = X_calibration.columns
                cf_cf_df['target'] = cf_result["counterfactual_predictions"]
                
                res_dictionary = {'instance_idx': idx,
                    'original_instance': X_calibration.iloc[idx:idx+1],
                    'true_class': y_calibration.iloc[idx],
                    'counterfactuals': cf_cf_df,
                    'rule': cf_result["rule"],
                    'deltas': cf_result["deltas"],
                    'metadata': {
                    'success':len(cf_result["counterfactual_samples"])
                    }
                }
                # print(len(cf_result["counterfactuals_samples"])/num_cf,"%")
                rules_and_trees.append({"instance_idx": idx,
                                      "tree": lore.surrogate,
                                      "rule": cf_result["rule"]})
                model_results.append(res_dictionary)
            results[model_name] = model_results
            results_rt[model_name] = rules_and_trees
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
            self.save_trees_and_rules(results_rt=rules_and_trees,
                                        model_name=model_name,
                                        dt_name=dt_name,
                                        set_name=set_name)

        return results

    def save_trees_and_rules(self,results_rt, model_name, dt_name, set_name):
        """
        Save the results of the counterfactual generation.
        this is specific to LORE, the only method that generates
        rules and counterfactuals rules as well.
        """
        # Save the dictionary 
        filepath = os.path.join(self.base_path,
                                "rules_and_trees",
                                f"{model_name}_{dt_name}_rules_and_trees_{set_name}.pkl")
        # make sure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath,"wb+") as f:
            pickle.dump(results_rt,f)
            print(f"Saved results to {filepath}")
