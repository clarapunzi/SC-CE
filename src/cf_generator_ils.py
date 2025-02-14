"""
Handles counterfactual generation for different models using Interpretable Latent Space (ILS).
"""
import os
from typing import Dict, Any, List, Union
import warnings
#from dice_ml.utils import helpers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from src.cf_generator_base import CFGeneratorBase
from src.utils import write_time
import src.cp_ils.cpils as cpils
warnings.filterwarnings("ignore",
                        message="X has feature names, but StandardScaler was fitted without feature names")
class IlsCFGenerator(CFGeneratorBase):
    """
    Handles counterfactual generation for different models using ILS.
    It implements the CFGeneratorBase class.
    """
    def __init__(self, config: Dict[str, Any],dataset_name:str):
        super().__init__(config,dataset_name)
        self.config = config
        self.dataset = None

        self.latent = None

        self._explainers = {}
        self._method = config.get("counterfactuals",
                                     {}).get('ils',
                                              {}).get('method',
                                                      'latent')
        self.dimensions = config.get("counterfactuals",
                                        {}).get('ils',
                                                {}).get('dimentions',
                                                        2)

    def _get_method_name(self) -> str:
        return "ils"

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
        self.dataset = pd.DataFrame(reference_data)
        self.continuous_features = continuous_features
        self.target_name = target_name
        self.categorical_features = categorical_features


    def _setup_model_components(self, model, model_name, X_calibration, y_calibration) -> None:
        """
        Setup the model components for LORE.
        The Lore object and the blackbox model are initialized here.
        """

        explainer_path = os.path.join(self.base_path,
            f'ils_explainer_{self.dataset_name}_{model_name}_Ls_{self.dimensions}.pkl')
        print("trying to load explainer model from",explainer_path)
        # Try to load existing components

        self._explainers[model_name] = self._load_component(explainer_path)
        X_train = self.dataset.drop(columns=[self.target_name])
        y_train = self.dataset[self.target_name].values.reshape(-1,1)
        if self._explainers[model_name] is None:
            print("Explainer model not found. Training new ILS.")
            self._explainers[model_name] = cpils.CP_ILS(model.predict,
                              model.predict_proba,
                              latent_dim=self.dimensions,
                              max_epochs=2000,
                              early_stopping=50)
            latent = self._explainers[model_name]
            idx_num_cat = [] # indices of numerical and categorical features grouped by "one_hot"
            losses = latent.fit((X_train,X_calibration), idx_num_cat, seed=42)
            self._save_component(self._explainers[model_name], explainer_path)
            plt.plot(losses[0], label='train')
            plt.plot(losses[1], label='calibration')
            plt.title(f"Losses of ILS {model_name} on {self.dataset_name}")
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.yscale('log')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join('plots',
                                    self.dataset_name,
                                    f'losses_{model_name}_ils_{self.dimensions}.pdf'))
            plt.close()
        else:
            print("Explainer model found. Loading existing ILS.")
            latent = self._explainers[model_name]
            print("Explainer model loaded.")
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
            # Setup model-specific components if not already done
            start = time.time()
            self._setup_model_components(model, model_name, X_calibration, y_calibration)
            latent = self._explainers[model_name]
            model_results = []
            # change them all if needed
            change_f =  list(range(X_calibration.values.shape[1]))
            # number of features to change (change all if needed)
            max_f = len(change_f)

            # Generate counterfactuals for each instance

            for idx,instance in X_calibration.iterrows():
                # instace has to be a dataframe
                instance = instance.to_frame().T
                label = y_calibration.loc[idx]
  
                assert instance.shape[0] == 1,str(instance)+" has more than one row:"+str(instance.shape)
                #print(f"Generating CFs for instance {idx}",instance)
                cf_result = latent.get_counterfactuals(df_test=instance,
                                                    features_to_change=change_f,
                                                    max_features_to_change=max_f,
                                                    max_steps=128,
                                                    n_cfs=num_cf,
                                                    n_feats_sampled=max_f,
                                                    topn_to_check=32,
                                                    seed=69)
                #print("---"*4)
                # cf_result is a couple.
                #  The first element is the dataframe with the counterfactuals in the original space
                # the second element is the dataframe with the counterfactuals in the latent space
                # we save them BOTH in the results dictionary
                #print(y_calibration.loc[idx],label)
                #print("predicted as:",model.predict(instance))
                res_dictionary = {'instance_idx': idx,
                        'original_instance': instance,
                        'true_class': label,
                        'counterfactuals': cf_result[0],
                        'latent_counterfactuals': cf_result[1],
                    }

                model_results.append(res_dictionary)

                # here we have to see and check the counterfactuals dataframe
            end = time.time()
            fancy_time = write_time(end-start)
            results[model_name] = model_results
            with open(os.path.join(self.base_path,"time_to_generate_dice.txt"),"a+",encoding='utf-8') as f:
                f.write(f"{model_name} {fancy_time} {num_cf} {self._method} {set_name} {dt_name} {len(X_calibration)} dimentions_{self.dimensions}\n")
        
            self.save_results(results=model_results,
                                model_name= model_name,
                                dt_name=dt_name,
                                set_name=set_name)
        print(f"Time to generate counterfactuals: {fancy_time} seconds")
        # append on file the time to generate the counterfactuals

        # Store results for this model

        # Save results locally
        return results
