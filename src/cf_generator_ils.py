"""
Handles counterfactual generation for different models using Interpretable Latent Space (ILS).
"""
import os
from typing import Dict, Any, List, Union
import warnings
#from dice_ml.utils import helpers
import pandas as pd
import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import matplotlib.pyplot as plt

from src.cf_generator_base import CFGeneratorBase
from src.utils import write_time
import src.cp_ils.cpils as cpils


warnings.filterwarnings("ignore",
                        message="X has feature names, but StandardScaler was fitted without feature names")
class CPILSSklearnWrapper(BaseEstimator,TransformerMixin):
    """
    Scikit-learn compatible wrapper for CP_ILS that preserves original data separation.
    Assumes training and calibration data are provided separately in a second wrapper.
    This is needed since the original CP_ILS implementation does not support sklearn's fit(X, y) interface, nor other sklearn methods.
    """
    def __init__(self,
                  latent_dim=2,
                 max_epochs=2000, early_stopping=50, batch_size=32,
                 learning_rate=0.001, sigma=1.0):
        # Store all initialization parameters
        self.latent_dim = latent_dim
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.latent_model = None
        # self.base_model_ = model
        # self.X_validation_ = X_calibration
    def fit(self, X, y=None):
        """
        Fits the CP_ILS model using the stored training and calibration data.
        X and y parameters are ignored but kept for sklearn compatibility.
        """
        # Initialize the latent model with current hyperparameters
        self.latent_model = cpils.CP_ILS(
            self.base_model_.predict,
            self.base_model_.predict_proba,
            latent_dim=self.latent_dim,
            max_epochs=self.max_epochs,
            early_stopping=self.early_stopping,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            sigma=self.sigma
        )

        # Fit using stored training and calibration data
        self.idx_cat = []  # Assume all features are numerical
        self.losses_ = self.latent_model.fit(
            (X, self.X_validation_),
            self.idx_cat,
            seed=42
        )

        # You might want to use the validation loss for scoring
        self.score_ = -self.losses_[1][-1]
        return self

    def score(self, X, y=None):
        """
        Returns the negative mean loss on the calibration set.
        X and y parameters are ignored but kept for sklearn compatibility.
        """

        Z = self.latent_model.encode(X)
        loss = self.latent_model.kld_loss_function(X, Z, self.idx_cat, self.sigma)
        return loss

    def transform(self, X):
        """
        Returns the latent representation of the input data.
        """
        return self.latent_model.transform(X)
    def get_counterfactuals(self, df_test, features_to_change, max_features_to_change,
                                max_steps=50, n_cfs=-1, n_feats_sampled=5, topn_to_check=5, seed=42):
        """
        Returns the counterfactuals for the input data.
        """
        return self.latent_model.get_counterfactuals(df_test, features_to_change, max_features_to_change,
                                max_steps, n_cfs, n_feats_sampled, topn_to_check, seed)
    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return {
            #'base_model': self.base_model_,
            #'X_validation': self.X_validation_,

            'latent_dim': self.latent_dim,
            'max_epochs': self.max_epochs,
            'early_stopping': self.early_stopping,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'sigma': self.sigma
        }

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
        self.search_best = config.get("counterfactuals",
                                          {}).get('ils',
                                                    {}).get('search_best',
                                                            True)

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
            f'ils_explainer_{self.dataset_name}_{model_name}.pkl')
        print("trying to load explainer model from",explainer_path)
        # Try to load existing component
        self._explainers[model_name] = self._load_component(explainer_path)
        
        X_train = self.dataset.drop(columns=[self.target_name])
        y_train = self.dataset[self.target_name].values.reshape(-1,1)
        if self._explainers[model_name] is None:
            print("Explainer model not found. Training new ILS.")
            if not self.search_best:
                latent = CPILSSklearnWrapper(model, X_calibration, latent_dim=2,
                                            max_epochs=2000, early_stopping=50, batch_size=32,
                                            learning_rate=0.001, sigma=1.0)

                idx_num_cat = [] # indices of numerical and categorical features grouped by "one_hot"
                #losses = latent.fit((X_train,X_calibration), idx_num_cat, seed=42)
                losses = latent.fit(X=X_train)

            else:
                print("performing hyperparameter search")
                param_grid = {
                    'latent_dim': [2, 3, 4],
                    'batch_size': [4, 8, 16, 32, 64,128],
                    'learning_rate': [0.0001, 0.001, 0.005, 0.008,0.01],
                    'sigma': [0.5, 1.0, 2.0]
                }

                # Initialize the wrapper with the model and data
                class DoubleWrapper(CPILSSklearnWrapper):
                    # takse the same parameters as the original wrapper
                    def __init__(self,latent_dim=2, 
                            max_epochs=2000, early_stopping=70, batch_size=32, 
                            learning_rate=0.001, sigma=1.0):
                        self.base_model_ = model
                        self.X_validation_ = X_calibration
                        super().__init__(latent_dim=latent_dim,
                                         max_epochs=max_epochs,
                                            early_stopping=early_stopping,
                                            batch_size=batch_size,
                                            learning_rate=learning_rate,
                                            sigma=sigma)
                wrapper = DoubleWrapper()
                from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold
                stratified_cv = StratifiedKFold(
                    n_splits=3, 
                    shuffle=True, 
                    random_state=42
                )
                # Create GridSearchCV object
                grid_search = RandomizedSearchCV(wrapper,
                                            param_grid,
                                            cv = stratified_cv,
                                            n_jobs=-1,
                                            n_iter=20,
                                            random_state=42)
                # The fit method will be called multiple times with different parameters
                # y_train is not used in the fit method, it is optional also for the sklearn API
                # but is needed for the stratified cross-validation
                grid_search.fit(X_train, y_train)

                # Get best parameters
                print("Best parameters:", grid_search.best_params_)
                print("the score of the search are:")
                latent = grid_search.best_estimator_.latent_model
                
                losses = grid_search.best_estimator_.losses_
                with open(os.path.join(self.base_path,"best_hp.txt"),"a+",encoding='utf-8') as f:
                    # remember the final new line
                    f.write(f"{model_name} {self.dataset_name} latent_dim={latent.latent_dim} batch_size={latent.batch_size} lr={latent.learning_rate} sigma={latent.sigma} \n") 

            
            self._explainers[model_name] = latent
            self._save_component(self._explainers[model_name], explainer_path)
            plt.figure(figsize=(8, 6))
            plt.plot(losses[0], label='train')
            plt.plot(losses[1], label='calibration')
            plt.title(f"Losses of ILS {model_name} on {self.dataset_name}")
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.yscale('log')
            plt.legend()
            plt.tight_layout()
            
            fname = os.path.join('plots',
                                    self.dataset_name,
                                    f'losses_{model_name}_ils.pdf')
            plt.savefig(fname)
            print(f"Losses plot saved to {fname}")
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
                cf_result = latent.get_counterfactuals(df_test=instance,
                                                    features_to_change=change_f,
                                                    max_features_to_change=max_f,
                                                    max_steps=128,
                                                    n_cfs=num_cf,
                                                    n_feats_sampled=max_f,
                                                    topn_to_check=32,
                                                    seed=69)
                # cf_result is a couple.
                #  The first element is the dataframe with the counterfactuals in the original space
                # the second element is the dataframe with the counterfactuals in the latent space
                # we save them BOTH in the results dictionary
                res_dictionary = {'instance_idx': idx,
                        'original_instance': instance,
                        'true_class': label,
                        'counterfactuals': cf_result[0],
                        'latent_counterfactuals': cf_result[1],
                    }

                model_results.append(res_dictionary)
            end = time.time()
            fancy_time = write_time(end-start)
            results[model_name] = model_results
            with open(os.path.join(self.base_path,"time_to_generate_ils.txt"),"a+",encoding='utf-8') as f:
                f.write(f"{model_name} {fancy_time} {num_cf} {self._method} {set_name} {dt_name} {len(X_calibration)} dimentions={latent.latent_dim}\n")

            self.save_results(results=model_results,
                                model_name= model_name,
                                dt_name=dt_name,
                                set_name=set_name)
        print(f"Time to generate counterfactuals: {fancy_time} seconds")
        
        # append on file the time to generate the counterfactuals

        return results