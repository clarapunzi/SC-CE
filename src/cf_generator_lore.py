"""
Handles counterfactual generation for different models using DiCE, LoRE and other methods.
"""
import os
from typing import Dict, Any, List, Union
import warnings
import pickle
from datetime import datetime
import dice_ml
#from dice_ml.utils import helpers
import pandas as pd
import numpy as np
from src.cf_generator_base import CFGeneratorBase
#import wandb
try:
    os.chdir(os.path.join("src",
                        "LORE_sa"))
    from src.LORE_sa.lore_sa.lore import Lore
    from src.LORE_sa.lore_sa.bbox import AbstractBBox


    from src.LORE_sa.lore_sa.bbox import sklearn_classifier_bbox

    from src.LORE_sa.lore_sa.dataset import TabularDataset
    from src.LORE_sa.lore_sa.neighgen import GeneticGenerator, RandomGenerator
    from src.LORE_sa.lore_sa.encoder_decoder import ColumnTransformerEnc
    from src.LORE_sa.lore_sa.lore import (TabularRandomGeneratorLore,
                                          TabularGeneticGeneratorLore,
                                           TabularGeneticProbaGeneratorLore)
    from src.LORE_sa.lore_sa.surrogate import DecisionTreeSurrogate
except ImportError:
    print("Could not import LORE_sa")
    raise
finally:
    os.chdir("..")
    os.chdir("..")

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
        self.model = None
        self.surrogate = None
        self.generator = None
        self.encoder = None
        self.dataset = None
        self.bbox = None
        self.lore = None
    

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
        self.dataset = TabularDataset(reference_data,
                                      target_name,
                                      continuous_features,
                                      categorical_features)
        self.encoder = ColumnTransformerEnc(self.dataset)
        self.surrogate = DecisionTreeSurrogate()
        self.generator = TabularRandomGeneratorLore(self.dataset, self.encoder)
        self.bbox = sklearn_classifier_bbox
        self.lore = Lore(self.surrogate, self.generator, self.bbox)

    def _setup_model_components(self, model, model_name):
        return super()._setup_model_components(model, model_name)
    
    def generate_counterfactuals(self,
                                 models: Dict[str, Any],
                                 X_calibration: Union[pd.DataFrame, np.ndarray],
                                 y_calibration: Union[pd.Series, np.ndarray],
                                 cf_method: str,
                                 num_cf: int = 32,
                                 dt_name: str = 'dataset_name') -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate counterfactuals for given models and instances.
        """
        counterfactuals = {}
        for model_name, model in models.items():
            print(f"Generating counterfactuals for model {model_name}")
            counterfactuals[model_name] = []
            for i in range(len(X_calibration)):
                instance = X_calibration.iloc[i]
                instance = instance.to_dict()
                instance = {k: [v] for k, v in instance.items()}
                instance = pd.DataFrame.from_dict(instance)
                instance = self.encoder.transform(instance)
                instance = instance.values
                instance = instance[0]
                cf = self.lore.generate_counterfactual(instance, num_cf)
                counterfactuals[model_name].append(cf)
                                 