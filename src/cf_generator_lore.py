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
from cf_generator_base import CFGeneratorBase
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
    