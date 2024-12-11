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
#import wandb



warnings.filterwarnings("ignore",
    message="X has feature names, but StandardScaler was fitted without feature names")


print("Loading config")