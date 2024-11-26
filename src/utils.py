"""
This model contains the basaic utils for the project
"""
import os
import pandas as pd
import yaml
def check_file_exists(file_path):
    """
    Check if the file exists
    """
    if not os.path.exists(file_path):
        return False
    else:
        return True
def load_config(config_path):
    """
    Load the configuration file is a yaml file
    """
    check_file_exists(config_path)
    
    return yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
