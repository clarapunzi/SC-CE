"""
This model contains the basaic utils for the project
"""
import os
import yaml
import numpy as np

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

def print_balancing(set_of):
    """ fancy printing of the balancing of a dataset"""
    v,c = np.unique(set_of, return_counts=True)

    mstr = "Balancing: "+f"{len(set_of)} samples"
    print(mstr)
    percentages = ""
    # here we use the special ascii character the filled square: █
    # to represent the percentage of each class in the dataset
    total_squares = 20
    old_perc=0
    for i,cs in enumerate(c):

        perc = total_squares*cs/(sum(c))
        percentages += "|"+" "*int(old_perc)+\
            "█"*int(perc)+" "*(total_squares-int(perc+old_perc))+"| class " + f"{v[i]}: {c[i]} "+f"({np.round(cs/(sum(c))*100,2)}%)\n"
        old_perc+=perc
    print(percentages)

from sklearn.metrics import confusion_matrix
def plot_confusion_ascii(y_true, y_pred):
    """
    Plot a simple ASCII confusion matrix with class precision.
    Each cell shows a single character representing the percentage.
    """
    chars = ' ░▒▓█'
    chars = ' ·∙●■'  # Alternative
    chars = ' .:;█'
    chars = ' .oO@'
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=0)
    for i, row in enumerate(cm):
        print(''.join(chars[int(cell * len(chars))] for cell in row))
    #print(''.join(str(i) for i in range(2)))
