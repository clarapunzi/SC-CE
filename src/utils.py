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
def plot_confusion_ascii(y_true, y_pred,ascii_set= -1):
    """
    Plot a simple ASCII confusion matrix with class precision.
    Each cell shows a single character representing the percentage.
    """
    chars = []
    # Alternative
    chars.append(' ░▒▓█')
    chars.append(' .:;█')
    chars.append(' ·∙●■')
    chars.append(' .∙o●O@')
    chars.append('▁▂▃▄▅▆▇█')        # Rising blocks gradient
    chars.append(' ∙•○●◐◑◈■█')         # Circle to square gradient
    chars.append('0123456789')           # Digits
    chars.append('▁.⠆⠖⠶⣦⣶⣾⣿')     # Braille
    chars = chars[ascii_set]
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=0)
    for i, row in enumerate(cm):
        print(''.join(chars[int(cell * (len(chars)-1))] for cell in row))
    #print(''.join(str(i) for i in range(2)))

def write_time(s):
    s = np.round(s,2)
    if s<60:
        return str(s)+"s"
    elif s<60*60:
        s = np.round(s/60,2)
        return str(s)+"m"
    else:
        s = np.round(s/60/60,2)
        return str(s)+"h"



from sklearn.metrics import accuracy_score

def evaluate_coverage_accuracy(model, splits, percentages,selective_classifier_name="PLG"):
    """
    Evaluate the accuracy of selective classifiers at specified coverage percentages.

    Parameters:
        model: A trained PlugInRule or similar model with a `predict` method and a `qband` method.
        splits: A dictionary containing 'X_test' and 'y_test' as test data and true labels.
        percentages: A list of percentages for which to calculate accuracy.

    Returns:
        A dictionary where keys are coverage percentages and values are accuracy scores.
    """
    selected_plg = model.qband(splits['X_test'])  # Get acceptance levels for the test set
    y_test = splits['y_test'].values.reshape(-1)  # Reshape true labels for indexing
    X_test = splits['X_test']  # Test features

    results = {}
    for pct in percentages:
        threshold = int((100 - pct) / 100 * len(selected_plg))  # Map percentage to threshold
        mask = selected_plg >= threshold  # Create mask for selected cases
        accuracy = accuracy_score(y_test[mask], model.predict(X_test[mask]))
        results[pct] = accuracy
        print(f"{selective_classifier_name}: predicting {pct}% of cases: {accuracy:.4f}")

    return results
