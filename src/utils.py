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
        threshold = np.percentile(selected_plg, 100 - pct)

        mask = selected_plg >= threshold  # Create mask for selected cases
        accuracy = accuracy_score(y_test[mask], model.predict(X_test[mask]))
        results[pct] = accuracy
        print(f"{selective_classifier_name}: predicting {pct}% of cases: {accuracy:.4f}")

    return results


# create: rejected_list is a 0/1 list with 1 at rejected indices
# qband is a list with levels of acceptance: for j in range(len(target_coverage)), qband>=j is rejected
# qband >= i means setting to 0 (accept) all samples above i


def get_rejected_list(qband, i):
    '''
    This function creates a list of 0/1 values, where 1 means that the sample is rejected
    qband: list of quantiles
    i: the threshold for the quantile
    '''
    return [0 if qband >= i else 1 for qband in qband]

from typing import Any, Dict
from OLD_src.MyLoreSA.metrics import nonrejected_accuracy, classification_quality, rejection_quality, rejection_classification_report
def compute_selective_metrics(model_key: str,
    selected_data: np.ndarray,
    classifier_type: str,
    selective_classifier: Any,
    splits: Dict,
    target_coverages: np.ndarray,
    n: int,
    metric_dicts: Dict[str, Dict]):
    '''
    This function calculates the selective metrics for a given model and selective technique
    model_key: the key of the model in the models dictionary
    selected_data: the data selected by the selective technique
    classifier_type: the type of the classifier
    plug_in_ruler: the plug-in ruler of the model
    splits: the splits of the data
    coverage_index: the index of the coverage
    n: the number of samples
    metric_dicts: the dictionary of the metrics
    '''
    # assume the selective classifier is fitted
    k = model_key
    rejected_array = metric_dicts["rejected_by_coverage"][k][classifier_type]
    classification_array = metric_dicts["classification_quality_dict"][k][classifier_type]
    rejection_array = metric_dicts["rejection_quality_dict"][k][classifier_type]
    included_array = metric_dicts["included_samples"][k][classifier_type]
    for i,t_c in enumerate(target_coverages):
        #rejected list is a 0/1 list with 1 at rejected indices
        rejected_list = np.array(get_rejected_list(selected_data, i))
        (correct_nonrejected,
        correct_rejected,
        miscl_nonrejected,
        miscl_rejected,
            df_plg) = rejection_classification_report(splits['y_test'].values.reshape(-1),
                                                      selective_classifier.predict(splits['X_test']), 
                                                      rejected_list)
        acc_nrej = nonrejected_accuracy(correct_nonrejected, miscl_nonrejected)
        class_quality = classification_quality(correct_nonrejected, miscl_rejected, n)
        rej_quality = rejection_quality(correct_rejected, correct_nonrejected, miscl_rejected, miscl_nonrejected)
            #print(f"model {k}: non-rejected accuracy: {AN_plg:.2f}, {i}")
            #print(f"model {k}: classification quality: {CQ:.2f}, {i}")
            #print(f"model {k}: rejection quality: {RQ:.2f}, {i}")

        rejected_array[i] = acc_nrej
        classification_array[i] = class_quality
        rejection_array[i] = rej_quality
        included_array[i] = np.sum(1-rejected_list)/len(rejected_list)

