"""
This script implements and evaluates selective classifiers using counterfactual distances.
It compares different selective classification methods including PlugInRule, PlugInRuleAUC,
and various CFDistRejector methods with different distance metrics.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import time
import argparse
from sklearn.metrics import accuracy_score
import sys
from tqdm import tqdm

# Load necessary modules
import src.utils as utl
import src.fancy_plots as fplt
from src.data_processor import DataProcessor, fancy_dataset_names,name_dataset_command
from src.model_trainer import ModelTrainer
from src.utils import compute_selective_metrics
from OLD_src.Lib.L2R.code.model_agnostic import PlugInRule, PlugInRuleAUC, SCRoss
import rejectmodels.CFDistRejector as cfdr

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Selective Classifiers Analysis for Counterfactual Examples')
    parser.add_argument('--dataset', type=str, required=True, choices=['adult48k', 'breast_cancer','german_credit', 'toy_dataset'],
                        help='Name of the dataset to analyze')
    # parser.add_argument('--method', type=str, required=True, choices=['dice', 'ils', 'lore'],
    #                    help='Counterfactual generation method')
    #parser.add_argument('--latent', action='store_true',
    #                    help='Use latent space for counterfactual generation')
    
    return parser.parse_args()

def create_directories(dataset):
    """Create necessary directories for saving results"""
    os.makedirs(os.path.join("plots", dataset), exist_ok=True)
    return os.path.join("plots", dataset)

def load_config(config_path):
    """Load configuration from file"""
    try:
        config = utl.load_config(config_path)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def load_data(config, dataset):
    """Load dataset splits"""
    try:
        data_processor = DataProcessor(config=config)
        splits = data_processor.load_splits(dataset)
        return splits
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def load_or_train_models(config, splits, dataset):
    """Load or train models for the dataset"""
    try:
        model_trainer = ModelTrainer(config)
        models = model_trainer.train_model(
            X_train=splits['X_train'],
            y_train=splits['y_train'],
            X_test=splits['X_test'],
            y_test=splits['y_test'],
            dataset_name=dataset
        )
        print("Available models:", models.keys())
        return models
    except Exception as e:
        print(f"Error training models: {e}")
        sys.exit(1)


def check_res(stats_dict, k="mlp"):
    """Check if distances are computed for a model"""
    try:
        print("(" + str(len(stats_dict[k]['distances']['l2']['mean'])), "elements computed)")
        return len(stats_dict[k]['distances']['l2']['mean']) >= 200
    except Exception as e:
        print(e)
        return False

def load_distance_stats(method, dataset, latent):
    """Load distance statistics from files"""
    fname = os.path.join(method, dataset + "_ranking_latent.pkl")
    fname_test = os.path.join(method, dataset + "_ranking_test_latent.pkl")
    
    if not latent:
        fname = fname.replace("_latent", "")
        fname_test = fname_test.replace("_latent", "")
    
    all_stats = None
    all_stats_test = None
    
    if utl.check_file_exists(fname):
        with open(fname, "rb") as f:
            all_stats = pickle.load(f)
            print("Loaded", fname)
    else:
        print("!" * 6, fname, "not found")
        sys.exit(1)
        
    if utl.check_file_exists(fname_test):
        with open(fname_test, 'rb') as f:
            all_stats_test = pickle.load(f)
            print("Loaded", fname_test)
    else:
        print("!" * 6, fname_test, "not found")
        sys.exit(1)
    
    return all_stats, all_stats_test


def convert_lists_to_arrays(models, statistics):
    """Convert all lists in the dictionaries to numpy arrays"""
    for k in models.keys():
        for metric in statistics[k]['distances'].keys():
            for stat in ["min", "max", "mean", "std"]:
                try:
                    statistics[k]['distances'][metric][stat] = np.array(statistics[k]['distances'][metric][stat])
                except Exception as e:
                    print(e)
                    print(k, metric, stat, len(statistics[k]['distances'][metric][stat]))
    
    for k in models.keys():
        statistics[k]['probabs'] = np.array(  statistics[k]['probabs'])
        statistics[k]['corrects'] = np.array( statistics[k]['corrects']).reshape(-1)
        statistics[k]['mean_conf'] = np.array(statistics[k]['mean_conf'])

    return statistics
def get_focus_metrics():
    """
    Define the distance metrics to focus on.
    To compare check the correlation between the different metrics among themselves from the notebook
    "
    """
    return [
        'inf', # 'chebyshev', 
        'l2', # 'minkowski',
        'sqeuclidean',
        'mae', #'l0', 'hamming', # l0 and hamming are the same for continuous data 
        'cosine',
        'correlation',
        'canberra',
        'braycurtis',
        #'hamming', # 'l0', # this is not very informatieve
        # 'wasserstein', # wasserstein excluded because is very very similar to l1
        'spearman',
        'kendall',
        'centered_cosine',
        'centered_l2',
        ]

def initialize_rejectors(models, splits):
    """Initialize selective classifiers and metrics dictionaries"""
    # Target coverages
    target_coverages_n = 30
    target_coverages = list(np.round(np.linspace(0.5, 1, target_coverages_n - 6)[::-1][1:],2))
    target_coverages = sorted(list(set(target_coverages +
                                [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
                                        )), reverse=True)
    target_coverages_n = len(target_coverages)
    n = len(splits['X_test'])
    
    focus_metrics = get_focus_metrics()
    
    # Initialize dictionaries for different rejector types
    # Basic rejectors
    plug_in_rulers = {}
    selected_plg = {}
    selected_plg_calibration = {}
    plug_in_rulers_auc = {}
    selected_plg_auc = {}
    selected_plg_auc_calibration = {}
    # distance-based rejectors
    l2_rejectors_distances = {}
    selected_l2_rejectors_distances = {}
    selected_l2_rejectors_distances_calibration = {}

    # tree based rejectors
    tree_rejectors = {}
    selected_tree_rejectors = {}

    # Dictionaries for metrics
    non_rejected_accuracy = {}
    classification_quality_dict = {}
    rejection_quality_dict = {}
    coverage = {}
    calibration_coverage = {}
    rejection_rate_class_0 = {}
    rejection_rate_class_1 = {}
    all_original_scores = {}
    
    # Initialize the selective classifiers
    for k in models.keys():      
        # Baseline methods
        plug_in_rulers[k] = PlugInRule(model=models[k])
        plug_in_rulers_auc[k] = PlugInRuleAUC(model=models[k])
        all_original_scores[k] = accuracy_score(splits['y_test'], models[k].predict(splits['X_test']))
        
        # Result dictionaries
        non_rejected_accuracy[k] = {}
        classification_quality_dict[k] = {}
        rejection_quality_dict[k] = {}
        coverage[k] = {}
        calibration_coverage[k] = {}
        rejection_rate_class_0[k] = {}
        rejection_rate_class_1[k] = {}

        # Initialize for basic methods
        for selective_c in ["PlugInRule", "PlugInRuleAUC"]:
            non_rejected_accuracy[k][selective_c] = np.zeros(target_coverages_n)
            classification_quality_dict[k][selective_c] = np.zeros(target_coverages_n)
            rejection_quality_dict[k][selective_c] = np.zeros(target_coverages_n)
            coverage[k][selective_c] = np.zeros(target_coverages_n)
            calibration_coverage[k][selective_c] = np.zeros(target_coverages_n)
            rejection_rate_class_0[k][selective_c] = np.zeros(target_coverages_n)
            rejection_rate_class_1[k][selective_c] = np.zeros(target_coverages_n)
    
    # Initialize for distance-based rejectors
    for k in models.keys():
        l2_rejectors_distances[k] = {}
        tree_rejectors[k] = {}
        for metric in focus_metrics:
            for m_type in ["min", "max", "mean"]:
                for gamma_flag in [False, True]:
                    selective_c = "CFDistRejector_" + metric + "_" + m_type + ("_gamma" if gamma_flag else "")
                    l2_rejectors_distances[k][selective_c] = cfdr.CFDistRejector(
                        model=models[k],
                        coverages=target_coverages
                    )
                    
                    # Initialize result dictionaries for this rejector
                    non_rejected_accuracy[k][selective_c] = np.zeros(target_coverages_n)
                    classification_quality_dict[k][selective_c] = np.zeros(target_coverages_n)
                    rejection_quality_dict[k][selective_c] = np.zeros(target_coverages_n)
                    coverage[k][selective_c] = np.zeros(target_coverages_n)
                    calibration_coverage[k][selective_c] = np.zeros(target_coverages_n)
                    rejection_rate_class_0[k][selective_c] = np.zeros(target_coverages_n)
                    rejection_rate_class_1[k][selective_c] = np.zeros(target_coverages_n)

                    # Initialize for tree-based rejectors
                    # selective_c = "CFTreeRejector_" + metric + "_" + m_type + ("_gamma" if gamma_flag else "")
                    # tree_rejectors[k][selective_c] = cftree.CFTreeRejector(
                    #     model=models[k],
                    #     coverages=target_coverages
                    # )
                    # Initialize result dictionaries for this rejector
                    # non_rejected_accuracy[k][selective_c] = np.zeros(target_coverages_n)
                    # classification_quality_dict[k][selective_c] = np.zeros(target_coverages_n)
                    # rejection_quality_dict[k][selective_c] = np.zeros(target_coverages_n)
                    # coverage[k][selective_c] = np.zeros(target_coverages_n)
                    
    
    # Group metric dictionaries
    metric_dicts = {
        "non_rejected_accuracy": non_rejected_accuracy,
        "classification_quality_dict": classification_quality_dict,
        "rejection_quality_dict": rejection_quality_dict,
        "coverage": coverage,
        "calibration_coverage": calibration_coverage,
        "rejection_rate_class_0": rejection_rate_class_0,
        "rejection_rate_class_1": rejection_rate_class_1
    }
    
    rejectors = {
        "plug_in_rulers": plug_in_rulers,
        "selected_plg": selected_plg,
        "selected_plg_calibration": selected_plg_calibration,
        "plug_in_rulers_auc": plug_in_rulers_auc,
        "selected_plg_auc": selected_plg_auc,
        "selected_plg_auc_calibration": selected_plg_auc_calibration,
        "l2_rejectors_distances": l2_rejectors_distances,
        "selected_l2_rejectors_distances": selected_l2_rejectors_distances,
        "selected_l2_rejectors_distances_calibration": selected_l2_rejectors_distances_calibration,
        "tree_rejectors": tree_rejectors,
        "selected_tree_rejectors": selected_tree_rejectors
    }
    
    info = {
        "focus_metrics": focus_metrics,
        "target_coverages": target_coverages,
        "target_coverages_n": target_coverages_n,
        "n": n,
        "all_original_scores": all_original_scores
    }
    
    return rejectors, metric_dicts, info


def calibrate_basic_rejectors(models, rejectors, splits, all_stats, all_stats_test, info):
    """Calibrate PlugInRule, PlugInRuleAUC, and basic L2 rejectors"""
    target_coverages = info["target_coverages"]
    focus_metrics = info["focus_metrics"]
    # for k in models.keys():
    # use the fancy bar from tqdm
    for k in tqdm(models.keys(), desc="Calibrating basic rejectors"):
        print(f"Calibrating basic rejectors for model {k}")
        
        # Calibrate PlugInRule
        plug_in_rule = rejectors["plug_in_rulers"][k]
        plug_in_rule.calibrate(splits["X_calibration"], target_coverages=target_coverages)
        rejectors["selected_plg"][k] = plug_in_rule.qband(splits["X_test"])
        rejectors["selected_plg_calibration"][k] = plug_in_rule.qband(splits["X_calibration"])

        # Calibrate PlugInRuleAUC
        plug_in_ruler_auc = rejectors["plug_in_rulers_auc"][k]
        plug_in_ruler_auc.calibrate(
            X=splits["X_calibration"],
            y=splits["y_calibration"].values.reshape(-1),
            target_coverages=target_coverages
        )
        rejectors["selected_plg_auc"][k] = rejectors["plug_in_rulers_auc"][k].qband(splits["X_test"])
        rejectors["selected_plg_auc_calibration"][k] = rejectors["plug_in_rulers_auc"][k].qband(splits["X_calibration"])
        
        print("Basic rejectors calibrated:")
        print(f"PlugInRule: {rejectors['selected_plg'][k][:5]}")
        print(f"PlugInRuleAUC: {rejectors['selected_plg_auc'][k][:5]}")
    
    return rejectors


def calibrate_distance_rejectors(models, rejectors, splits, all_stats, all_stats_test, info,calibrate_trees=False):
    """Calibrate all distance-based rejectors"""
    focus_metrics = info["focus_metrics"]
    rejectors["selected_l2_rejectors_distances"] = {}
    if calibrate_trees:
        rejectors["selected_tree_rejectors"] = {}
    
    for k in tqdm(models.keys(), desc="Calibrating distance-based rejectors"):
        rejectors["selected_l2_rejectors_distances"][k] = {}
        rejectors["selected_l2_rejectors_distances_calibration"][k] = {}
        if calibrate_trees:
            rejectors["selected_tree_rejectors"][k] = {}
        
        for metric in focus_metrics:
            for m_type in ["min", "max", "mean"]:
                # Standard rejector
                selective_c = f"CFDistRejector_{metric}_{m_type}"
                l2_rejector_dist = rejectors["l2_rejectors_distances"][k][selective_c]
                l2_rejector_dist.calibrate(
                    splits["X_calibration"], 
                    all_stats[k]["distances"][metric][m_type]
                )
                rejectors["selected_l2_rejectors_distances"][k][selective_c] = l2_rejector_dist.qband(
                    splits["X_test"],
                    all_stats_test[k]["distances"][metric][m_type]
                    )
                rejectors["selected_l2_rejectors_distances_calibration"][k][selective_c] = l2_rejector_dist.qband(
                    splits["X_calibration"],
                    all_stats[k]["distances"][metric][m_type]
                    )

                # Gamma rejector
                selective_c_gamma = f"CFDistRejector_{metric}_{m_type}_gamma"
                l2_rejector_dist_gamma = rejectors["l2_rejectors_distances"][k][selective_c_gamma]
                l2_rejector_dist_gamma.calibrate(
                    splits["X_calibration"],
                    all_stats[k]["distances"][metric][m_type],
                    use_gamma=True
                )
                rejectors["selected_l2_rejectors_distances"][k][selective_c_gamma] = l2_rejector_dist_gamma.qband(
                    splits["X_test"],
                    all_stats_test[k]["distances"][metric][m_type]
                    )
                rejectors["selected_l2_rejectors_distances_calibration"][k][selective_c_gamma] = l2_rejector_dist_gamma.qband(
                    splits["X_calibration"],
                    all_stats[k]["distances"][metric][m_type]
                    )
                
                if calibrate_trees:
                    # Tree rejector
                    selective_c_tree = f"CFTreeRejector_{metric}_{m_type}"
                    tree_rejector = rejectors["tree_rejectors"][k][selective_c_tree]
                    tree_rejector.calibrate(
                        splits["X_calibration"],
                        splits["y_calibration"],
                        target_distances=all_stats[k]["distances"][metric][m_type]
                    )
                    rejectors["selected_tree_rejectors"][k][selective_c_tree] = tree_rejector.qband(
                        splits["X_test"],
                        all_stats_test[k]["distances"][metric][m_type]
                        )
                    # Gamma tree rejector
                    selective_c_tree_gamma = f"CFTreeRejector_{metric}_{m_type}_gamma"
                    tree_rejector_gamma = rejectors["tree_rejectors"][k][selective_c_tree_gamma]
                    tree_rejector_gamma.calibrate(
                        splits["X_calibration"],
                        splits["y_calibration"],
                        target_distances=all_stats[k]["distances"][metric][m_type],
                        use_gamma=True
                    )
                    rejectors["selected_tree_rejectors"][k][selective_c_tree_gamma] = tree_rejector_gamma.qband(
                        splits["X_test"],
                        all_stats_test[k]["distances"][metric][m_type]
                        )
                
    
    return rejectors


def evaluate_rejectors(models, rejectors, splits, metric_dicts, info, calibrate_trees=False):
    """Evaluate all rejector types"""
    target_coverages = info["target_coverages"]
    n = info["n"]
    
    for k in tqdm(models.keys(), desc="Evaluating rejectors"):        
        # Evaluate basic rejectors
        basic_rejectors = {
            "PlugInRule": {
                "selected_data": rejectors["selected_plg"][k],
                "selected_data_calibration": rejectors["selected_plg_calibration"][k],
                "rejector": rejectors["plug_in_rulers"][k]
            },
            "PlugInRuleAUC": {
                "selected_data": rejectors["selected_plg_auc"][k],
                "selected_data_calibration": rejectors["selected_plg_auc_calibration"][k],
                "rejector": rejectors["plug_in_rulers_auc"][k]
            }
        }

        for selective_k, data in basic_rejectors.items():
            compute_selective_metrics(
                model_key=k,
                selected_data=data["selected_data"],
                selected_data_calibration=data["selected_data_calibration"],
                classifier_type=selective_k,
                selective_classifier=data["rejector"],
                splits=splits,
                target_coverages=target_coverages,
                n=n,
                metric_dicts=metric_dicts
            )

        # Evaluate distance-based rejectors
        for selective_c, selected_data in rejectors["selected_l2_rejectors_distances"][k].items():
            compute_selective_metrics(
                model_key=k,
                selected_data=selected_data,
                selected_data_calibration=rejectors["selected_l2_rejectors_distances_calibration"][k][selective_c],
                classifier_type=selective_c,
                selective_classifier=rejectors["l2_rejectors_distances"][k][selective_c],
                splits=splits,
                target_coverages=target_coverages,
                n=n,
                metric_dicts=metric_dicts
            )
        if calibrate_trees:
            # Evaluate tree-based rejectors
            for selective_c, selected_data in rejectors["selected_tree_rejectors"][k].items():
                compute_selective_metrics(
                    model_key=k,
                    selected_data=selected_data,
                    classifier_type=selective_c,
                    selective_classifier=rejectors["tree_rejectors"][k][selective_c],
                    splits=splits,
                    target_coverages=target_coverages,
                    n=n,
                    metric_dicts=metric_dicts
                )
    
    return metric_dicts


def create_dataframes(models, metric_dicts, info,
                      selective_metric="non_rejected_accuracy"):
    """Create DataFrames for visualization"""
    target_coverages = info["target_coverages"]
    
    dataframes = {}
    for k in models.keys():
        df = pd.DataFrame()
        for method, val in metric_dicts[selective_metric][k].items():
            df[method] = val
        df = df.T
        df.columns = target_coverages
        dataframes[k] = df
    
    return dataframes
def fancy_generator(method,command=False):
    if "latent" in method.lower() and "ils" in method.lower():
        if command:
            return "\\ilslatent"
        return r"ILS$_L$"
    if "growing" in method.lower():
        if command:
            return "\\growingspheres"
        return "GS"
    if "dice" in method.lower():
        if command:
            return "\\dice"
        return "DiCE"
    if command:
        return "\\"+method.lower()
    return method.upper()

def fancy_names(name,command=False):
    """Return a cooler name for the selective classifier"""
    name = name.replace("centered_l2", "centered L2").replace("centered_cosine", "centered cosine")
    name = name.replace("l2", "L2").replace("mae", "MAE")
    if "PlugInRuleAUC" in name:
        return "PlugInRuleAUC"
    elif "PlugInRule" in name:
        return "PlugInRule"
    else:
        """Return a cooler name for the selective classifier"""
        if name.endswith("duplicate"):
            duplicate_flag = " (D)"
            name = name.replace("_duplicate", "")
        else:
            duplicate_flag = ""
        distr = name.split("_")[-1]
        if distr == "gamma":
            apex = "^{\gamma}"
            name = name.replace("_gamma", "")
        else:
            apex = ""
        distr = name.split("_")[-1]
        if distr == "mean":
            distr = "_{avg}"
        elif distr == "min":
            distr = "_{min}"
        elif distr == "max":
            distr = "_{max}"
        method = name.split("_")[0]

        method = fancy_generator(method,command=command)
        distance = name.split("_")[2]
        if "growing" in method.lower():
            print(name, distance, method)
        newn = method+" - "+distance+"$"+apex+distr+"$" + duplicate_flag
        # print(name,newn)
        return newn

subpartial_yet_easier_results = {}

def generate_latex_table(ldf,
                         black_box_score,
                         black_box_name,
                         top_policies,
                         top_policies_names, 
                         target_coverages,
                         dataset_name,                 
                         sorted_rows,
                         print_heaedr=False,
                         print_footer=False,   
                         cmap="YlGnBu",
                         vmax=1,
                         minimum=0.5,
                         num_models = 4,
                         show = False,):
    """Generate a LaTeX table for the top selective classifiers."""
    # plt.figure(figsize=(8, 0.3*len(ldf)))
    # sns.heatmap(ldf.loc[sorted_rows,:].values, annot=True, fmt=".3f", cmap=cmap, cbar=True,
    #             xticklabels=target_coverages,
    #             yticklabels=[fancy_names(fnam) for fnam in ldf.loc[sorted_rows,:].index],
    #             vmax=vmax,
    #             vmin=minimum)
    # plt.title(f"Top Selective Classifiers for {dataset_name} ({black_box_name})")
    # plt.xlabel("Target Coverage")
    # # save the figure
    # figname = dataset_name.replace(" ","_") + "_" + black_box_name + "_top_selective_classifiers_TABLE.pdf"
    # fpath = os.path.join("results", figname)
    # plt.savefig(fpath, bbox_inches='tight')
    # print("Figure saved as", fpath)
    # if show:
    #     plt.show() 
    best_vals_by_col = ldf.values[top_policies, :].max(axis=0)
    #print(ldf)
    #print("best_vals_by_col", best_vals_by_col)
    if print_heaedr:
        # make the table using header
        table = "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        table += "\n% This table is generated by the script generate_latex_table.py\n"
        table += "%"+"For dataset: "+dataset_name + "\n"
        table += "%"+"For black box model: "+black_box_name + "\n"
        table += "%"+"For target coverages: "+str(target_coverages) + "\n"
        table += "% Do not edit it manually\n"
        table += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        table += "\\begin{table*}[t]\n\\centering\n\\resizebox{1\linewidth}{!}{\n\\begin{tabular}{c|c|l|" + "c|" * len(ldf.columns) + "}\n"
        # the header have the dataset name, the cf method, and the target coverages
        # table += "\\hline\n\\textbf{Dataset} & \\textbf{Black Box} & \\textbf{Rejection Policy} & " + " & ".join([f"{round(c, 2)}" for c in  ldf.columns]) + " \\\\\n\\hline\n"
        table += "\\multirow{2}{*}{\\textbf{Dataset}} & \\multirow{2}{*}{\\textbf{Black Box}} & \\multirow{2}{*}{\\textbf{Rejection Policy}} & \\multicolumn{" + str(len(ldf.columns)) + "}{c|}{Target Coverages}\\\\\n\\cline{4-"+str(3+ len(ldf.columns))+"}\n"
        table += "& & & " + " & ".join([f"{round(c*100):.0f}\\%" for c in ldf.columns]) + " \\\\\n\\hline\n"
        # the name of the dataset is written vertically and taks as many lines as the number of the top policies
        table += "\\multirow{" + str(len(top_policies)*num_models) + "}{*}{\\rotatebox[origin=c]{90}{\\textbf{" + name_dataset_command[dataset_name] + "}}}"
    else: # the header is already been printed so we just add a \cline
        table = "\\cline{2-"+str(3+ len(ldf.columns))+"}\n"
        #table = "\\hline\n"
    
    # this function is called for each black box model
    # so the second column is the name of the black box model
    table += " & \\multirow{" + str(len(top_policies)) + "}{*}{\\rotatebox[origin=c]{90}{\\textbf{\\" + black_box_name.replace('_','') + "}}}"
    # print(best_vals_by_col.round(3))
    for jjjj,(idx, name) in enumerate(zip(top_policies, top_policies_names)):
        #row = f"& {name} & "+ " & ".join([f"{round(df.values[idx, j], 2)}" for j in range(len(target_coverages))]) + " \\\\\n"
        # the first two "empty" columns are the dataset name and the black box name
        if idx in [0,1]:
            row = f"{'&' if jjjj !=0 else ''} & {name} & " + " & ".join(
            [(f"{val:.3f}" if abs(val - best_vals_by_col[jjj])> 0.0001  else 
              "\\textbf{" + f"{val:.3f}" + "}")
              for jjj,val in enumerate(ldf.values[idx])]
            ) + " \\\\\n"
        else:
            row = f"{'&' if jjjj !=0 else ''} & "+"\\cellcolor[gray]{0.9}{"+f"{name}"+"} & " + " & ".join(
            [("\\cellcolor[gray]{0.9}{"+            f"{val:.3f}"+"}" if abs(val - best_vals_by_col[jjj])> 0.0001  else 
              "\\cellcolor[gray]{0.9}{\\textbf{" + f"{val:.3f}" + "}}")
              for jjj,val in enumerate(ldf.values[idx])]
            ) + " \\\\\n"

        table += row
        # print([(str(np.round(v,3))+"_"+str(np.round(best_vals_by_col[jj],3)) if abs(v-best_vals_by_col[jj])>0.0001 else "MAX_"+str(v)+"_MAX" ) for jj,v in enumerate(ldf.values[idx]) ])
    if print_footer:
        table += "\\hline\n\\end{tabular}\n}\n\\caption{Top Selective Classifiers for the dataset "
        table += name_dataset_command[dataset_name] + "}\n\\end{table*}"
    else:
        table += ''
    # print("Generated LaTeX Table:\n", table)
    return table

def plot_heatmap(ax, df, top_policies, top_policies_names, target_coverages_n, minimum, vmax):
    """Plot heatmap of top methods."""
    custom_cmap = 'inferno'
    im = ax.imshow(df.values[top_policies, :], cmap=custom_cmap, vmin=minimum, vmax=vmax)
    ax.set_yticks(np.array([i for i in range(len(top_policies_names))]))
    ax.set_yticklabels(top_policies_names, rotation=0, fontsize=16)
    ax.set_xticks(range(target_coverages_n // 2))
    ax.set_xticklabels([str(round(e, 2)) for e in df.columns], rotation=0)
    plt.colorbar(im, ax=ax, label="Rejection Quality")

def simplified_plot_styling(method_name):
    """Simplified styling based on method name parsing."""
    metrics_colors = {
        # Blue
        'inf': '#1f77b4',         
        # Orange
        'cosine': '#ff7f0e',      
        # Green
        'braycurtis': '#2ca02c',  
        # Red
        'chebyshev': '#d62728',   
        # Purple
        'l2': '#9467bd',          
        # Brown
        # 'minkowski': '#8c564b',   
        # Pink
        'wasserstein': '#e377c2', 
        # dark blue
        # 'l1': '#5f7e0b',
        # Yellow-green
        'mae': '#bcbd22',         
        # Cyan
        'sqeuclidean': '#17becf'  
    }
    style = {}
    method_name = method_name.lower()
    # Set defaults
    style['ls'] = '-'
    style['marker'] = ''
    style['color'] = 'blue'
    style['linewidth'] = 1
    style['markersize'] = 8
    
    # Check for baseline methods
    if 'plugin' in method_name:
        style['ls'] = '--'
        style['color'] = 'black'
        if 'auc' in method_name:
            style['marker'] = '$A$'
            style['markersize'] = 10
        return style
    # Check for gamma distribution
    if 'gamma' in method_name:
        # use  the symbol \gamma itself
        style['marker'] = '$\gamma$'
    else:
        style['markersize'] = 4

    
    # Check for counterfactual method
    if 'lore' in method_name:
        style['color'] = '#1f77b4'  # Blue
    elif 'latent' in method_name:
        style['color'] = '#ff7f0e'  # Orange
    elif 'dice' in method_name:
        style['color'] = '#2ca02c'  # Green
    elif 'ils' in method_name:
        style['color'] = '#d62728'  # Red
    elif 'spheres' in method_name:
        style['color'] = '#9467bd'  # Purple
    if 'min' in method_name:
        style['linewidth']= 2.5
        style["ls"] = ':'
    if 'max' in method_name:
        style['linewidth']= 3

    # Check for metrics in the method name
    for metric, color in metrics_colors.items():
        if metric in method_name.lower():
            style['color'] = color
            break
    
    return style

def plot_line_graph_nb(ax, df, top_policies, top_policies_names, 
                    target_coverages, all_original_scores, 
                    model_name, minimumm, vmaxx,
                    selective,# the name of the selective metric
                    indx=0):
    """Plot line graph of top methods."""
    for j, policy_idx in enumerate(top_policies):
        label = top_policies_names[j]
        style = simplified_plot_styling(label)
        ax.plot(df.columns, df.values[policy_idx, :], 
                label=label, 
                ls=style['ls'],
                marker=style['marker'],
                color=style['color'],
                linewidth=style['linewidth'],
                markersize=style['markersize'],
                )
    if selective == "non_rejected_accuracy":
        ax.axhline(all_original_scores[model_name], label=f"Original {model_name}", color='k', ls='--')
    target_coverages = target_coverages.copy()
    target_coverages = np.array(target_coverages)[((np.array(target_coverages)*100)%10==0)]
    
    ax.set_xticks(target_coverages)
    ax.set_xticklabels([f"{e*100:.0f}%" for e in target_coverages], rotation=0,
                       fontsize=18)
    xlims = ax.get_xlim()
    ax.set_xlim(xlims[1], xlims[0])
    ax.set_xlabel("Target Coverage"   , fontsize=21)
    # ax.legend()
    if minimumm is not None and vmaxx is not None:
        print(minimumm, vmaxx)
        ax.set_ylim(minimumm, vmaxx)
    else:
        ax.set_ylim(0, 1)
    if indx == 0:
        if selective == "non_rejected_accuracy":
            ylab = "Non-rejected Accuracy"
        elif selective == "rejection_quality_dict":
            ylab = "Rejection Quality"
        elif selective == "classification_quality_dict":
            ylab = "Classification Quality"
        elif selective == "selective_accuracy":
            ylab = "Selective Accuracy"
        elif selective == "coverage":
            ylab = "Coverage"
        elif selective == "rejection_rate_class_0":
            ylab = "Rejection Rate (Class 0)"
        elif selective == "rejection_rate_class_1":
            ylab = "Rejection Rate (Class 1)"
        ax.set_ylabel(ylab, fontsize=24)
    if indx == 3:
        # place the yticklabels on the right
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        #ax.set_yticks(ax.get_yticks()[::2])
        
        ax.set_yticklabels([f"{e*100:.1f}%" for e in ax.get_yticks()], fontsize=20)
    else:
        ax.set_yticklabels(["" for e in ax.get_yticks()], fontsize=20)
        
    ax.grid(True,alpha=0.6)


def create_legend_file(top_policies_names, model_name,legend_fig_name):
    """Create a standalone legend file with metrics-based coloring."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_axis_off()  # Hide the axes
    print(top_policies_names)
    
    top_policies_names = list(set(top_policies_names))
    metric_methods = top_policies_names
    # place pluginrule and pluginruleauc at the beginning
    metric_methods = ["PlugInRule", "PlugInRuleAUC", "Original Black Box"] + [e for e in metric_methods if e not in ["PlugInRule", "PlugInRuleAUC", "Original Black Box"]] 
    ax.plot([], [], label=f"Original {model_name}", color='k', ls='--')
    for name in metric_methods:
        style = simplified_plot_styling(name)
        label = name
        ax.plot([], [], 
                    label=label,
                    marker=style['marker'],  # Standard marker
                    ls=style['ls'],
                    color=style['color'],
                    linewidth=style['linewidth'],
                    markersize=style['markersize'])
    
    # Create the legend with multiple columns
    legend = ax.legend(loc='center', ncol=4, frameon=False, fontsize=18, 
                      handlelength=2, handletextpad=0.5)
    # Save just the legend
    fig.savefig(f"{legend_fig_name}", bbox_inches='tight')
    #plt.close(fig)
fancy_model = {"mlp": "MLP",
               "random_forest": "Random Forest",
               'xgboost': 'XGBoost',
               'lgbm': 'LGBM',
               'lip_mlp': 'LipMLP'}
def visualise_results(models, dataframes, info, name_dataset, plot_dir,
                      fig_name="alternative_selective_classifiers",
                      show =False,
                      top_k=2+5, # Number of top methods to select
                      selective='rejection_quality_dict',
                      ):
    
    target_coverages = info["target_coverages"]
    all_original_scores = info["all_original_scores"]

    if selective == "non_rejected_accuracy":
        
        minimum = np.array([all_original_scores[k] for k in all_original_scores.keys()]).min()
        vmax =    0
        for k in models.keys():

            # Get indices of top performing methods by sum across coverages

            fake_auc = dataframes[k].copy().values[:,:].sum(axis=1)
            # sort the fake_auc
            fake_auc = fake_auc.argsort()[::-1]
            # take the top_k rows
            fake_auc = fake_auc[:top_k]
            vmax = max(vmax, dataframes[k].values[fake_auc, :].max())
        minimum-=0.007
        vmax+=0.007
    else:
        minimum = None
        vmax = None
    print("vmax",vmax)
    print("minimum",minimum)
    # create the latex tables
    latex_tables = {k:"" for k in models.keys()}
    # big_tables =   {k:"" for k in models.keys()}
    # use the fancy bar from tqdm
    fig, ax = plt.subplots(1, len(models), figsize=(7*len(models), 7))
    ax = ax.flatten()
    plotted_methods = []
    for indx,k in enumerate(tqdm(models.keys(), desc="Visualizing results")):
        # Select top performing methods
        df = dataframes[k].copy().iloc[:, :]
        
        # it could happen that there is no difference between the methods (some rows are the same and differ only by the index)
        # so we need to drop the duplicates and modify the index of the duplicated rows that are kept by adding a suffix "_duplicate"
        old_index = df.index
        old_df = df.copy()
        # print("Dropping duplicates")
        
        df = df.drop_duplicates()
        for i in old_index:
            if i not in df.index:
                # using the old_df to get the row that is duplicated
                old_row = old_df.loc[i]
                for j in df.index:
                    if j.endswith("_duplicate"):
                        continue
                    if (old_row == old_df.loc[j]).all():
                        if j.endswith("_duplicate"):
                            # I've already found the duplicated row
                            break
                        else:
                            df.loc[j+"_duplicate"] = old_row
                            df = df.drop(j)
                        break
                
        
        # add a flag column to test that the values are always worst than the original model (so that it will act as a regularizer for the next sorting)
        # the following line has a bug that makes it not work for boolean values and flats and blah blah
        # df["flag"] = (df.values > all_original_scores[k]).all(axis=1)
        
        # the correct way to do it is to use the following line
        df["flag"] = 0
        for i in range(len(df)):
            if (df.iloc[i, :-1] < all_original_scores[k]).any():
                df["flag"].iloc[i] = -20
                #print("flag",df.iloc[i, :].index,"has a value lower than the original model")
                #print(df.iloc[i, :]< all_original_scores[k])
        # Get indices of top performing methods by sum across coverages
        fake_auc = df.values.sum(axis=1).argsort()[::-1]
        
        # now remove the flag column
        df = df.drop(columns=["flag"], axis=1)

        # take the top_k rows    
        top_policies = list(fake_auc[:top_k])
        # Ensure PlugInRule and PlugInRuleAUC are included
        if 0 not in top_policies:
            top_policies = [0] + top_policies[:-1]
        if 1 not in top_policies:
            top_policies = [1] + top_policies[:-1]
        # now place the PlugInRule and PlugInRuleAUC at the beginning
        top_policies = [0, 1] + [e for e in top_policies if e not in [0, 1]]
        top_policies_names = [fancy_names(fnam) for fnam in df.index[top_policies]]       
        plotted_methods += [fancy_names(fnam) for fnam in df.index[top_policies] if fnam not in plotted_methods]
        column_values = np.concatenate([[0.95],
                                        np.array(target_coverages[:])[((
                                            np.array(target_coverages[:])*100)%10==0)]
                                        ])
        custom_cmap = 'inferno'
        custom_cmap = 'Greys_r'
        custom_cmap = fplt.parula
        # fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        # Plot line graph of top methods
        plot_line_graph_nb(ax=ax[indx],
                df=df,
                top_policies=top_policies,
                top_policies_names=top_policies_names,
                target_coverages=column_values,
                all_original_scores=all_original_scores,
                model_name=k,
                minimumm=minimum,vmaxx=vmax,
                selective= selective,
                indx=indx,
                )
    
        # set title for the model
        ax[indx].set_title(f"{fancy_model.get(k, k)}")
        # Set title (not sure if it is needed)
        # plt.suptitle(
        #     f"{name_dataset} - Model: {k}",
        #       # {args.method.upper()} {'Latent' if args.latent else ''} - 
        #     fontsize=16
        # )
        # Save figure
        # Create LaTeX table
        # big_tables[k] = generate_latex_table(ldf=df,
        #                                    black_box_score=all_original_scores[k],
        #                                    black_box_name=k,
        #                         top_policies= list(fake_auc),
        #                         top_policies_names=[fancy_names(fnam) for fnam in df.index[fake_auc]],
        #                         target_coverages=column_values,
        #                         dataset_name=name_dataset,
        #                         sorted_rows = list(df.index[fake_auc]),
        #                         print_heaedr=(indx==0),
        #                         print_footer=(indx==len(models.keys())-1),
        #                         cmap =custom_cmap,
        #                         vmax=vmax,
        #                         minimum=minimum
        #                         )
        latex_tables[k] = generate_latex_table(ldf=df.loc[:,(column_values).tolist()],
                                            black_box_score=all_original_scores[k],
                                            black_box_name=k,
                                 top_policies=top_policies,
                                 top_policies_names=top_policies_names,
                                 target_coverages=column_values,
                                 dataset_name=name_dataset,
                                 sorted_rows = list(df.index[fake_auc]),
                                 print_heaedr=(indx==0),
                                 print_footer=(indx==len(models.keys())-1),
                                 cmap =custom_cmap,
                                 vmax=vmax,
                                 minimum=minimum
                                 )
    fig_path = os.path.join(
        plot_dir,
        fig_name+"_"+selective+".pdf"
    )
    plt.tight_layout()
    if show:
        plt.show()
        plt.clf()
    else:
        plt.savefig(fig_path)
        print(f"Saved figure to {fig_path}")  
        # Also save PNG for easy viewing
        plt.savefig(fig_path.replace(".pdf", ".png"))
        plt.close(fig)
    # the legend
    create_legend_file(plotted_methods, "Black Box", fig_path.replace(".pdf", "_legend.pdf"))

    # make sure the tables directory exists
    table_name = "latex_table_"+selective+"_"+ name_dataset.replace(" ","_")+".tex"
    path = os.path.join("results","tables")
    os.makedirs(path, exist_ok=True)
    # save the latex tables
    # with open(os.path.join(path, "BIG_"+table_name), "w") as f:
    #     for k in models.keys():
    #         #print(latex_tables[k])
    #         f.write(big_tables[k])
    with open(os.path.join(path, table_name), "w") as f:
        for k in models.keys():
            f.write(latex_tables[k])

    return


def visualise_results_by_rejector(models, dataframes, info, name_dataset, plot_dir,
                                 fig_name="rejectors_across_models",
                                 show=False,
                                 top_k=5,
                                 selective='rejection_quality_dict',
                                 cf_methods=None,
                                 ):
    """
    Visualize results as a grid: rows=models, columns=CF generators.
    Within each cell, the top-k distance metrics for that (model, CF generator) are plotted,
    with baselines (PlugInRule, PlugInRuleAUC) always shown for reference.
    Since baselines are constant within each row, visual differences across columns
    reveal the effect of each CF generator.
    """
    if cf_methods is None:
        cf_methods = ["growingspheres", "ils", "ils<latent", "lore"]

    target_coverages = info["target_coverages"]
    all_original_scores = info["all_original_scores"]

    column_values = np.array(target_coverages)[
        ((np.array(target_coverages) * 100) % 10 == 0)
    ]

    n_models = len(models)
    n_cf = len(cf_methods)

    fig, axes = plt.subplots(n_models, n_cf, figsize=(n_cf * 4, n_models * 4), sharey='row')
    if n_models == 1:
        axes = axes.reshape(1, -1)
    if n_cf == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, model_key in enumerate(models.keys()):
        df = dataframes[model_key].copy()
        y_min = all_original_scores[model_key] - 0.025
        y_max = 1.025

        # Baseline rows - constant across all columns in this row
        baselines = {name: df.loc[name] for name in ["PlugInRule", "PlugInRuleAUC"] if name in df.index}

        for col_idx, cf_method in enumerate(cf_methods):
            ax_curr = axes[row_idx, col_idx]

            # Filter rows belonging to this CF generator
            cf_rows = [idx for idx in df.index if idx.startswith(cf_method + "_")]
            if not cf_rows:
                ax_curr.set_visible(False)
                continue

            cf_df = df.loc[cf_rows]

            # Select top-k by sum across all coverages
            scores = cf_df.values.sum(axis=1)
            top_idx = scores.argsort()[::-1][:top_k]

            ax_curr.axhline(all_original_scores[model_key], color='k', ls=':', alpha=0.8, label='Original score')
            for i in top_idx:
                row_name = cf_df.index[i]
                style = simplified_plot_styling(row_name)
                full_label = fancy_names(row_name)
                # Strip CF method prefix (already shown as column title)
                short_label = full_label.split(" - ", 1)[-1] if " - " in full_label else full_label
                ax_curr.plot(cf_df.columns, cf_df.iloc[i],
                             label=short_label,
                             ls=style['ls'], marker=style['marker'],
                             color=style['color'], linewidth=style['linewidth'],
                             markersize=style['markersize'])

            # Always plot baselines
            for bname, bvals in baselines.items():
                bstyle = simplified_plot_styling(bname)
                ax_curr.plot(df.columns, bvals,
                             label=bname,
                             ls=bstyle['ls'], marker=bstyle['marker'],
                             color=bstyle['color'], linewidth=bstyle['linewidth'],
                             markersize=bstyle['markersize'])


            # Axis formatting
            ax_curr.set_xticks(column_values)
            ax_curr.set_xticklabels([f"{e*100:.0f}%" for e in column_values], fontsize=9)
            ax_curr.invert_xaxis()
            ax_curr.set_ylim(y_min, y_max)
            ax_curr.grid(True, alpha=0.4)

            if row_idx == 0:
                ax_curr.set_title(fancy_generator(cf_method), fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax_curr.set_ylabel(fancy_model[model_key], fontsize=12, fontweight='bold')
            if row_idx == n_models - 1:
                ax_curr.set_xlabel("Target Coverage", fontsize=10)

    plt.suptitle(f"{name_dataset} - {selective.replace('_', ' ').replace('dict','')}", fontsize=14, fontweight='bold')

    # Shared legend: baselines first, then CF metric lines (unique labels only)
    _priority = {"Original score", "PlugInRule", "PlugInRuleAUC"}
    baseline_labels, cf_labels = {}, {}
    for ax_row in axes:
        for ax_ in ax_row:
            if ax_.get_visible():
                for h, lbl in zip(*ax_.get_legend_handles_labels()):
                    if lbl in _priority and lbl not in baseline_labels:
                        baseline_labels[lbl] = h
                    elif lbl not in _priority and lbl not in cf_labels:
                        cf_labels[lbl] = h
    all_legend = {**baseline_labels, **cf_labels}
    if all_legend:
        fig.legend(all_legend.values(), all_legend.keys(),
                   loc='lower center', ncol=7,
                   fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    fig_path = os.path.join(plot_dir, f"{fig_name}_{selective}.pdf")
    if show:
        plt.show()
        plt.close(fig)
    else:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"Saved figure to {fig_path}")
        plt.savefig(fig_path.replace(".pdf", ".png"), bbox_inches='tight', dpi=150)
        plt.close(fig)

def visualise_fairness_metrics(models, metric_dicts_class_0, metric_dicts_class_1, info,
                                name_dataset, plot_dir, fig_name="fairness_analysis"):
    """
    Visualize per-class rejection rates to identify fairness issues.

    Shows rejection rates for class 0 and class 1 side-by-side for each model,
    making it easy to spot when rejection is biased toward one class.

    Args:
        models: Dict of models
        metric_dicts_class_0: rejection_rate_class_0 from metric_dicts
        metric_dicts_class_1: rejection_rate_class_1 from metric_dicts
        info: Info dict with target_coverages
        name_dataset: Dataset name for title
        plot_dir: Directory to save plots
        fig_name: Base name for output files
    """
    target_coverages = info["target_coverages"]

    # Create figure: 2 columns (class 0, class 1) x N models rows
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 5*n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    class_names = ["Class 0", "Class 1"]
    metric_data = [metric_dicts_class_0, metric_dicts_class_1]

    for row_idx, model_key in enumerate(models.keys()):
        for col_idx, (class_name, metrics_dict) in enumerate(zip(class_names, metric_data)):
            ax = axes[row_idx, col_idx]

            # Plot all rejectors for this model and class
            if model_key in metrics_dict:
                methods_dict = metrics_dict[model_key]

                # Get max values for consistent y-axis
                max_val = 0
                for method_name, rates in methods_dict.items():
                    max_val = max(max_val, np.max(rates[:]))

                # Plot each rejector - use ALL coverage values like plot_line_graph_nb does
                coverage_indices = np.array(target_coverages[:])

                for method_name, rates in methods_dict.items():
                    style = simplified_plot_styling(method_name)
                    ax.plot(coverage_indices, rates[:],
                           label=fancy_names(method_name),
                           ls=style['ls'],
                           marker=style['marker'],
                           color=style['color'],
                           linewidth=style['linewidth'],
                           markersize=style['markersize'])

            # Formatting
            ax.set_xlabel("Target Coverage", fontsize=12)
            ax.set_ylabel("Rejection Rate", fontsize=12)
            if row_idx == 0:
                ax.set_title(f"{class_name}", fontsize=14, fontweight='bold')

            # Left column: show model name
            if col_idx == 0:
                ax.text(-0.25, 0.5, fancy_model[model_key],
                       transform=ax.transAxes, fontsize=14, fontweight='bold',
                       ha='right', va='center', rotation=90)

            ax.set_ylim(0, min(1.0, max_val * 1.1))
            ax.grid(True, alpha=0.3)

            # Format x-axis: plot all points, but only show ticks at multiples of 0.05
            # This follows the same pattern as plot_line_graph_nb
            target_cov_copy = np.array(target_coverages[:]).copy()
            tick_indices = target_cov_copy[((target_cov_copy*100)%10==0)]

            ax.set_xticks(tick_indices)
            ax.set_xticklabels([f"{e*100:.0f}%" for e in tick_indices], fontsize=10)

    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=5,
                  bbox_to_anchor=(0.5, -0.01), fontsize=11, frameon=False)

    plt.suptitle(f"Fairness Analysis: Rejection Rates by Class - {name_dataset}",
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    # change the xlim to be decreasing
    for ax_row in axes:
        for ax in ax_row:
            ax.invert_xaxis()
            
    # Save both formats
    pdf_path = os.path.join(plot_dir, f"{fig_name}.pdf")
    png_path = os.path.join(plot_dir, f"{fig_name}.png")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=100)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=100)
    print(f"Saved fairness visualization: {pdf_path}, {png_path}")
    plt.close()


def main():
    # --- SELECTED REJECTORS (second-pass filter) ---
    # Leave empty [] for first-pass (all rejectors used, top-k selected naively).
    # After statistical analysis, populate with the best combinations, e.g.:
    #   ["growingspheres_CFDistRejector_l2_average",
    #    "ils_CFDistRejector_cosine_average",
    #    "lore_CFDistRejector_l2_min"]
    # When non-empty, only these rejectors (plus PlugInRule/PlugInRuleAUC) are kept.
    SELECTED_REJECTORS = []
    # -----------------------------------------------

    # Parse command line arguments
    args = parse_arguments()
    
    # Create directories for output
    plot_dir = "results"
    #plot_dir = os.path.join(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Load data
    splits = load_data(config, args.dataset)
    
    # Load or train models
    models = load_or_train_models(config, splits, args.dataset)
    # Exclude LGBM and enforce model order matching the paper tables
    model_order = ['mlp', 'lip_mlp', 'random_forest', 'xgboost']
    models = {k: models[k] for k in model_order if k in models}
    print(name_dataset_command)
    print("processing dataset",name_dataset_command[fancy_dataset_names[args.dataset]])
    all_all_dataframes = {m:{} for m in [
                                    "rejection_quality_dict",
                                    "classification_quality_dict",
                                    "non_rejected_accuracy",
                                    "coverage",
                                    "calibration_coverage"
    ]}
    # Load distance statistics for the cf_methods 
    for cf_method in ["growingspheres","ils","ils_latent","lore"]:#,"dice"]:
        if cf_method == "ils_latent":
            args.latent = True
            cf_method = "ils"
        else:
            args.latent = False
        all_stats, all_stats_test = load_distance_stats(cf_method, args.dataset, args.latent)
        # Check if distances are computed
        for k in models.keys():
            print(k, "computed?", check_res(stats_dict=all_stats, k=k))
            print("\t", "test?", check_res(stats_dict=all_stats_test, k=k))
        
        # Convert lists to arrays
        all_stats = convert_lists_to_arrays(models, all_stats)
        all_stats_test = convert_lists_to_arrays(models, all_stats_test)
        # retain only the L2 and the cosine distance
        #all_stats = {k: {"distances": {m: all_stats[k]["distances"][m] for m in ["l2", "cosine"]}} for k in all_stats.keys()}
        #all_stats_test = {k: {"distances": {m: all_stats_test[k]["distances"][m] for m in ["l2", "cosine"]}} for k in all_stats_test.keys()}

        # Initialize selective classifiers
        rejectors, metric_dicts, info = initialize_rejectors(models, splits)
    
        # Calibrate basic rejectors
        rejectors = calibrate_basic_rejectors(models, rejectors, splits, all_stats, all_stats_test, info)
    
        rejectors = calibrate_distance_rejectors(models, rejectors, splits, all_stats, all_stats_test, info,calibrate_trees=False)
        
        # Evaluate all rejectors
        metric_dicts = evaluate_rejectors(models, rejectors, splits, metric_dicts, info)
        for jjjjj,selective in enumerate([
            "rejection_quality_dict",
            "classification_quality_dict",
            "non_rejected_accuracy",
            "coverage",
            "calibration_coverage"
            ]):
            dataframes_ = create_dataframes(models,metric_dicts, info,
                                            selective_metric=selective)
            # Create dataframes for visualization
            # dataframes = create_dataframes(models, metric_dicts, info,
            #                             selective_metric="non_rejected_accuracy")
            
            print(type(dataframes_["mlp"]), dataframes_["mlp"].columns)
            # dataframes is a dict of dataframes, the key is the model, the value the dataframe
            if cf_method == "ils":
                if args.latent:
                    cf_method = "ils<latent"
            for km in dataframes_.keys():
                # remove the name CFDistRejector from the indexes and replace it by the cf_method
                dataframes_[km].index = [(f"{cf_method}_{col}" if col not in ["PlugInRule", "PlugInRuleAUC"] else col) for col in dataframes_[km].index]
            for km in dataframes_.keys():
                if km not in all_all_dataframes[selective].keys():
                    all_all_dataframes[selective][km] = dataframes_[km]
                else:
                    # drop the raws relatively to the PlugInRule and PlugInRuleAUC
                    dataframes_[km] = dataframes_[km].drop(["PlugInRule", "PlugInRuleAUC"], axis=0)
                    all_all_dataframes[selective][km] = pd.concat([all_all_dataframes[selective][km], dataframes_[km]], axis=0)
    # Filter to selected rejectors if specified (second-pass mode)
    if SELECTED_REJECTORS:
        baselines = ["PlugInRule", "PlugInRuleAUC"]
        for selective in all_all_dataframes:
            for model_key in all_all_dataframes[selective]:
                df = all_all_dataframes[selective][model_key]
                keep = [r for r in df.index if r in SELECTED_REJECTORS or r in baselines]
                all_all_dataframes[selective][model_key] = df.loc[keep]

    fig_name = f"selective_classifier_{args.dataset}_results"
    dataset_name = fancy_dataset_names[args.dataset]

    if SELECTED_REJECTORS:
        # Second pass: generate plots only when a curated subset is selected
        print("Visualizing results (second pass — selected rejectors)")
        for jjjjj,selective in enumerate([
            "non_rejected_accuracy",
            "rejection_quality_dict",
            "classification_quality_dict",
            "coverage",
            "calibration_coverage"
            ]):
            visualise_results(models, all_all_dataframes[selective],
                               info, dataset_name,
                                plot_dir,fig_name=fig_name,
                                selective=selective)

        print("\nGenerating alternative visualization layout (rejectors × models)...")
        for jjjjj,selective in enumerate([
            "non_rejected_accuracy",
            #"rejection_quality_dict",
            "classification_quality_dict",
            "coverage"
            "calibration_coverage"
            ]):
            visualise_results_by_rejector(models, all_all_dataframes[selective],
                                         info, dataset_name,
                                         plot_dir, fig_name=f"{fig_name}_by_rejector",
                                         selective=selective)

        # Generate fairness analysis plot: per-class rejection rates
        print("\nGenerating fairness analysis (per-class rejection rates)...")
        visualise_fairness_metrics(models,
                                metric_dicts["rejection_rate_class_0"],
                                metric_dicts["rejection_rate_class_1"],
                                info, dataset_name, plot_dir,
                                fig_name=f"{fig_name}_fairness")
    
    else:
        print("Skipping plots, only saving general_results.pkl")

    if len(SELECTED_REJECTORS)==0:
        # finally save the results in a pickle file, so that we can load them later,
        # the file is a pickle. If there exist the file CONCAT_RESULTS.pkl,
        # it will be loaded and the new results will be updated (it is a huge dictionary, ù
        # the first key is the dataset name, the value is the all_all_dataframes dict)
        if not os.path.exists(f"general_results.pkl"):
            already_computed = {args.dataset: all_all_dataframes}
        else:
            with open(f"general_results.pkl", "rb") as f:
                already_computed = pickle.load(f)
            already_computed[args.dataset] = all_all_dataframes

        with open(f"general_results.pkl", "wb") as f:
            pickle.dump(already_computed, f)

if __name__ == "__main__":
    main()