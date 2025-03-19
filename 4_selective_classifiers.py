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
from src.data_processor import DataProcessor, fancy_dataset_names
from src.model_trainer import ModelTrainer
from src.utils import compute_selective_metrics
from OLD_src.Lib.L2R.code.model_agnostic import PlugInRule, PlugInRuleAUC, SCRoss
import rejectmodels.CFDistRejector as cfdr
import rejectmodels.CFTreeRejector as cftree

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Selective Classifiers Analysis for Counterfactual Examples')
    parser.add_argument('--dataset', type=str, required=True, choices=['adult48k', 'german_credit', 'toy_dataset'],
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
    """Define the distance metrics to focus on"""
    """
    """
    return ["cosine","l2"]
    return [
        'inf', 'cosine', 'braycurtis',
        'chebyshev', 'l2', 'minkowski', 'wasserstein', 'l1', 'mae', 'sqeuclidean'
        ]

def initialize_rejectors(models, splits):
    """Initialize selective classifiers and metrics dictionaries"""
    # Target coverages
    target_coverages_n = 30
    target_coverages = list(np.round(np.linspace(0.001, 1, target_coverages_n - 6)[::-1][1:],2))
    target_coverages = sorted(list(set(target_coverages +
                                [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
                                        )), reverse=True)
    target_coverages_n = len(target_coverages)
    n = len(splits['X_test'])
    
    focus_metrics = get_focus_metrics()
    
    # Initialize dictionaries for different rejector types
    # Basic rejectors
    plug_in_rulers = {}
    selected_plg = {}
    plug_in_rulers_auc = {}
    selected_plg_auc = {}
    # distance-based rejectors
    l2_rejectors_distances = {}
    selected_l2_rejectors_distances = {}
    
    # tree based rejectors
    tree_rejectors = {}
    selected_tree_rejectors = {}

    # Dictionaries for metrics
    rejected_by_coverage = {}
    classification_quality_dict = {}
    rejection_quality_dict = {}
    included_samples = {}
    all_original_scores = {}
    
    # Initialize the selective classifiers
    for k in models.keys():      
        # Baseline methods
        plug_in_rulers[k] = PlugInRule(model=models[k])
        plug_in_rulers_auc[k] = PlugInRuleAUC(model=models[k])
        all_original_scores[k] = accuracy_score(splits['y_test'], models[k].predict(splits['X_test']))
        
        # Result dictionaries
        rejected_by_coverage[k] = {}
        classification_quality_dict[k] = {}
        rejection_quality_dict[k] = {}
        included_samples[k] = {}
        
        # Initialize for basic methods
        for selective_c in ["PlugInRule", "PlugInRuleAUC"]:
            rejected_by_coverage[k][selective_c] = np.zeros(target_coverages_n)
            classification_quality_dict[k][selective_c] = np.zeros(target_coverages_n)
            rejection_quality_dict[k][selective_c] = np.zeros(target_coverages_n)
            included_samples[k][selective_c] = np.zeros(target_coverages_n)
    
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
                    rejected_by_coverage[k][selective_c] = np.zeros(target_coverages_n)
                    classification_quality_dict[k][selective_c] = np.zeros(target_coverages_n)
                    rejection_quality_dict[k][selective_c] = np.zeros(target_coverages_n)
                    included_samples[k][selective_c] = np.zeros(target_coverages_n)

                    # Initialize for tree-based rejectors
                    selective_c = "CFTreeRejector_" + metric + "_" + m_type + ("_gamma" if gamma_flag else "")
                    tree_rejectors[k][selective_c] = cftree.CFTreeRejector(
                        model=models[k],
                        coverages=target_coverages
                    )
                    # Initialize result dictionaries for this rejector
                    rejected_by_coverage[k][selective_c] = np.zeros(target_coverages_n)
                    classification_quality_dict[k][selective_c] = np.zeros(target_coverages_n)
                    rejection_quality_dict[k][selective_c] = np.zeros(target_coverages_n)
                    included_samples[k][selective_c] = np.zeros(target_coverages_n)
                    
    
    # Group metric dictionaries
    metric_dicts = {
        "rejected_by_coverage": rejected_by_coverage,
        "classification_quality_dict": classification_quality_dict,
        "rejection_quality_dict": rejection_quality_dict,
        "included_samples": included_samples
    }
    
    rejectors = {
        "plug_in_rulers": plug_in_rulers,
        "selected_plg": selected_plg,
        "plug_in_rulers_auc": plug_in_rulers_auc,
        "selected_plg_auc": selected_plg_auc,
        "l2_rejectors_distances": l2_rejectors_distances,
        "selected_l2_rejectors_distances": selected_l2_rejectors_distances,
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
        
        # Calibrate PlugInRuleAUC
        plug_in_ruler_auc = rejectors["plug_in_rulers_auc"][k]
        plug_in_ruler_auc.calibrate(
            X=splits["X_calibration"],
            y=splits["y_calibration"].values.reshape(-1),
            target_coverages=target_coverages
        )
        rejectors["selected_plg_auc"][k] = rejectors["plug_in_rulers_auc"][k].qband(splits["X_test"])
        
        print("Basic rejectors calibrated:")
        print(f"PlugInRule: {rejectors['selected_plg'][k][:5]}")
        print(f"PlugInRuleAUC: {rejectors['selected_plg_auc'][k][:5]}")
    
    return rejectors


def calibrate_distance_rejectors(models, rejectors, splits, all_stats, all_stats_test, info):
    """Calibrate all distance-based rejectors"""
    focus_metrics = info["focus_metrics"]
    rejectors["selected_l2_rejectors_distances"] = {}
    rejectors["selected_tree_rejectors"] = {}
    
    for k in tqdm(models.keys(), desc="Calibrating distance-based rejectors"):
        rejectors["selected_l2_rejectors_distances"][k] = {}
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


def evaluate_rejectors(models, rejectors, splits, metric_dicts, info):
    """Evaluate all rejector types"""
    target_coverages = info["target_coverages"]
    n = info["n"]
    
    for k in tqdm(models.keys(), desc="Evaluating rejectors"):        
        # Evaluate basic rejectors
        basic_rejectors = {
            "PlugInRule": {
                "selected_data": rejectors["selected_plg"][k],
                "rejector": rejectors["plug_in_rulers"][k]
            },
            "PlugInRuleAUC": {
                "selected_data": rejectors["selected_plg_auc"][k],
                "rejector": rejectors["plug_in_rulers_auc"][k]
            }
        }
        
        for selective_k, data in basic_rejectors.items():
            compute_selective_metrics(
                model_key=k,
                selected_data=data["selected_data"],
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
                classifier_type=selective_c,
                selective_classifier=rejectors["l2_rejectors_distances"][k][selective_c],
                splits=splits,
                target_coverages=target_coverages,
                n=n,
                metric_dicts=metric_dicts
            )
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


def create_dataframes(models, metric_dicts, info):
    """Create DataFrames for visualization"""
    target_coverages = info["target_coverages"]
    selective_metric = "rejected_by_coverage"
    
    dataframes = {}
    for k in models.keys():
        df = pd.DataFrame()
        for method, val in metric_dicts[selective_metric][k].items():
            df[method] = val
        df = df.T
        df.columns = target_coverages
        dataframes[k] = df
    
    return dataframes

def fancy_names(name):
    """Return a cooler name for the selective classifier"""
    if name == "PlugInRule":
        return "PlugInRule"
    elif name == "PlugInRuleAUC":
        return "PlugInRuleAUC"
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
        if name.split("_")[1] == "CFDistRejector":
            tree = ""
        elif name.split("_")[1] == "CFTreeRejector":
            tree = " (Tree)"
        distr = name.split("_")[-1]
        if distr == "mean":
            distr = "_{mean}"
        elif distr == "min":
            distr = "_{min}"
        elif distr == "max":
            distr = "_{max}"
        method = name.split("_")[0]

        if method == "ils<latent":
            method = "ILS$_{latent}$" 
        else:
            method = method.upper()
        distance = name.split("_")[2]
        
        newn= method+tree+" - "+distance+"$"+apex+distr+"$" + duplicate_flag
        print(name,newn)
        return newn

subpartial_yet_easier_results = {}
def visualize_results(models, dataframes, info, name_dataset, plot_dir,
                      fig_name="selective_classifiers",
                      show =False,
                      top_k=2+6 # Number of top methods to select
                      ):
    """Visualize results for each model"""
    target_coverages = info["target_coverages"]
    target_coverages_n = info["target_coverages_n"]
    all_original_scores = info["all_original_scores"]
    
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    # Scale limits
    minimum = np.array([all_original_scores[k] for k in all_original_scores.keys()]).min()
    vmax =    0    
    for k in models.keys():

        # Get indices of top performing methods by sum across coverages

        fake_auc = dataframes[k].copy().values[:,:target_coverages_n//2].sum(axis=1)
        # sort the fake_auc
        fake_auc = fake_auc.argsort()[::-1]
        # take the top_k rows
        fake_auc = fake_auc[:top_k]
        vmax = max(vmax, dataframes[k].values[fake_auc, :target_coverages_n//2].max())
    minimum-=0.007
    vmax+=0.007
    # use the fancy bar from tqdm
    for k in tqdm(models.keys(), desc="Visualizing results"):
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))
        
        # Use a basic colormap
        custom_cmap = fplt.parula
        # black and white
        custom_cmap = 'Greys_r'
        custom_cmap = 'inferno'
        
        # Select top performing methods
        df = dataframes[k].copy().iloc[:, :target_coverages_n//2]
        
        # it could happen that there is no difference between the methods (some rows are the same and differ only by the index)
        # so we need to drop the duplicates and modify the index of the duplicated rows that are kept by adding a suffix "_duplicate"
        old_index = df.index
        old_df = df.copy()
        print("Dropping duplicates")
        
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
                



        
        # Get indices of top performing methods by sum across coverages
        fake_auc = df.values.sum(axis=1).argsort()[::-1][:top_k]
        top_policies = list(fake_auc)
        
        # Ensure PlugInRule and PlugInRuleAUC are included
        if 0 not in top_policies:
            top_policies = [0] + top_policies[:-1]
        if 1 not in top_policies:
            top_policies = [1] + top_policies[:-1]
        # now place the PlugInRule and PlugInRuleAUC at the beginning
        top_policies = [0, 1] + [e for e in top_policies if e not in [0, 1]]
        top_policies_names = [fancy_names(fnam) for fnam in df.index[top_policies]]
        

        
        # Plot heatmap of top methods
        im = ax[0].imshow(df.values[top_policies, :], cmap=custom_cmap, vmin=minimum, vmax=vmax)
        #ax[0].set_title(f"Top Selective Classifiers for Model: {k}")
        ax[0].set_yticks(np.array([i for i in range(len(top_policies_names))]))
        ax[0].set_yticklabels(top_policies_names, rotation=0, fontsize=16)
        try:
            print(df.columns)
        except Exception as e:
            print(e)
            print(df.head())
        ax[0].set_xticks(range(target_coverages_n//2))
        ax[0].set_xticklabels([str(round(e, 2)) for e in df.columns], rotation=0)
        
        # Add colorbar
        plt.colorbar(im, ax=ax[0], label="Rejection Quality")
        
        # Plot line graph of top methods
        for j, policy_idx in enumerate(top_policies):
            if policy_idx ==0:  # PlugInRule and PlugInRuleAUC
                ls = '--'
                marker = 'd'
            elif policy_idx == 1:
                ls = '--'
                marker = 's'
            else:
                ls = '-'
                marker = 'o'
            
            label = fancy_names(df.index[policy_idx])
            ax[1].plot(df.columns,df.values[policy_idx, :][::-1], label=label, marker=marker, ls=ls)
        
        ax[1].set_xticks(df.columns)#range(target_coverages_n//2))
        ax[1].set_xticklabels([str(round(e, 2)) for e in df.columns[::-1]], rotation=0)
        # Add line for original score
        ax[1].axhline(
            all_original_scores[k],
            label=f"Original {k}",
            color='k',
            ls='--'
        )
        
        ax[1].set_xlabel("Target Coverage")
        ax[1].set_ylabel("Rejection Accuracy")
        ax[1].legend()
        ax[1].grid(True)
        ax[1].set_ylim(minimum, vmax)
        
        plt.suptitle(
            f"{name_dataset} - Model: {k}",
              # {args.method.upper()} {'Latent' if args.latent else ''} - 
            fontsize=16
        )
        # Save figure
        fig_path = os.path.join(
            plot_dir,
            fig_name+"_"+k+".pdf"
        )
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(fig_path)
            print(f"Saved figure to {fig_path}")
            
            # Also save PNG for easy viewing
            plt.savefig(fig_path.replace(".pdf", ".png"))
            plt.close(fig)

def main():
    """Main function - orchestrates the overall workflow"""
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
    
    all_dataframes = {}
    # Load distance statistics for the cf_methods 
    for cf_method in ["ils","ils_latent","lore"]:#,"dice"]:
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
    
        rejectors = calibrate_distance_rejectors(models, rejectors, splits, all_stats, all_stats_test, info)
        

        # Evaluate all rejectors
        metric_dicts = evaluate_rejectors(models, rejectors, splits, metric_dicts, info)
    
        # Create dataframes for visualization
        dataframes = create_dataframes(models, metric_dicts, info)
        print(type(dataframes["mlp"]), dataframes["mlp"].columns)
        # dataframes is a dict of dataframes, the key is the model, the value the dataframe
        if cf_method == "ils":
            if args.latent:
                cf_method = "ils<latent"
        for km in dataframes.keys():
            # remove the name CFDistRejector from the indexes and replace it by the cf_method
            dataframes[km].index = [(f"{cf_method}_{col}" if col not in ["PlugInRule", "PlugInRuleAUC"] else col) for col in dataframes[km].index]
        for km in dataframes.keys():
            if km not in all_dataframes.keys():
                all_dataframes[km] = dataframes[km]
            else:
                # drop the raws relatively to the PlugInRule and PlugInRuleAUC
                dataframes[km] = dataframes[km].drop(["PlugInRule", "PlugInRuleAUC"], axis=0)
                all_dataframes[km] = pd.concat([all_dataframes[km], dataframes[km]], axis=0)
    # Visualize results using the all_dataframes
    print("Visualizing results")
    # now is different, because we have all the cf_methods, so we need to iterate over the dataset only
    fig_name = f"selective_classifier_{args.dataset}_results"
    dataset_name = fancy_dataset_names[args.dataset]
    visualize_results(models, all_dataframes, info, dataset_name, plot_dir,fig_name=fig_name)

if __name__ == "__main__":
    main()