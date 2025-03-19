#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script performs correlation and distance analysis for counterfactual examples.
It analyzes the relationship between distances and model confidence, creating
various plots to visualize these relationships.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import time
import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import IsolationForest
from matplotlib import ticker
import importlib

# Create data processor and load splits
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer


def check_res(stats_dict, k="mlp"):
    """
    This function returns True if the distances related to the model k are already computed
    """
    # try if the list containing the mean dists has length > 0
    try:
        print("(" + str(len(stats_dict[k]['distances']['l2']['mean'])), "elements computed)")
        return len(stats_dict[k]['distances']['l2']['mean']) >= 200
    except Exception as e:
        print(e)
        return False

def create_plot_distances(dictionary,
                          metric="l2",
                          sortby="mean",
                          axe=None,
                          title=None,
                          legend=False,
                          colors=None,
                          ylim=None,
                          anomalies=None,
                          anomalies_test=None):
    if colors is None:
        colors = {"min": "blue", "max": "red", "mean": "black"}

    if axe is None:
        axe = plt.gca()

    min_color = colors["min"]
    max_color = colors["max"]
    mean_color = colors["mean"]

    argsorted = dictionary[sortby].argsort()
    rank_m = dictionary["min"][argsorted]
    rank_M = dictionary["max"][argsorted]
    rank_mean = dictionary["mean"][argsorted]

    if "correct" in dictionary.keys():
        rank_corrects = dictionary["correct"][argsorted]

        # shade the background with vertical lines colored by the corrects
        # if the model is correct the line is green, otherwise it's red
        # the alpha is 0.2, use axvline to plot the vertical lines
        for i in range(len(rank_corrects)):
            if rank_corrects[i]:
                axe.axvline(i, color="yellow", alpha=0.2)
            else:
                axe.axvline(i, color="purple", alpha=0.2)

    axe.fill_between(np.arange(len(rank_m)), rank_m, rank_M, alpha=0.5, label="Min-Max")
    axe.plot(rank_m, alpha=0.5, color=min_color)
    axe.plot(rank_mean, label="Mean", alpha=0.5, color=mean_color)
    axe.plot(rank_M, alpha=0.5, color=max_color)

    # plot the "smoothed" values
    axe.plot(np.arange(len(rank_m)), pd.Series(rank_m).rolling(5).mean(), label="Min", color=min_color)
    axe.plot(np.arange(len(rank_mean)), pd.Series(rank_mean).rolling(5).mean(), label="Mean", color=mean_color)
    axe.plot(np.arange(len(rank_M)), pd.Series(rank_M).rolling(5).mean(), label="Max", color=max_color)

    if legend:
        axe.legend(fontsize=15)
    axe.set_xlabel("$\\delta$s sorted by " + sortby, fontsize=15)
    axe.set_ylabel(metric + " - Distance", fontsize=15)
    if title is not None:
        axe.set_title(title, fontsize=18)
    axe.grid()
    if ylim is not None:
        axe.set_ylim(ylim[0], ylim[1])

    if "Predict Proba" in dictionary.keys():
        ax2 = axe.twinx()
        rank_prob = dictionary["Predict Proba"][argsorted]
        ax2.plot(np.arange(len(rank_m)), rank_prob, label="Confidence", color="black", linestyle="--")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Probability", fontsize=15, rotation=270)
        ax2.set_yticks(np.arange(0, 1.1, 0.25))
        if legend:
            # location is lower right
            ax2.legend(fontsize=15, loc="lower right")
    else:
        ax2 = axe.twiny()

    # place some ticks x shaped in correspondence of the anomalies
    if anomalies is not None and len(argsorted) == len(anomalies):
        ax2.scatter(np.arange(len(rank_m))[anomalies[argsorted]],
                 0.05 * np.ones(anomalies.sum()),
                 marker="x", color="black", label="Anomalies")
    elif anomalies_test is not None:
        ax2.scatter(np.arange(len(rank_m))[anomalies_test[argsorted]],
                 0.05 * np.ones(anomalies_test.sum()),
                 marker="x", color="black", label="Anomalies test")

    axe.set_title(title + sortby)
    five_perc = int(len(rank_m) * 0.05)
    axe.set_xlim([-five_perc, len(rank_m) + five_perc])
    axe.xaxis.set_tick_params(labelsize=15)
    axe.yaxis.set_tick_params(labelsize=15)
    # tight layout
    plt.tight_layout()
    return axe

def main():
    parser = argparse.ArgumentParser(description='Correlations and Distance Plots for Counterfactual Analysis')
    parser.add_argument('--dataset', type=str, required=True, choices=['adult48k', 'german_credit', 'toy_dataset'],
                        help='Name of the dataset to analyze')
    parser.add_argument('--method', type=str, required=True, choices=['dice', 'ils', 'lore'],
                        help='Counterfactual generation method')
    parser.add_argument('--latent', action='store_true',
                        help='Use latent space for counterfactual generation')

    args = parser.parse_args()

    dt_name = args.dataset
    cf_method = args.method
    cf_latent = args.latent

    # Import modules from src
    try:
        import src.utils as utl
        import src.fancy_plots as fplt
    except ImportError:
        print("Error: Required modules not found. Make sure src directory is in the path.")
        return

    importlib.reload(utl)
    importlib.reload(fplt)

    # Create required directories
    os.makedirs(os.path.join("plots", dt_name), exist_ok=True)
    os.makedirs("correlations", exist_ok=True)

    # Dataset name mapping
    name_dataset = {
        "german_credit": "German Credit",
        "adult48k": "Adult",
        "toy_dataset": "Toy Dataset"
    }[dt_name]

    # Find configuration file and load data
    config = utl.load_config('config.yaml')
    cf_results = config['paths']['counterfactuals']

    data_processor = DataProcessor(config=config)
    splits = data_processor.load_splits(dt_name)

    # Train models
    model_trainer = ModelTrainer(config)
    models = model_trainer.train_model(
        X_train=splits['X_train'],
        y_train=splits['y_train'],
        X_test=splits['X_test'],
        y_test=splits['y_test'],
        dataset_name=dt_name
    )
    print("Available models:", models.keys())

    # Load distance files
    fname = os.path.join(cf_method, dt_name + "_ranking_latent.pkl")
    fname_test = os.path.join(cf_method, dt_name + "_ranking_test_latent.pkl")

    if not cf_latent:
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
        return

    if utl.check_file_exists(fname_test):
        with open(fname_test, 'rb') as f:
            all_stats_test = pickle.load(f)
            print("Loaded", fname_test)
    else:
        print("!" * 6, fname_test, "not found")
        return

    # Check if distances are computed for each model
    for k in models.keys():
        print(k, "computed?", check_res(stats_dict=all_stats, k=k))
        print("\t", "test?", check_res(stats_dict=all_stats_test, k=k))

    # Convert lists to numpy arrays for all_stats
    for k in models.keys():
        for metric in all_stats[k]['distances'].keys():
            for stat in ["min", "max", "mean", "std"]:
                try:
                    all_stats[k]['distances'][metric][stat] = np.array(all_stats[k]['distances'][metric][stat])
                except Exception as e:
                    print(e)
                    print(k, metric, stat, len(all_stats[k]['distances'][metric][stat]))
                    for e in all_stats[k]['distances'][metric][stat]:
                        print(e)

    for k in models.keys():
        all_stats[k]['probabs'] = np.array(all_stats[k]['probabs'])
        all_stats[k]['corrects'] = np.array(all_stats[k]['corrects']).reshape(-1)
        all_stats[k]['mean_conf'] = np.array(all_stats[k]['mean_conf'])

    # Convert lists to numpy arrays for all_stats_test
    for k in models.keys():
        for metric in all_stats_test[k]['distances'].keys():
            for stat in ["min", "max", "mean", "std"]:
                try:
                    all_stats_test[k]['distances'][metric][stat] = np.array(all_stats_test[k]['distances'][metric][stat])
                except Exception as e:
                    print(e)
                    print(k, metric, stat, len(all_stats_test[k]['distances'][metric][stat]))
                    for e in all_stats_test[k]['distances'][metric][stat]:
                        print(e)

    for k in models.keys():
        all_stats_test[k]['probabs'] = np.array(all_stats_test[k]['probabs'])
        all_stats_test[k]['corrects'] = np.array(all_stats_test[k]['corrects']).reshape(-1)
        all_stats_test[k]['mean_conf'] = np.array(all_stats_test[k]['mean_conf'])

    # Calculate correlations for both calibration and test sets
    all_all_correlations = {}

    colors = {"min": "blue", "max": "red", "mean": "black"}

    for compute_test in [False, True]:
        all_stats_tmp = all_stats_test if compute_test else all_stats
        verbose = False
        correlations = {}
        metrics = list(all_stats_tmp[list(models.keys())[0]]['distances'].keys())

        for metric in metrics:
            correlations[metric] = {}
            correlations["found_cfs"] = {}

            for k in models.keys():
                mins = all_stats_tmp[k]['distances'][metric]['min'][:]
                good_idxs = mins != np.inf
                mins = mins[good_idxs]

                if good_idxs.sum() == 0:
                    print("No counterfactuals found for model", k, "and metric", metric)
                    correlations[metric][k] = {"min": np.nan, "max": np.nan, "mean": np.nan}
                    correlations["found_cfs"][k] = 0
                    continue

                maxs = all_stats_tmp[k]['distances'][metric]['max'][:][good_idxs]
                means = all_stats_tmp[k]['distances'][metric]['mean'][:][good_idxs]
                probs = all_stats_tmp[k]['probabs'][:][good_idxs]

                correlations[metric][k] = {}
                correlations[metric][k]["min"] = np.corrcoef(mins, probs.max(axis=1))[0, 1]
                correlations[metric][k]["max"] = np.corrcoef(maxs, probs.max(axis=1))[0, 1]
                correlations[metric][k]["mean"] = np.corrcoef(means, probs.max(axis=1))[0, 1]

                if verbose:
                    print(f"{metric} correlation for {k}:")
                    print("Min", np.round(correlations[metric][k]["min"], 3))
                    print("Max", np.round(correlations[metric][k]["max"], 3))
                    print("Mean", np.round(correlations[metric][k]["mean"], 3))
                    print()

                correlations["found_cfs"][k] = good_idxs.sum()

        # Compare confidence of the model with mean confidence of counterfactuals
        correlations["mean_conf"] = {}
        for k in models.keys():
            good_idx = all_stats_tmp[k]['distances']['l2']['min'] != np.inf
            if good_idx.sum() == 0:
                correlations["mean_conf"][k] = np.nan
            else:
                correlations["mean_conf"][k] = np.corrcoef(
                    all_stats_tmp[k]['mean_conf'][good_idx[:]].max(axis=1),
                    all_stats_tmp[k]['probabs'][good_idx[:]].max(axis=1)
                )[0, 1]

        # Create correlation matrix
        all_correlations = []
        for metric in metrics:
            metric_corr = []
            for stat in ["mean", "min", "max"]:
                row = [correlations[metric][model][stat] for model in models.keys()]
                metric_corr.append(row)
            all_correlations.extend(metric_corr)

        all_correlations.append([np.nan for model in models.keys()])
        all_correlations.append([correlations["mean_conf"][model] for model in models.keys()])
        all_correlations.append([np.nan for model in models.keys()])

        all_correlations = np.array(all_correlations)

        # Create labels for y-axis
        y_labels = []
        for metric in metrics:
            for stat in ["mean", "min", "max"]:
                y_labels.append(f"{stat}")
        y_labels.append("")
        y_labels.append("mean")
        y_labels.append("")

        total_metrics = len(metrics) + 1
        rows_per_metric = 3

        # Plot correlation heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, len(metrics) * 0.55))
        gold_map_r = fplt.build_cmap(0.1, '#3c6b5e', '#fff1f9', '#deb062')
        sns.heatmap(all_correlations, annot=True, ax=ax, vmin=-1, vmax=1, cmap=gold_map_r)
        ax.set_yticks(0.5 + np.arange(len(y_labels)), y_labels, rotation=0)
        ax.set_xticklabels([model + "\n" + str(correlations["found_cfs"][model]) for model in models.keys()])
        ax.set_title("Correlations between distances and output probability of the model")
        plt.xlabel("Models")

        # Add horizontal lines between metric sections
        for i in range(total_metrics + 1):
            ax.axhline(y=i * rows_per_metric, color='black', linewidth=1)

        # Set up ticks for metrics' names
        ax_metrics = ax.twinx()
        ax_metrics.spines["left"].set_position(("axes", -0.2))
        ax_metrics.tick_params('both', length=3, direction='out', which='major')
        ax_metrics.yaxis.set_ticks_position("left")
        ax_metrics.yaxis.set_label_position("left")

        # Set metric positions and labels
        metric_positions = [(i * 3 + 1.5) for i in range(len(metrics) + 1)]  # Center of each metric section
        ax_metrics.set_ylim(ax.get_ylim())
        ax_metrics.set_yticks(metric_positions)
        ax_metrics.set_yticklabels([m.upper() for m in metrics] + ["mean CONF on cfs"])
        ax_metrics.set_ylabel("Distance Metrics", rotation=90, fontsize=15)

        # Get a new axis
        ax_ = ax.twinx()
        ax_.spines["left"].set_position(("axes", -.2))
        ax_.tick_params('both', length=7, direction='in', which='major')
        ax_.yaxis.set_ticks_position("left")

        ticks_positions = (3 * np.arange(0, len(metrics) + 2))
        ticks_positions = ticks_positions / ticks_positions.max()
        print(ticks_positions)
        ax_.set_yticks(ticks_positions)
        ax_.set_yticklabels([""] * len(metrics) + [""] * 2)

        ax_.spines["right"].set_visible(False)
        ax_metrics.spines["right"].set_visible(False)

        plt.suptitle(
            name_dataset +
            " - " + ("test set" if compute_test else "calibration set") +
            " - " + cf_method.upper() +
            (" latent" if cf_latent else ""),
            fontsize=20
        )
        plt.tight_layout()

        # Save the figure
        fig_name = os.path.join(
            "plots", dt_name,
            "Correlations_" + dt_name + "_" + ("test" if compute_test else "cal") + "_" + cf_method + ("_latent" if cf_latent else "") + ".pdf"
        )
        plt.savefig(fig_name)
        print(f"Saved correlation heatmap to {fig_name}")


        plt.close(fig)
        all_all_correlations[("test" if compute_test else "cal")] = all_correlations

    # Save correlation data
    fname = os.path.join(
        "correlations",
        cf_method + "_" + ("latent_" if cf_latent and cf_method == 'ils' else "") + dt_name + "_correlations_cal.pkl"
    )
    print(f"Saving correlation data to {fname}")
    with open(fname, "wb") as f:
        #
        pickle.dump(all_all_correlations["cal"], f)

    fname_test = fname.replace("_cal.pkl", "_test.pkl")
    with open(fname_test, "wb") as f:
        pickle.dump(all_all_correlations["test"], f)
    print(f"Saving test correlation data to {fname_test}")

    # Find metrics with highest correlations
    max_correlations = []
    for j, m in enumerate(metrics):
        mmax = max([abs(correlations[m][model]["min"]) for model in models.keys()])
        mmean = max([abs(correlations[m][model]["mean"]) for model in models.keys()])
        Mmax = max([abs(correlations[m][model]["max"]) for model in models.keys()])

        # If any of the three is nan, place -inf
        if np.isnan(mmax) or np.isnan(mmean) or np.isnan(Mmax):
            max_correlations.append((j, -np.inf))
        else:
            max_correlations.append((j, max([mmax, mmean, Mmax])))

    max_correlations.append((j + 1, max([abs(correlations["mean_conf"][model]) for model in models.keys()])))
    max_correlations = np.array(sorted(max_correlations, key=lambda x: x[1], reverse=True))

    # Plot max correlations
    plt.figure(figsize=(10, 3))
    plt.bar(range(len(max_correlations)), max_correlations[:, 1])
    plt.xticks(
        range(len(max_correlations)),
        [m + " - " + str(j + 1) for j, m in enumerate(np.array(metrics + ["mean_conf"])[max_correlations[:, 0].astype(int)])],
        rotation=90
    )
    plt.grid(axis='y')
    plt.title("Max absolute correlation for each metric for " + name_dataset)
    plt.tight_layout()

    plt.savefig(os.path.join(
        "plots", dt_name,
        "Max_correlations_" + dt_name + "_" + cf_method + ("_latent" if cf_latent else "") + ".pdf"
    ))

    # Select top metrics with highest correlations
    top_metrics = 5
    focus_metrics = np.array(metrics + ["mean_conf"])[max_correlations[:top_metrics, 0].astype(int)]
    focus_metrics = [fm for fm in focus_metrics if fm != "mean_conf"]

    if "l2" not in focus_metrics:
        # add the l2 metric at the start
        focus_metrics = ["l2"] + focus_metrics
    else:
        # place the l2 metric at the start
        focus_metrics.remove("l2")
        focus_metrics = ["l2"] + focus_metrics

    print("Focus metrics:", focus_metrics)

    # Get class balancing information
    v, c = utl.print_balancing(splits["y_calibration"])
    class_w = np.array([1 - c[i] / c.sum() for i, v1 in enumerate(v)])

    utl.print_balancing(splits["y_test"])
    utl.print_balancing(splits["y_train"])

    # Find outliers using isolation forest
    anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
    anomaly_detector.fit(splits["X_train"])
    anomalies = anomaly_detector.predict(splits["X_calibration"]) < 0
    print("Found", anomalies.sum(), "anomalies")

    anomaly_detector2 = IsolationForest(contamination=0.05, random_state=42)
    anomaly_detector2.fit(splits["X_train"], sample_weight=class_w[splits["y_train"]].reshape(-1))
    anomalies2 = anomaly_detector2.predict(splits["X_calibration"]) < 0
    print("Found", anomalies2.sum(), "anomalies with class weights")

    anomalies_test = anomaly_detector.predict(splits["X_test"]) < 0
    print("Found", anomalies_test.sum(), "anomalies in the test set")

    # Create distance plots for each focus metric
    for metric in focus_metrics:
        fig, axs = plt.subplots(4, 4, figsize=(20, 20))
        axs = axs.flatten()

        c = 0
        max_ = np.max([
            pd.Series(all_stats[k]["distances"][metric]["max"][
                pd.Series(all_stats[k]["distances"][metric]["max"]).apply(np.isfinite)
            ]).max(skipna=True)
            for k in models.keys()
        ])

        min_ = np.min([
            pd.Series(all_stats[k]["distances"][metric]["min"][
                pd.Series(all_stats[k]["distances"][metric]["min"]).apply(np.isfinite)
            ]).min(skipna=True)
            for k in models.keys()
        ])

        data_range = max_ - min_
        max_ylim = max_ + (data_range * 0.33)  # Add 33% of range above max
        min_ylim = min_ - (data_range * 0.1)   # Subtract 10% of range below min
        min_ylim = max(-0.05, min_ylim)

        print("max", max_, "min", min_, "max_ylim", max_ylim, "min_ylim", min_ylim)

        for name in ["Predict Proba"] + ["mean", "min", "max"]:
            for k in models.keys():
                create_plot_distances(
                    dictionary={
                        "mean": all_stats[k]["distances"][metric]["mean"],
                        "std": all_stats[k]["distances"][metric]["std"],
                        "min": all_stats[k]["distances"][metric]["min"],
                        "max": all_stats[k]["distances"][metric]["max"],
                        "Predict Proba": all_stats[k]["probabs"].max(axis=1),
                        "correct": all_stats[k]["corrects"]
                    },
                    metric=metric,
                    sortby=name,
                    axe=axs[c],
                    title=k + " - ",
                    legend=(True if c == 0 else False),
                    colors=colors,
                    ylim=None,  # (min_ylim, max_ylim),
                    anomalies=anomalies,
                    anomalies_test=anomalies_test
                )
                c += 1

        date = time.strftime("%Y-%m-%d")
        plt.suptitle(
            name_dataset + " - " + metric.upper() + " " + cf_method.upper() + (" latent" if cf_latent else ""),
            fontsize=18
        )
        plt.tight_layout()

        fig_name = os.path.join(
            "plots", dt_name,
            "counterfactuals_" + cf_method + "_" + dt_name + "_" + metric + "_" +
            cf_method + ("_latent" if cf_latent else "") + ".pdf"
        )

        plt.savefig(fig_name)
        print("Saved Figure at", fig_name)

        plt.close(fig)

    # Plot distributions
    cumulative = False
    top_k = 3
    fig, axs = plt.subplots(top_k + 1, 4, figsize=(20, 3 * top_k))
    axs = axs.flatten()

    for i, metric in enumerate(focus_metrics[:top_k]):
        for name in ["mean", "min", "max"]:
            for j, k in enumerate(models.keys()):
                indx = 4 * i + j
                # plot the distribution of the distances (calibration and test)
                sns.kdeplot(
                    all_stats[k]["distances"][metric][name],
                    ax=axs[indx],
                    cumulative=cumulative,
                    label='calibration ' + name,
                    lw=3,
                    color=colors[name]
                )
                sns.kdeplot(
                    all_stats_test[k]["distances"][metric][name],
                    ax=axs[indx],
                    cumulative=cumulative,
                    label='test ' + name,
                    ls="--",
                    color=colors[name],
                    fill=True
                )
        if i == 0:
            axs[0].legend(loc="upper right")

    # Remove y ticks and labels
    for j, ax in enumerate(axs):
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        if j % 4 == 0 and j <= 4 * top_k:
            ax.yaxis.set_ticks_position("left")
            ax.yaxis.set_label_position("left")
            ax.set_ylabel(focus_metrics[j // 4], fontsize=15)
            ax.yaxis.set_tick_params(labelsize=15)
        else:
            ax.set_ylabel("")

    for indx, k in enumerate(models.keys()):
        indx += 4 * top_k
        sns.kdeplot(
            all_stats[k]["probabs"].max(axis=1),
            ax=axs[indx],
            cumulative=cumulative,
            label='calibration confs',
            lw=3,
            color="k"
        )
        sns.kdeplot(
            all_stats_test[k]["probabs"].max(axis=1),
            ax=axs[indx],
            cumulative=cumulative,
            label='test confs',
            ls="--",
            color='k',
            fill=True
        )
        axs[indx].set_xlim(0.45, 1.05)
        # plot vertical lines at 0.5 and 1.0
        axs[indx].axvline(0.5, color="red", lw=1, ls='--')
        axs[indx].axvline(1.0, color="red", lw=1, ls='--')
        if indx % 4 == 0:
            axs[indx].legend(loc="upper left")
            axs[indx].set_ylabel("Confidence", fontsize=15)

    date = time.strftime("%Y-%m-%d")
    plt.suptitle(
        name_dataset + " Distributions for " + cf_method + (" latent" if cf_latent else ""),
        fontsize=20
    )
    plt.tight_layout()

    fig_name = os.path.join(
        "plots", dt_name,
        "distributions_" + dt_name + "_" + cf_method + ("_latent" if cf_latent else "") + ".pdf"
    )
    plt.savefig(fig_name)
    print("Saved Figure at", fig_name)
    plt.close(fig)

if __name__ == "__main__":
    main()