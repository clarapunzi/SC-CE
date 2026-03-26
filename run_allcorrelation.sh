#!/bin/bash

datasets=("adult48k" "german_credit" "toy_dataset" "breast_cancer")
methods=("dice" "ils" "lore" "growingspheres")
datasets=("breast_cancer")
methods=("ils")
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    for method in "${methods[@]}"; do
        echo "  Running method: $method"
        python 3_correlation_and_distances_plot.py --dataset "$dataset" --method "$method" &
        if [ "$method" == "ils" ]; then
            python 3_correlation_and_distances_plot.py --dataset "$dataset" --method "ils" --latent &
        fi
    done
    wait  # Wait for all methods to finish before moving to next dataset
    echo "Dataset $dataset completed"
done