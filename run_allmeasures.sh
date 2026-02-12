#!/bin/bash
#cat banner2.txt
datasets=("german_credit" "toy_dataset" "adult48k")
datasets=("adult48k" "german_credit" "toy_dataset")
datasets=("toy_dataset" "german_credit" "adult48k" "breast_cancer")
i=0
for dataset in "${datasets[@]}"; do
    # if i%2==0; then cat banner.txt; fi else cat banner2.txt
    cat banner.txt
    start_time=$(date +%s) 
    command="python 4_selective_classifiers.py --dataset $dataset > latentlearnTO$dataset.OUT 2>&1 &"
    echo $command
    eval $command
    i=$((i+1))

    wait
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "Elapsed time in :$((elapsed_time / 3600)) hours, $((elapsed_time / 60)) minutes and $((elapsed_time % 60)) seconds"

    # interrupt for debugging
    # if [ $i -eq 1 ]; then
    #     echo "Press any key to continue..."
    #     return

done