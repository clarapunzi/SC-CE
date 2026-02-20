#!/bin/bash
cat banner.txt
datasets=("toy_dataset" "german_credit" "adult48k" "breast_cancer")
cf_generators=("growingspheres" "dice" "ils" "lore")

for dataset in "${datasets[@]}"; do
    for cf_generator in "${cf_generators[@]}"; do
        command="python main.py --dataset $dataset --cf_method $cf_generator > OUT_$dataset.$cf_generator.OUT 2>&1 &"
        echo $command
        eval $command
        wait
    done
done

