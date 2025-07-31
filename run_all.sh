#!/bin/bash
cat banner.txt
datasets=("german_credit" "adult48k")

for dataset in "${datasets[@]}"; do
    command="python main.py --dataset $dataset --cf_method lore > L2LOREOUT_$dataset.OUT 2>&1 &"
    echo $command
    eval $command
    wait
done