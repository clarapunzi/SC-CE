<!--#
Dice Clara, che assumeva la continuità "locale".
I piani d'attacco sono 2.
1 rilanciamo gli experimients guardando i modelli con la continuità Lipshitz continuo.

Train a mlp regolarizzato o SVM
posso rilassare questa cosa per qualsiasi modello? Mi sto perdendo qualocosa ma è così importante?  

Le proprietà dei controfattuali 

-->
# SC-CE: A Method for Explaining the Reject Option
finding the best threshold!

The main script will perform a *cross-fold validation* and will save the best model for each type of blackbox classifier (random forest, multi-layer perceptron, xgboost and lgbm)
``` python main.py --dataset german```

The remaining part of the code is in the notebooks: (it is also refactored into scripts, for better reproducibility and scalability)

To compute the distances of the counterfactuals, generated at te previous step
```
1_compute_distances.ipynb
```

To compute the distances of the counterfactuals generated at te previous step using the ILS method
```
2_compute_latent_distances.ipynb
```

To plot the distances and the correlation between the distances and the confidence of the classifier
```
3_correlation_and_distances_plot.ipynb
```
    
To wrap the classifiers and compute the metrics for the selective classifiers

```
4_selective_classifiers.ipynb
```

The scripts version of ```4_selective_classifiers.ipynb``` do the same thing as the notebook, but takes into consideration all the different counterfactuals generators methods and it's called ```selective_classifiers.py```

<!-- # IDEA
 Instead of only the prediction threshold XOR the distance threshold, we could use a combination of both. 
Than train a decision tree to learn the rules that optimize the performance of the classifier looking at the confidence of the classifier and the distance of the counterfactuals (we can still use the min, max, mean, also different measures of the distance)-->

### The main idea is to use the counterfactuals to learn the best threshold for the classifier

Inside the old code, there is a file called utils.py. Inside it there is a function called coumpute_rejection_policy and within, it calls the functions nonrejected_accuracy, classification_quality, rejection_quality.
<!--
TODO datasets:
TODO:
- [ ] Separate the computation of the counterfactuals in chunks
- [ ] test the other lore generators (check the naming of the saved files)

TODO refactorings:
- [ ] remove the debug* notebook files from main folder

DONE:
- [x] multiprocess computation of conterfactuals for different models
- [x] add the lore genetic 
- [x] add the ILS cf generator
- [x] add GROWING SPHERES
- [x] add ALL the remaining metrics (2)
- [x] save the counterfactuals as well as the rule and the tree for each sample
dataset:
- [x] Add the winsconsin dataset (https://archive.ics.uci.edu/ml/datasets/Wisconsin+Breast+Cancer)
- [x] add toydataset
- [x] add german credit
- [x] add income dataset
- [x] create class SelectiveClassifier for CFDistRejector, which will be a wrapper for the classifier and will have the method predict_proba and predict, calibrate and others.
- [x] add the adult dataset
refactorings:
- [x] add classification 
- [x] understand the imports
- [x] add the metric to the evaluation notebook (non rejected accuracy)
- [x] refactored the plot and the computation of the metrics
- [x] improve the eval notebook (add plots for the ALL metrics (1 for each blackbox))
- [x] add the metrics as tables! (still need to have python function to produce the latex table)

-->

# ''I know that I don’t know... and I explain why'' Robust abstention via counterfactual explanations
SC-CE: A Method for Explaining the Reject Option - Official Implementation

Please cite as 
```
@article{article,
author = {Bonsignori, V. and Punzi, C. and Pellungrini, Roberto and Giannotti, F.},
year = {2026},
month = {01},
pages = {1-1},
title = {‘‘I know that I don’t know... and I explain why’’ Robust abstention via counterfactual explanations},
journal = {IEEE Access},
doi = {10.1109/ACCESS.2026.3705102}
}
```

<!--
### remeber that for adding the submodules you have to 
git submodule add link/to/submodule
### and after
git submodule init \n

git submodule update

### to remove the submodule
git submodule deinit submodule_path
-->
