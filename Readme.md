# Learning to Reject with Counterfactual Data Augmentation
finding the best threshold!

TODO:
- [ ] Separate the computation of the counterfactuals in chunks
- [ ] save the counterfactuals as well as the rule and the tree for each sample
- [ ] test the other lore generators (check the naming of the saved files)
- [ ] add GROWING SPHERES
TODO datasets:
- [ ] Add the winsconsin dataset (https://archive.ics.uci.edu/ml/datasets/Wisconsin+Breast+Cancer)
- [ ] add income dataset
- [ ] add the adult dataset

TODO refactorings:
- [ ] create class SelectiveClassifier for L2Lore, which will be a wrapper for the classifier and will have the method predict_proba and predict, calibrate and others.
- [ ] add ALL the remaining metrics (2)
- [ ] improve the eval notebook (add plots for the metrics (1 for each blackbox))
- [ ] add the metrics as tables!

DONE:
- [x] add the lore genetic - partitioning
dataset:
refactorings:
- [x] add classification 
- [x] understand the imports
- [x] add the metric to the evaluation notebook (non rejected accuracy)


# IDEA
Instead of only the prediction threshold XOR the distance threshold, we could use a combination of both. 
Than train a decision tree to learn the rules that optimize the performance of the classifier looking at the confidence of the classifier and the distance of the counterfactuals (we can still use the min, max, mean, also different measures of the distance)

### The main idea is to use the counterfactuals to learn the best threshold for the classifier

Inside the old code, there is a file called utils.py. Inside it there is a function called coumpute_rejection_policy and within, it calls the functions nonrejected_accuracy, classification_quality, rejection_quality and rejection_classification_report. 


### remeber that for adding the submodules you have to 
git submodule add link/to/submodule
### and after
git submodule init \n

git submodule update

### to remove the submodule
git submodule deinit submodule_path