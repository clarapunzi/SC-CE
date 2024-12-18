# Learning to Reject with Counterfactual Data Augmentation
finding the best threshold!

TODO:
- [ ] test the other lore generators (check the naming of the saved files)
- [ ] add the metrics
- [ ] Separate the computation of the counterfactuals in chunks
- [ ] save the counterfactuals as well as the rule and the tree for each sample
- [ ] add GROWING SPHERES

TODO dataset:
- [ ] Add tge winsconsin dataset (https://archive.ics.uci.edu/ml/datasets/Wisconsin+Breast+Cancer)
- [ ] add income dataset

TODO refactorings:
- [ ] understand the imports
- [ ] improve the eval notebook 
- [ ] create class SelectiveClassifier, which will be a wrapper for the classifier and will have the method predict_proba and predict. 



DONE:
- [x] add the lore genetic - partitioning



### The main idea is to use the counterfactuals to learn the best threshold for the classifier

Inside the old code, there is a file called utils.py. Inside it there is a function called coumpute_rejection_policy and within, it calls the functions nonrejected_accuracy, classification_quality, rejection_quality and rejection_classification_report. 


### remeber that for adding the submodules you have to 
git submodule add link/to/submodule
### and after
git submodule init \n

git submodule update

### to remove the submodule
git submodule deinit submodule_path