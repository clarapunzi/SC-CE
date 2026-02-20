""" Since this project is huge I want a script to check for the results.
I have many dataset,
many models,
many counterfactual generation methods,
and two sets of distances to be computed.

I want to be able to run this script and get a table like this:
with the letter P meaning that the method is computed, and the letter A meaning that the method is not computed.

| Dataset | RF | MLP | XGBoost | LGBM | LipMLP |
| Adult   | P  | P   | P       | P    | P      |
| Toy     | P  | P   | P       | P    | P      |
... and so on

Then, if the models are there, I need to check if the counterfactual generation methods are computed, and I want a table like this:
in this case A means that the counterfactuals are not computed, P means that they are computed (but not the distances)
and D means that the counterfactuals are computed and the distances are computed (it's complete)

================ ADULT =================
| Model   | LORE | GS | DiCE | ILS |
|---------|------|----|------|-----|
| MLP     | P    | P  | P    | A   |
| RF      | P    | P  | P    | A   |
| XGBoost | D    | P  | P    | P   |
| LGBM    | D    | P  | P    | D   |
| LipMLP  | P    | P  | P    | P   |
================ TOY =================
| Model   | LORE | GS | DiCE | ILS |
|---------|------|----|------|-----|
| MLP     | P    | P  | P    | P   |
and so on 

In the folder models, there are the models. models/datasetfolder/modelname.pkl

in the folder counterfactuals, there are the counterfactuals. counterfactuals/modelname/datasetfolder/cf_results_methodname*_test_DATE.pkl (and also _calibration_)
The distances computed are a single file for each model and dataset. Each one is in their respective folder.
growingspheres, ils (ils have also the ils_latent version, so they should be twice the number), lore, dice
"""

import os
import pickle
from typing import Dict, Any

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists at the given path."""
    return os.path.exists(file_path)
def check_model_exists(model_name: str, dataset_name: str) -> bool:
    """Check if a trained model exists for the given model and dataset."""
    return check_file_exists(os.path.join("models", dataset_name, f"{model_name}.pkl"))
def check_cf_results_exists(model_name: str, dataset_name: str, cf_method: str, set_file: str) -> bool:
    """Check if counterfactual results exist for the given model, dataset, and CF method."""
    cf_results_path = os.path.join(
        "counterfactuals",
        model_name,
        dataset_name,
    )
    # as the distances are computed many times sometimes they are not saved with tje date as ending,
    # so we should check the content of the folder firs than check the amount of files with that cf_method and if the set is that one
    possibile_files = os.listdir(cf_results_path) if os.path.exists(cf_results_path) else []
    possibile_files = [f for f in possibile_files if cf_method in f and ("_"+set_file+"_" in f if set_file in ["test"] else ("_test_" not in f))]
    return len(possibile_files) > 0
    return check_file_exists(cf_results_path)
def check_distances_exists(model_name: str, dataset_name: str, cf_method: str, set_file: str) -> bool:
    """Check if distances are computed for the given model, dataset, and CF method."""
    # set file can be "test" or "calibration"
    distances_path = os.path.join(
        "distances",
        model_name,
        dataset_name,
    )
    # as the distances are computed many times sometimes they are not saved with tje date as ending,
    # so we should check the content of the folder firs than check the amount of files with that cf_method and if the set is that one
    possibile_files = os.listdir(distances_path) if os.path.exists(distances_path) else []
    possibile_files = [f for f in possibile_files if cf_method in f and "_"+set_file+"_" in f]
    return len(possibile_files) > 0

if __name__ == "__main__":
    # Example usage
    datasets = ["adult48k", "toy_dataset", "german_credit", "breast_cancer"]
    models = ["random_forest", "mlp", "xgboost", "lgbm", "lip_mlp"]
    cf_methods = ["lore", "gs", "dice", "ils", "ils_latent"]

    for dataset in datasets:
        # check if models are computed
        print(f"Checking models for dataset: {dataset}")
        print("| Dataset        |   RF   |  MLP   |XGBoost |  LGBM  | LipMLP |")
        print("|----------------|--------|--------|--------|--------|--------|")
        for j,model in enumerate(models):
            model_exists = "" if check_model_exists(model, dataset) else "A"
            if j == 0:
                print(f"| {dataset:<14} | {model_exists:<6}", end =" |")
            else:
                print(f" {model_exists:<6}", end =" |")
        print("\n")
    input("Press Enter to check counterfactual results...")
    for dataset in datasets:
        print(f"================ {dataset.upper()} =================")
        print("| Model          | LORE   |  GS  |  DiCE  | ILS  | ILS_l |")
        print("|----------------|--------|------|--------|------|-------|")
        for model in models:
            for cf_method in cf_methods:
                cf_exists = "P" if check_cf_results_exists(model, dataset, cf_method, "test") else "A"
                cf_exists_cal = "P" if check_cf_results_exists(model, dataset, cf_method, "calibration") else "A"
                if cf_exists_cal == "P" and cf_exists == "P":
                    cf_exists = ""
                else:
                    cf_exists = cf_exists_cal+"/"+cf_exists
                if cf_method == "lore":
                    lore_status = cf_exists
                elif cf_method == "gs":
                    gs_status = cf_exists
                elif cf_method == "dice":
                    dice_status = cf_exists
                elif cf_method == "ils":
                    ils_status = cf_exists
                elif cf_method == "ils_latent":
                    ils_latent_status = cf_exists
            
            print(f"| {model.upper():<14} | {lore_status:<6} | {gs_status:<4} | {dice_status:<6} | {ils_status:<4} | {ils_latent_status:<7} |")