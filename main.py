"""
The main script to run the pipeline. The pipeline consists of three main phases:
1. Data Processing: Load and preprocess the dataset.
2. Model Training: Train a model on the dataset.
3. Counterfactual Generation: Generate counterfactuals for the model.
3.1 Save the counterfactuals to a file.
"""
from multiprocessing import Process
import multiprocessing as mp
import argparse
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.cf_generator_dice import DiceCFGenerator
from src.cf_generator_ils import IlsCFGenerator
from src.cf_generator_lore import LoreCFGenerator
from src.utils import load_config, check_file_exists


def run_pipeline(
    dataset_name: str,
    cf_method: str,
    config_path: str = "config.yaml",
    which_models = "all"
    ):
    """ The main pipeline to run the counterfactual generation process
        Args:
            dataset_name (str): The name of the dataset to use
            cf_method (str): The counterfactual generation method to use
            config_path (str): The path to the configuration file
    """
    print(cf_method)
    config = load_config(config_path)
    # Initialize components
    data_processor = DataProcessor(config=config)
    model_trainer = ModelTrainer(config=config)
    num_cf = config.get("counterfactuals", {}).get("num_counterfactuals", 8)
    cf_generator = None
    if cf_method == "dice":
        cf_generator = DiceCFGenerator(config=config,
                                       dataset_name=dataset_name)
    elif cf_method == "ils":
        cf_generator = IlsCFGenerator(config=config,
                                       dataset_name=dataset_name,
                                       )

    elif cf_method == "lore":
        cf_generator = LoreCFGenerator(config=config,
                                       dataset_name=dataset_name)
    ########################################
    ######### Data Processing Phase ########
    ########################################
    splits_path = f"data/processed/{dataset_name}/y_train.npz"
    if not check_file_exists(splits_path):
        print(splits_path,"not found")
        print("Processing dataset")
        data = data_processor.load_dataset(dataset_name)
        processed_data = data_processor.preprocess_data(data,
                                                        dataset_name)
        splits = data_processor.split_data(features=processed_data[0],
                                           target=processed_data[1])
        data_processor.save_splits(splits, dataset_name)
    else:
        print("Loading existing dataset")
        splits = data_processor.load_splits(dataset_name)
    ########################################
    ######### Model Training Phase #########
    # Train or load all models
    models = model_trainer.train_model(
            splits['X_train'],
            splits['y_train'],
            splits['X_test'],
            splits['y_test'],
            dataset_name=dataset_name,
            n_folds=5,
            which_models = which_models
        )

    # Models can now be used for counterfactual generation
    for model_name, model in models.items():
        print(f"Model {model_name} is ready for use at {model}")


    ########################################
    ####### Counterfactual Generation ######
    ########################################
    #cf_path = f"data/processed/{dataset_name}/counterfactuals_{cf_method}.pkl"
    #if not check_file_exists(cf_path):

    # if the counterfactuals do not exist, the generator has not been
    # run yet, nor has it been saved, we need to set it up
    # now we join the X_train and y_train to be passed as a reference Dataframe
    # we stack as last column the target variable
    reference_set = splits['X_train'].copy()
    feat_names = splits['X_train'].columns.to_list()
    target_name = data_processor.target_name
    # we add the target variable to the reference set
    reference_set[target_name] = splits['y_train']
    cf_generator.setup(
        reference_data=reference_set,
        #feature_names=feat_names,
        continuous_features = feat_names,
        categorical_features=[],#data_processor.categorical_features,
        target_name=target_name,
    )
    print("Generating counterfactuals")
    counterfactuals = cf_generator.generate_counterfactuals(
        models=models,
        X_calibration=splits['X_calibration'][:].copy(),
        y_calibration=splits['y_calibration'][:].copy(),
        num_cf=num_cf,
        dt_name=dataset_name,
        set_name="calibration"
    )
    counterfactuals = cf_generator.generate_counterfactuals(
        models=models,
        X_calibration=splits['X_test'][:].copy(),
        y_calibration=splits['y_test'][:].copy(),
        num_cf=num_cf,
        dt_name=dataset_name,
        set_name="test"
    )
    return counterfactuals

if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="Run the counterfactual generation pipeline")
    parser.add_argument("--dataset", type=str,
                        help="The name of the dataset to use",default="adult48k")
    parser.add_argument("--cf_method", type=str,
                        help="The counterfactual generation method to use",default="lore")
    parser.add_argument("--config_path", type=str,
                        help="The path to the configuration file",default="config.yaml")
    args = parser.parse_args()
    import time
    start = time.time()


    # Create a list to keep track of processes
    processes = []
    model_types = ["mlp", "random_forest", "xgboost", "lgbm"]
    try:
        # Start processes
        for model_type in model_types:
            p = Process(target=run_pipeline,
                       args=(args.dataset, args.cf_method, args.config_path, [model_type]))
            p.start()
            processes.append(p)
            if len(processes)>=2:
                # Wait for the first process to finish
                processes[0].join()
                processes.pop(0)
                print("joined a process")
        # Wait for any remaining processes to finish
        for p in processes:
            p.join()

    except Exception as e:
        print(f"Error in parallel processing: {e}")
        # Terminate any remaining processes
        for p in processes:
            if p.is_alive():
                p.terminate()

    end = time.time()
    print(f"FINISHED the computation in {end-start:.2f} seconds"+
           f"for the dataset {args.dataset}"+
           f"with the method {args.cf_method}")
    # run_pipeline(
    #     dataset_name=args.dataset,
    #     cf_method=args.cf_method,
    #     config_path=args.config_path,
    #     which_models=["mlp","random_forest","xgboost","lgbm"]
    # )
