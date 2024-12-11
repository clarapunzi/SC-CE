"""
The main script to run the pipeline. The pipeline consists of three main phases:
1. Data Processing: Load and preprocess the dataset.
2. Model Training: Train a model on the dataset.
3. Counterfactual Generation: Generate counterfactuals for the model.
4. Counterfactual Evaluation: Evaluate the counterfactuals generated. (Not implemented yet)
5. 
"""
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.counterfactual_generator import CFGDice
from src.utils import load_config, check_file_exists


def run_pipeline(
    dataset_name: str,
    cf_method: str,
    config_path: str = "config.yaml"
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
    cf_generator = CFGDice(config=config)
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

    # print the balance of the splits
    print("Train balance: \n", splits['y_train'].value_counts())
    print("Test balance: \n", splits['y_test'].value_counts())
    print("Calibration balance: \n", splits['y_calibration'].value_counts())

    ########################################
    ######### Model Training Phase #########
    # Train or load all models
    models = model_trainer.train_model(
            splits['X_train'],
            splits['y_train'],
            splits['X_test'],
            splits['y_test'],
            dataset_name=dataset_name,
            n_folds=5
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
    counterfactuals = cf_generator.generate_counterfactuals(
        models=models,
        X_calibration=splits['X_calibration'][:],
        y_calibration=splits['y_calibration'][:],
        cf_method=cf_method,
        num_cf=8,
        dt_name=dataset_name,
    )
    return counterfactuals

if __name__ == "__main__":
    
    run_pipeline(
        dataset_name="german_credit",
        cf_method="dice"
    )
