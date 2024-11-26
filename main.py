# main.py
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
# from src.counterfactual_generator import CounterfactualGenerator
from src.utils import load_config, check_file_exists

def run_pipeline(
    dataset_name: str,
    model_type: str,
    cf_method: str,
    config_path: str = "config.yaml"
):
    print(model_type , cf_method)
    # Initialize components
    data_processor = DataProcessor(config=load_config(config_path))
    #model_trainer = ModelTrainer()
    #cf_generator = CounterfactualGenerator()

    # Data Processing Phase
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

    # Model Training Phase
    model_path = f"models/trained_models/{model_type}/{dataset_name}"
    if not check_file_exists(model_path):
        model = model_trainer.train_model(
            splits['X_train'],
            splits['y_train'],
            model_type
        )
        model_trainer.save_model(model, model_path)
    else:
        model = model_trainer.load_model(model_path)

    '''
    # Counterfactual Generation Phase
    cf_path = f"counterfactuals/cf_results/{dataset_name}/{model_type}/{cf_method}"
    if not check_file_exists(cf_path):
        counterfactuals = cf_generator.generate_counterfactuals(
            model, 
            splits['X_calibration'], 
            cf_method
        )
        cf_generator.save_counterfactuals(counterfactuals, cf_path)
    else:
        counterfactuals = cf_generator.load_counterfactuals(cf_path)

    return counterfactuals
    '''
if __name__ == "__main__":
    run_pipeline(
        dataset_name="german_credit",
        model_type="random_forest",
        cf_method="dice"
    )
