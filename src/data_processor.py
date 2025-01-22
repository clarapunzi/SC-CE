"""
Module for loading, preprocessing, and splitting datasets.
"""
import os
from typing import Dict, Tuple, Optional, Union
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import category_encoders as ce
from src.toy_dataset import generate_toy
class DataProcessor:
    """Handles loading, preprocessing, and splitting of datasets."""

    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {}  # Store scalers for each numerical column
        self.encoders = {} # Store encoders for each categorical column
        self.target_name = "target"

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a specific dataset."""
        print(f"Loading {dataset_name} dataset")

        if dataset_name == "german_credit":
            return self._load_german_credit()
        elif dataset_name == "folktables":
            return self._load_folktables()
        if dataset_name == "toy_dataset":
            return self._load_toy_dataset()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _load_german_credit(self) -> pd.DataFrame:
        """Load German Credit dataset."""
        path = self.config['data']['german_credit']['path']
        try:
            df = pd.read_csv(path)
            print(f"Loaded German Credit dataset with shape {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading German Credit dataset: {str(e)}")
            raise

    def _load_folktables(self) -> pd.DataFrame:
        """Load Folktables dataset."""
        path = self.config['data']['folktables']['path']
        try:
            # Implement Folktables loading logic
            # This might involve more complex logic depending on your exact needs
            df = pd.read_csv(path)
            print(f"Loaded Folktables dataset with shape {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading Folktables dataset: {str(e)}")
            raise

    def preprocess_data(self,
                       data: pd.DataFrame,
                       dataset_name: str,
                       fit: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Preprocess the dataset."""
        print("Starting preprocessing")

        if dataset_name == "german_credit":
            return self._preprocess_german_credit(data, fit)
        elif dataset_name == "folktables":
            return self._preprocess_folktables(data, fit)
        elif dataset_name == "toy_dataset":
            return self._preprocess_toy(data, fit)

    def _preprocess_german_credit(self,
                                data: pd.DataFrame,
                                fit: bool) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess German Credit dataset."""
        # Copy data to avoid modifying original
        df = data.copy()
        # Extract target
        df["target"] = df["default"]
        del df["default"]
        target = df['target']
        features = df.drop('target', axis=1)
        self.target_name = "Default"
        # Identify numerical and categorical columns
        num_cols = features.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = features.select_dtypes(include=['object', 'category']).columns

        # Handle numerical features
        if fit:
            for col in num_cols:
                self.scalers[col] = StandardScaler()
                features[col] = self.scalers[col].fit_transform(features[[col]])
        else:
            for col in num_cols:
                if col in self.scalers:
                    features[col] = self.scalers[col].transform(features[[col]])

        # Handle categorical features
        if fit:
            for col in cat_cols:
                self.encoders[col] = ce.TargetEncoder()
                features[col] = self.encoders[col].fit_transform(features[col], target)
        else:
            for col in cat_cols:
                if col in self.encoders:
                    features[col] = self.encoders[col].transform(features[col])

        return features, target

    def _preprocess_folktables(self,
                             data: pd.DataFrame,
                             fit: bool) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess Folktables dataset."""
        # Implement similar preprocessing for Folktables
        return

    def _load_toy_dataset(self) -> pd.DataFrame:
        X,y = generate_toy()
        self.target_name = "Toy_Class"
        self.feature_names = ['Feature1', 'Feature2']
        # create a dataframe for the toy dataset
        features = pd.DataFrame(X, columns=self.feature_names)
        features[self.target_name] = y
        return features
    def _preprocess_toy(self,
                          data: pd.DataFrame,
                            fit: bool) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess Toy dataset."""
        # Copy data to avoid modifying original
        df = data.copy()
        # Extract target
        target = df[self.target_name]
        features = df.drop(self.target_name, axis=1)
        # they are all numerical
        num_cols = features.columns
        # Handle numerical features
        if fit:
            for col in num_cols:
                self.scalers[col] = StandardScaler()
                features[col] = self.scalers[col].fit_transform(features[[col]])
        else:
            for col in num_cols:
                if col in self.scalers:
                    features[col] = self.scalers[col].transform(features[[col]])
        return features, target


    def split_data(self,
                  features: pd.DataFrame,
                  target: pd.Series,
                  train_size: float = 0.55,
                  test_size: float = 0.25,
                  random_state: int = 42) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """Split data into train, test, and calibration sets."""
        print("Splitting data into train, test, and calibration sets")

        # First split: separate train from the rest
        X_train, X_temp, y_train, y_temp = train_test_split(
            features,
            target,
            train_size=train_size,
            random_state=random_state,
            stratify=target
        )

        # Second split: divide the rest between test and calibration
        # Adjust test_size to get desired proportion from remaining data
        remaining_test_size = test_size / (1 - train_size)
        X_test, X_calibration, y_test, y_calibration = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - remaining_test_size),
            random_state=random_state,
            stratify=y_temp
        )

        splits = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_calibration': X_calibration,
            'y_calibration': y_calibration
        }

        # Log split sizes
        for name, data in splits.items():
            print(f"{name} shape: {data.shape}",
                    f"Percentage of total: {data.shape[0] / features.shape[0]:.2f}")

        return splits

    def save_splits(self,
                   splits: Dict[str, Union[pd.DataFrame, pd.Series]],
                   dataset_name: str,
                   output_dir: str = 'data/processed'):
        """Save the splits to disk."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print("created output directory: ", output_dir)
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        print("created dataset directory: ", dataset_dir)

        # Save each split
        for name, data in splits.items():
            path = os.path.join(dataset_dir, f"{name}")
            np.savez(path,patterns=data.values) # this will save the data in a .npz file
            print(f"Saved {name} to {path}")
        # save the names of the columns
        path = os.path.join(dataset_dir, 'feature_names.csv')
        splits['X_train'].columns.to_series().to_csv(path, index=False)
        print(f"Saved feature names to {path}")

    def load_splits(self,
                   dataset_name: str,
                   input_dir: str = 'data/processed') -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """Load the splits from disk."""
        dataset_dir = os.path.join(input_dir, dataset_name)
        splits = {}

        path = os.path.join(dataset_dir, 'feature_names.csv')
        print(f"Loaded feature names from {path}")
        feature_names = pd.read_csv(path)
        # transform the series into a list
        feature_names = feature_names['0'].tolist()

        for split_name in ['X_train', 'y_train',
                           'X_test', 'y_test',
                           'X_calibration', 'y_calibration']:
            path = os.path.join(dataset_dir, f"{split_name}.npz")
            loaded = np.load(path)
            print(f"Loaded {split_name} from {path}")
            splits[split_name] = pd.DataFrame(loaded['patterns'],
                        columns=feature_names if split_name.startswith('X') else None)
        return splits
