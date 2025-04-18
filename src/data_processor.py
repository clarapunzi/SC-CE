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

# Get dataset name for title
fancy_dataset_names = {
            "german_credit": "German Credit",
            "adult48k": "Adult",
            "toy_dataset": "Two Moons"
        }

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
        if dataset_name == "adult48k":
            return self._load_adult48k()
        if dataset_name == "breast_cancer":
            return self._load_breast_cancer()
        if dataset_name == "eye":
            return self._load_eye()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    def _load_eye(self) -> pd.DataFrame:
        """Load eye dataset."""
        path = self.config['data']['eye']['path']
        try:
            df = pd.read_csv(path)
            print(f"Loaded eye dataset with shape {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading eye dataset: {str(e)}")
            raise
    def _load_breast_cancer(self) -> pd.DataFrame:
        """Load breast cancer dataset."""
        path = self.config['data']['breast_cancer']['path']
        try:
            df = pd.read_csv(path)
            print(f"Loaded breast cancer dataset with shape {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading breast cancer dataset: {str(e)}")
            raise
    

    def _load_adult48k(self) -> pd.DataFrame:
        """Load adult48k dataset."""
        path = self.config['data']['adult48k']['path']
        try:
            df = pd.read_csv(path)
            print(f"Loaded adult48k dataset with shape {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading adult48k dataset: {str(e)}")
            raise

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
        elif dataset_name == "adult48k":
            return self._preprocess_adult48k(data)
        elif dataset_name == "breast_cancer":
            return self._preprocess_breast_cancer(data, fit)
        elif dataset_name == "eye":
            return self._preprocess_eye(data, fit)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return data
    def _preprocess_eye(self,
                        data: pd.DataFrame,
                        fit: bool) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess Eye dataset."""
        
        # Extract target
        data["Relevant"] = data["label"]
        # Copy data to avoid modifying original
        features = data.copy()
        # drop #line and #assg
        features = features.drop(columns=['#line', '#assg'], axis=1)
        # Extract target
        target = features["Relevant"]
        del features["Relevant"]
        if fit:
            # scale the data in the range [0,1]
            features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))
        else:
            features = features
        return features, target
        
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
    def _preprocess_breast_cancer(self,
                                  data: pd.DataFrame,
                                  fit: bool) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess Breast Cancer dataset."""
        # drop the first column (its the index)
        # Extract target
        data["Diagnosis"] = data["diagnosis"].map({'M': 1, 'B': 0})
        del data["diagnosis"]
        # remove the Unnamed: 32 column
        data = data.drop("Unnamed: 32", axis=1)
        # remove the id column
        data = data.drop("id", axis=1)        
        y = data["Diagnosis"]
        del data["Diagnosis"]
        self.target_name = "Diagnosis"
        self.feature_names = data.columns.tolist()
        # scale the data in the range [0,1]
        if fit:
            data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        else:
            data = data
        return data, y
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
        if X_calibration.shape[0] > 1000:
            # copy the old calibration set into a "X_calibration_full"
            X_calibration_full = X_calibration
            y_calibration_full = y_calibration
            # create a smaller calibration set, cappped at 200

            X_calibration, _, y_calibration, _ = train_test_split(
                X_calibration_full,
                y_calibration_full,
                train_size=300,
                random_state=random_state,
                stratify=y_calibration_full
            )
            # same for the test set
            X_test_full = X_test
            y_test_full = y_test
            X_test, _, y_test, _ = train_test_split(
                X_test_full,
                y_test_full,
                train_size=500,
                random_state=random_state,
                stratify=y_test_full
            )
            splits = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'X_calibration': X_calibration,
                'y_calibration': y_calibration,
                'X_calibration_full': X_calibration_full,
                'y_calibration_full': y_calibration_full,
                'X_test_full': X_test_full,
                'y_test_full': y_test_full
            }

        else:

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

    def _preprocess_adult48k(self,data: pd.DataFrame,fit:bool=True
                            ) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess German Credit dataset."""
        # Copy data to avoid modifying original
        # Extract target
        dt = data.copy()
        dt["target"] = dt["class"]
        del dt["class"]
        #print()
        print(dt.columns)
        #print(dt.columns)
        names = dt.columns
        # Map Ages, Education, Workclass, and Weekly-Hours to smaller category set.
        
        #special_category = 'occupation'
        #print(special_category)
        #print(dt[special_category].values)
        dt.loc[dt['occupation']=='Armed-Forces','occupation'] = 'Protective-serv'
        dt.loc[dt['workclass'].isin(['State-gov', 'Federal-gov', 'Local-gov']),
            'employment-type'] = 'Government'
        dt.loc[dt['workclass'].isin(['Self-emp-not-inc', 'Self-emp-inc']),
            'employment-type'] = 'Self-Employed'
        dt.loc[dt['workclass'].isin(['Private']),
            'employment-type'] = 'Privately-Employed'
        #special_category = 'employment-type'
        #print(special_category)
        #print(dt[special_category].values)
        #print(names)
        dt['education-num'] = dt['education-num'].values.astype(int)
        #special_category = 'education-num'
        #print(special_category)
        #print(dt[special_category].values)
        #print(type(dt[special_category].values[0]))
        #print(dt['education-num'].isin([8]))
        dt.loc[dt['education-num'].isin([2,3,4,5,6,7,8]),'education'] = 'Less than High School'
        dt.loc[dt['education-num'].isin([ 9,10]), 'education'] = 'High School'
        dt.loc[dt['education-num'].isin([11,12]), 'education'] = 'Associates'
        dt.loc[dt['education-num'].isin([13]),    'education'] = 'Bachelors'
        dt.loc[dt['education-num'].isin([14]),    'education'] = 'Masters'
        dt.loc[dt['education-num'].isin([15,16]), 'education'] = 'Pro'
        #education is now categorical
        numeric_features_idx = [0,10,11,12]
        x_float = np.array(dt.iloc[:,numeric_features_idx]).astype(np.float32)
        y = np.array(dt.iloc[:,14])#np.array(dt.iloc[1:,1])
        #Normalization of data
        x_float = (x_float - np.mean(x_float,axis=0)) / np.std(x_float,axis=0)
        y[y==' >50K'] = 1
        y[y==' <=50K'] = 0
        y = y.astype(np.intc)
        categorical_features_idx = [1,3,5,6,7,8,9,13]
        categorical_features = np.array(dt.iloc[:,categorical_features_idx])

        categorical_features = np.array([])
        cat_values = {}
        for cat_feat_idx in categorical_features_idx:
            x_c = dt.iloc[:,cat_feat_idx]
            x_c_values = set(x_c)
            if fit:
                #print(names[catFeatIdx],len(x_c_values))
                str_data = [str(e) for e in x_c]
                target_encoder = ce.TargetEncoder()
                target_encoder.fit(str_data,y)
                xtargenc = target_encoder.transform([str(e) for e in x_c]).to_numpy()
                categorical_features = np.hstack([categorical_features,xtargenc]) if categorical_features.size else xtargenc
                feat_name = "P(t="+str(1)+"|"+names[cat_feat_idx]+")"
                cat_values[feat_name] = np.unique(xtargenc)
                #print(feat_name,cat_values[feat_name],)
                cat_values[feat_name] = {c_val:target_encoder.transform([c_val])[0].to_numpy().astype(float)[0] for c_val in sorted(dt.iloc[:,cat_feat_idx].unique())}
                #print(feat_name,cat_values[feat_name],)
            else:
                one_hot_feats_names = []
                onehot = np.zeros((x_c.shape[0],len(set(x_c))))
                cat_feat_2_idx = {k:i for i,k in enumerate(x_c_values)}

                for i,cat_val in enumerate(x_c):
                    onehot[i,cat_feat_2_idx[cat_val]]=1
                    one_hot_feats_names.append("is "+cat_val)
                categorical_features = np.hstack([categorical_features ,onehot]) if categorical_features.size else onehot
        if fit:
            names = np.concatenate([names[numeric_features_idx],names[categorical_features_idx]])
        else:
            names = np.concatenate([names[numeric_features_idx],one_hot_feats_names])
        features = np.hstack([x_float,categorical_features])
        self.target_name = "Income"
        
        return pd.DataFrame(features, columns=names), pd.Series(y)