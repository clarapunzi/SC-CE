"""
Counterfactual Quality Metrics Computer

Extensible framework for computing quality metrics on counterfactual explanations.
Supports custom metric registration for easy extension.

Binary classification assumed throughout.
"""

import numpy as np
from typing import Dict, List, Callable, Any
import warnings
from sklearn.neighbors import KNeighborsClassifier


class CFQualityMetricsComputer:
    """
    Compute quality metrics on counterfactual explanations.

    Provides built-in metrics and extensible registry for custom metrics.
    All metrics return scalar values (aggregated per model/method/dataset).

    Example:
        >>> computer = CFQualityMetricsComputer()
        >>> metrics = computer.compute_all_metrics(
        ...     cfs_dict={'counterfactuals': cf_df},
        ...     original_instance=x_orig,
        ...     model=model,
        ...     y_true=0  # true class label
        ... )
        >>> print(metrics['fidelity'])  # 0.85

        # Add custom metric
        >>> def metric_custom(cfs, original, model, y_test):
        ...     return len(cfs) / 2  # example
        >>> computer.register_metric('custom', metric_custom)
        >>> metrics = computer.compute_all_metrics(...)  # now includes custom metric
    """

    def __init__(self):
        """Initialize with built-in metrics registered."""
        self.metrics = {}
        self.X_test = None  # For metrics that need the full test set
        self.y_test_pred = None  # For metrics that need model predictions
        self._register_builtin_metrics()

    def _register_builtin_metrics(self):
        """Register all built-in metrics."""
        self.register_metric('cf_count', self._metric_cf_count)
        self.register_metric('cf_success_rate', self._metric_cf_success_rate)
        self.register_metric('fidelity', self._metric_fidelity)
        self.register_metric('discriminative_power', self._metric_discriminative_power)
        self.register_metric('avg_l2_distance', self._metric_avg_l2_distance)
        self.register_metric('avg_l0_distance', self._metric_avg_l0_distance)
        self.register_metric('min_l0_distance', self._metric_min_l0_distance)
        self.register_metric('diversity_std_l2', self._metric_diversity_std_l2)

    def register_metric(self, name: str, func: Callable) -> None:
        """
        Register a custom metric.

        Args:
            name: Metric identifier (e.g., 'my_custom_metric')
            func: Function with signature (cfs_list, original_instance, model, y_true) -> float
        """
        if name in self.metrics:
            warnings.warn(f"Overwriting existing metric '{name}'")
        self.metrics[name] = func

    def compute_all_metrics(self, cfs_dict: Dict, original_instance: np.ndarray,
                           model: Any, y_true: int) -> Dict[str, float]:
        """
        Compute all registered metrics at once.

        Args:
            cfs_dict: Result dict with 'counterfactuals' key (DataFrame with 'target' column)
            original_instance: Single instance as numpy array (already in right shape/space)
            model: Trained classifier with predict/predict_proba methods
            y_true: True class label

        Returns:
            Dict mapping metric_name -> value
        """
        results = {}

        # Extract CFs as numpy array
        if 'counterfactuals' not in cfs_dict or cfs_dict['counterfactuals'] is None:
            for metric_name in self.metrics:
                results[metric_name] = 0.0
            return results

        cfs_df = cfs_dict['counterfactuals']
        if len(cfs_df) == 0:
            for metric_name in self.metrics:
                results[metric_name] = 0.0
            return results

        # Extract features: remove all columns except feature columns
        # If it's a numpy array, use directly; if DataFrame, remove metadata
        if hasattr(cfs_df, 'values'):
            # It's a DataFrame
            metadata_cols = {'target', 'true_class', 'instance_idx', 'original_class', 'predicted_class', 'prediction', 'predicted'}
            feature_cols = [c for c in cfs_df.columns if c not in metadata_cols]
            cfs_array = cfs_df[feature_cols].values
        else:
            # Assume it's already a numpy array of features
            cfs_array = cfs_df

        # Ensure original_instance is 2D
        if original_instance.ndim == 1:
            original_instance = original_instance.reshape(1, -1)

        # Compute cf_count first (doesn't require matching shapes)
        results['cf_count'] = len(cfs_array)

        # CHECK FOR SHAPE MISMATCH - only matters for metrics that use the model
        if cfs_array.shape[1] != original_instance.shape[1]:
            print(f"⚠️  SHAPE MISMATCH: CFs have {cfs_array.shape[1]} features, but original has {original_instance.shape[1]}")
            print(f"    This instance will be skipped for model-based metrics")
            # Return NaN for model-based metrics only
            for metric_name in self.metrics:
                if metric_name != 'cf_count':  # cf_count is already computed
                    results[metric_name] = np.nan
            return results

        # Compute each metric - NO ERROR HANDLING
        for metric_name, metric_func in self.metrics.items():
            if metric_name != 'cf_count':  # Already computed above
                results[metric_name] = metric_func(cfs_array, original_instance, model, y_true)

        return results

    # ========== BUILT-IN METRICS ==========

    def _metric_cf_count(self, cfs: np.ndarray, original_instance: np.ndarray,
                        model: Any, y_true: int) -> float:
        """
        Number of counterfactuals found.

        Higher is better (more solutions to choose from).
        """
        return float(len(cfs))

    def _metric_cf_success_rate(self, cfs: np.ndarray, original_instance: np.ndarray,
                               model: Any, y_true: int) -> float:
        """
        Success rate: % of instances where at least one CF exists.

        Binary: returns 1.0 if any CFs found, 0.0 otherwise.
        Higher is better.
        """
        return 1.0 if len(cfs) > 0 else 0.0

    def _metric_fidelity(self, cfs: np.ndarray, original_instance: np.ndarray,
                        model: Any, y_true: int) -> float:
        """
        Fidelity: % of CFs with predicted class ≠ original class.

        Measures whether CFs actually flip the predicted class.
        Range: [0, 1]. Higher is better.
        """
        if len(cfs) == 0:
            return 0.0

        original_pred = model.predict(original_instance)[0]
        cf_preds = model.predict(cfs)

        # Count CFs that flip the class
        flipped = np.sum(cf_preds != original_pred)
        return float(flipped) / len(cfs)

    def _metric_avg_l2_distance(self, cfs: np.ndarray, original_instance: np.ndarray,
                               model: Any, y_true: int) -> float:
        """
        Average L2 (Euclidean) distance of CFs from original.

        Measures average proximity/cost of solutions.
        Range: [0, ∞). Lower is better.
        """
        if len(cfs) == 0:
            return np.inf

        # Compute L2 distances
        distances = np.linalg.norm(cfs - original_instance, axis=1)
        return float(np.mean(distances))

    def _metric_avg_l0_distance(self, cfs: np.ndarray, original_instance: np.ndarray,
                               model: Any, y_true: int) -> float:
        """
        Average L0 distance: mean number of features changed.

        Sparsity measure (raw feature count).
        Range: [0, n_features]. Lower is better.
        """
        if len(cfs) == 0:
            return float(original_instance.shape[1])  # All features different

        # Count non-zero differences per CF
        differences = np.abs(cfs - original_instance) > 1e-10
        l0_distances = np.sum(differences, axis=1)
        return float(np.mean(l0_distances))

    def _metric_min_l0_distance(self, cfs: np.ndarray, original_instance: np.ndarray,
                               model: Any, y_true: int) -> float:
        """
        Minimum L0 distance: best-case number of features to change.

        Best sparsity achieved by any CF.
        Range: [0, n_features]. Lower is better.
        """
        if len(cfs) == 0:
            return float(original_instance.shape[1])  # All features different

        # Find CF with fewest feature changes
        differences = np.abs(cfs - original_instance) > 1e-10
        l0_distances = np.sum(differences, axis=1)
        return float(np.min(l0_distances))

    def _metric_diversity_std_l2(self, cfs: np.ndarray, original_instance: np.ndarray,
                                model: Any, y_true: int) -> float:
        """
        Diversity: standard deviation of L2 distances.

        Measures spread/variety of solutions. Std of L2 distances.
        Range: [0, ∞). Higher is better (more variety).
        """
        if len(cfs) <= 1:
            return 0.0

        # Compute L2 distances
        distances = np.linalg.norm(cfs - original_instance, axis=1)
        return float(np.std(distances))

    def _metric_discriminative_power(self, cfs: np.ndarray, original_instance: np.ndarray,
                                     model: Any, y_true: int) -> float:
        """
        Discriminative Power (dipo): How well CFs discriminate between similar instances.

        Builds a 1-NN classifier trained on CFs ∪ {original_instance} and tests it on:
        - X_= : k nearest neighbors with same prediction as original
        - X_≠ : k nearest neighbors with different prediction than original

        Range: [0, 1]. Higher is better (CFs better separate the decision boundary).
        Returns NaN if insufficient data.
        """
        # Check if we have test data available
        if self.X_test is None or self.y_test_pred is None or len(cfs) == 0:
            return np.nan

        k = 10  # Number of neighbors to select from each group

        # Get prediction of original instance
        original_pred = model.predict(original_instance)[0]

        # Compute distances from original to all test instances
        distances = np.linalg.norm(self.X_test - original_instance, axis=1)
        nearest_indices = np.argsort(distances)

        # Select k instances with same prediction (X_=)
        X_equal = []
        y_equal = []
        equal_count = 0
        for idx in nearest_indices:
            if self.y_test_pred[idx] == original_pred:
                X_equal.append(self.X_test[idx])
                y_equal.append(1)  # Label instances with same prediction as 1
                equal_count += 1
                if equal_count >= k:
                    break

        # Select k instances with different prediction (X_≠)
        X_not_equal = []
        y_not_equal = []
        not_equal_count = 0
        for idx in nearest_indices:
            if self.y_test_pred[idx] != original_pred:
                X_not_equal.append(self.X_test[idx])
                y_not_equal.append(0)  # Label instances with different prediction as 0
                not_equal_count += 1
                if not_equal_count >= k:
                    break

        # Need at least one instance in each group
        if len(X_equal) == 0 or len(X_not_equal) == 0:
            return np.nan

        # Combine test sets
        X_test_combined = np.vstack([X_equal, X_not_equal])
        y_test_combined = np.array(y_equal + y_not_equal)

        # Train set: CFs (class 0) + original instance (class 1)
        X_train = np.vstack([cfs, original_instance])
        y_train = np.concatenate([np.zeros(len(cfs)), [1]])

        # Train 1-NN classifier
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)

        # Predict on test set
        y_pred_test = knn.predict(X_test_combined)

        # Compute accuracy
        accuracy = np.mean(y_pred_test == y_test_combined)

        return float(accuracy)

# ========== UTILITY FUNCTIONS ==========

def compute_metrics_for_batch(cfs_list: List[Dict], models: Dict[str, Any],
                             dataset_splits: Dict, metric_computer: CFQualityMetricsComputer,
                             model_name: str, limit_samples: int = None,
                             dipo_split: str = 'X_calibration') -> Dict[str, List[float]]:
    """
    Compute metrics for a batch of counterfactuals.

    Args:
        cfs_list: List of CF dicts from loaded NPZ (one per instance)
        models: Dict of trained models by name
        dataset_splits: Dataset splits (contains X_test, y_test for getting original instances)
        metric_computer: CFQualityMetricsComputer instance
        model_name: Which model was used to generate CFs
        limit_samples: Optional limit on number of samples to process
        dipo_split: Which split to use for dipo metric ('X_calibration' or 'X_test'). Default: 'X_calibration'

    Returns:
        Dict mapping metric_name -> list of values (one per instance in cfs_list)
    """
    model = models[model_name]
    metric_names = list(metric_computer.metrics.keys())
    batch_results = {name: [] for name in metric_names}

    # Determine how many samples to process
    n_samples = min(len(cfs_list), limit_samples) if limit_samples else len(cfs_list)

    # Set X_test for dipo metric (configurable which split to use)
    if dipo_split in dataset_splits:
        metric_computer.X_test = dataset_splits[dipo_split].values
        metric_computer.y_test_pred = model.predict(metric_computer.X_test)
    else:
        print(f"⚠️  Warning: dipo_split '{dipo_split}' not found in dataset_splits. Available: {list(dataset_splits.keys())}")

    skipped_count = 0
    for i in range(n_samples):
        cfs_dict = cfs_list[i]

        # Get original instance using instance_idx (same as notebook 1)
        instance_idx = cfs_dict.get('instance_idx', i)

        # Try to get from splits (this is what works in notebook 1)
        if 'X_test' in dataset_splits:
            original_instance = dataset_splits['X_test'].iloc[instance_idx].values.reshape(1, -1)
        elif 'X_calibration' in dataset_splits:
            original_instance = dataset_splits['X_calibration'].iloc[instance_idx].values.reshape(1, -1)
        else:
            raise ValueError(f"Could not find original instance for CF {i}")

        # Get true class
        y_true = cfs_dict.get('true_class', None)
        if y_true is None and 'y_test' in dataset_splits:
            y_true = dataset_splits['y_test'].iloc[instance_idx]

        # Convert to scalar if it's a pandas Series
        if hasattr(y_true, 'item'):
            y_true = y_true.item()

        # Compute metrics - LET IT FAIL
        metrics = metric_computer.compute_all_metrics(
            cfs_dict, original_instance, model, y_true
        )

        # Store results
        skipped = False
        for metric_name, value in metrics.items():
            if np.isnan(value):
                skipped = True
            batch_results[metric_name].append(value)

        if skipped:
            skipped_count += 1

    print(f"\n{'='*70}")
    print(f"Batch processing complete: {n_samples - skipped_count}/{n_samples} instances processed")
    print(f"Skipped: {skipped_count} instances (shape mismatch - preprocessing inconsistency)")
    print(f"{'='*70}")

    return batch_results
