"""
This module contains functions for calculating distances between multiple vectors and a single reference vector.
"""

from scipy.spatial.distance import minkowski
from scipy.stats import spearmanr, kendalltau
import numpy as np

def center_scale_data(X, y):
    """Center and scale data to [-1,1] range"""
    # Center and scale X
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    X_scaled = 2 * (X - X_min) / (X_max - X_min) - 1
    
    # Center and scale y
    y_min = y.min()
    y_max = y.max()
    y_scaled = 2 * (y - y_min) / (y_max - y_min) - 1
    
    return X_scaled, y_scaled
def get_distances(X, y, policy="l2", **kwargs):
    """
    Calculate distances between multiple points X and a single point y.
    
    Parameters:
    -----------
    X : numpy array of shape (n_samples, n_features)
        Array of input vectors
    y : numpy array of shape (n_features,)
        Single reference vector
    policy : string, default="l2"
        The distance metric to use. Available metrics:
        - 'l2': Euclidean distance (L2 norm)
        - 'l1': Manhattan distance (L1 norm)
        - 'mae': Mean Absolute Error
        - 'cosine': Cosine distance
        - 'chebyshev': Chebyshev distance (L∞ norm)
        - 'minkowski': Minkowski distance with parameter p
        - 'canberra': Canberra distance
        - 'braycurtis': Bray-Curtis distance
        - 'hamming': Hamming distance
        - 'sqeuclidean': Squared Euclidean distance
        - 'correlation': Correlation distance
    **kwargs:
        p: power parameter for Minkowski distance
        
    Returns:
    --------
    distances : numpy array of shape (n_samples,)
        The distances between each input vector and the reference vector
    """
    if len(X.shape) != 2:
        raise ValueError(f"X must be 2D array. Got shape {X.shape}")
    #if len(y.shape) != 1:
    #    raise ValueError(f"y must be 1D array. Got shape {y.shape}")
    #if X.shape[1] != y.shape[0]:
        #raise ValueError(f"Incompatible dimensions: X.shape[1]={X.shape[1]} != y.shape[0]={y.shape[0]}")
    
    # Reshape y to (1, n_features) for broadcasting
    y = y.reshape(1, -1)
    
    if policy == "l0":
        return np.linalg.norm(X - y, ord=0, axis=1)
    elif policy == "l1":
        return np.linalg.norm(X - y, ord=1, axis=1)
    elif policy == "l2":
        return np.linalg.norm(X - y, ord=None,axis=1)
    elif policy == "inf":
        return np.linalg.norm(X - y, ord=np.inf, axis=1)
    elif policy == "-inf":
        return np.linalg.norm(X - y, ord=-np.inf, axis=1)
    elif policy == "sqeuclidean":
        return np.sum((X - y) ** 2, axis=1)
    
    elif policy == "mae":
        return np.mean(np.abs(X - y), axis=1)
    
    elif policy == "cosine":
        dot_product = np.sum(X * y, axis=1)
        norm_X = np.linalg.norm(X, axis=1)
        norm_y = np.linalg.norm(y)  # Single norm for y
        return 1 - dot_product / (norm_X * norm_y)
    
    elif policy == "correlation":
        X_centered = X - X.mean(axis=1, keepdims=True)
        y_centered = y - y.mean()
        dot_product = np.sum(X_centered * y_centered, axis=1)
        norm_X = np.linalg.norm(X_centered, axis=1)
        norm_y = np.linalg.norm(y_centered)
        return 1 - dot_product / (norm_X * norm_y)
    
    elif policy == "chebyshev":
        return np.max(np.abs(X - y), axis=1)
    
    elif policy == "minkowski":
        p = kwargs.get('p', 2)
        return np.power(np.sum(np.abs(X - y) ** p, axis=1), 1/p)
    
    elif policy == "canberra":
        numerator = np.abs(X - y)
        denominator = np.abs(X) + np.abs(y)
        # Avoid division by zero
        zero_mask = denominator == 0
        distances = numerator / np.where(zero_mask, 1, denominator)
        distances[zero_mask] = 0
        return np.sum(distances, axis=1)
    
    elif policy == "braycurtis":
        numerator = np.sum(np.abs(X - y), axis=1)
        denominator = np.sum(np.abs(X) + np.abs(y), axis=1)
        return numerator / denominator
    
    elif policy == "hamming":
        return np.mean(X != y, axis=1)

    elif policy == "jaccard":
        intersection = np.minimum(X, y).sum(axis=1)
        union = np.maximum(X, y).sum(axis=1)
        return 1 - intersection / union
    
    elif policy == "wasserstein":
        # Simple 1D Wasserstein distance implementation
        # For higher dimensions, consider using POT library
        X_sorted = np.sort(X, axis=1)
        y_sorted = np.sort(y, axis=1)
        return np.mean(np.abs(X_sorted - y_sorted), axis=1)
    
    ###################################################
    ####### CORRELATIONS AND RANK-BASED METRICS #######
    ###################################################
    elif policy == "spearman":
        return 1 - np.array([spearmanr(x, y.reshape(-1)).statistic for x in X])
    
    elif policy == "kendall":
        # Convert to correlation distance
        return 1 - np.array([kendalltau(x, y.reshape(-1)).statistic for x in X])
    
    elif policy == "centered_l2":
        X_scaled, y_scaled = center_scale_data(X, y.reshape(1, -1)[0])
        return np.linalg.norm(X_scaled - y_scaled.reshape(1, -1), axis=1)
    
    elif policy == "centered_cosine":
        X_scaled, y_scaled = center_scale_data(X, y.reshape(1, -1)[0])
        dot_product = np.sum(X_scaled * y_scaled.reshape(1, -1), axis=1)
        norm_X = np.linalg.norm(X_scaled, axis=1)
        norm_y = np.linalg.norm(y_scaled)
        return 1 - dot_product / (norm_X * norm_y)
    else:
        raise ValueError(f"Unknown distance policy: {policy}")
    
metrics = ["l0", "l1", "l2", "inf", "-inf", "sqeuclidean", "mae", "cosine", "correlation", "chebyshev", "minkowski", "canberra", "braycurtis", "hamming", "jaccard", "wasserstein", "spearman", "kendall", "centered_l2", "centered_cosine"]
