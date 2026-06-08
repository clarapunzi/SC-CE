"""
Scikit-learn compatible wrapper for PyTorch Lipschitz MLP
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import numpy as np
from .torchLipMLP import LipschitzMLP


class TorchLipschitzMLPClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for PyTorch Lipschitz MLP"""
    
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        init_cis='auto',
        lipschitz_lambda=0.001,
        learning_rate=0.001,
        max_iter=200,
        batch_size=32,
        random_state=None,
        verbose=False,
        use_internal_scaling=False  # NEW: control scaling behavior
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.init_cis = init_cis
        self.lipschitz_lambda = lipschitz_lambda
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.use_internal_scaling = use_internal_scaling  # NEW
        
        self.model_ = None
        self.scaler_ = None
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit the model"""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        
        # Only scale if not already scaled by pipeline
        if self.use_internal_scaling:
            if self.scaler_ is None:
                self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X  # Assume already scaled by pipeline
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y)
        
        # Determine input/output sizes
        input_size = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Build layer sizes
        layer_sizes = [input_size] + list(self.hidden_layer_sizes) + [n_classes]
        
        # Create model
        self.model_ = LipschitzMLP(
            layer_sizes,
            init_cis=self.init_cis,
            lipschitz_lambda=self.lipschitz_lambda
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.max_iter):
            self.model_.train()
            
            # Shuffle data
            permutation = torch.randperm(X_tensor.size(0))
            epoch_loss = 0
            
            for i in range(0, X_tensor.size(0), self.batch_size):
                indices = permutation[i:i+self.batch_size]
                batch_x, batch_y = X_tensor[indices], y_tensor[indices]
                
                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                
                # Classification loss + Lipschitz regularization
                loss = criterion(outputs, batch_y)
                lip_reg = self.model_.lipschitz_regularization()
                total_loss = loss + lip_reg
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            # Optional: print progress
            if self.verbose and (epoch + 1) % 50 == 0:
                train_acc = self.score(X, y)
                print(f'Epoch {epoch+1}/{self.max_iter}, Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        self.model_.eval()
        
        # Only scale if using internal scaling
        if self.use_internal_scaling and self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.numpy()
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        self.model_.eval()
        
        # Only scale if using internal scaling
        if self.use_internal_scaling and self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            proba = torch.softmax(outputs, dim=1)
        
        return proba.numpy()
    
    def score(self, X, y):
        """Compute accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'init_cis': self.init_cis,
            'lipschitz_lambda': self.lipschitz_lambda,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'use_internal_scaling': self.use_internal_scaling
        }
    
    def set_params(self, **params):
        """Set parameters"""
        for key, value in params.items():
            setattr(self, key, value)
        return self