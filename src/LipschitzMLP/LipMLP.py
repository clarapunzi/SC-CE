import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.neural_network import MLPClassifier
import torch

class LipschitzMLPClassifier(MLPClassifier):
    def __init__(self, softplus_ci='auto', ci_scale=2.0, **kwargs):
        """
        Parameters:
        -----------
        softplus_ci : 'auto', float, or array-like
            - 'auto': Initialize each ci to ci_scale * ||Wi||_inf
            - float: Use same value for all layers
            - array: Specify per-layer values
        ci_scale : float
            Multiplicative factor when softplus_ci='auto' (default: 2.0)
        """
        super().__init__(**kwargs)
        self.softplus_ci = softplus_ci
        self.ci_scale = ci_scale
        self._ci_values = None  # Will be set after first fit
        
    def _get_ci_values(self):
        """Get or compute ci values based on current weights"""
        if self.softplus_ci == 'auto':
            # Initialize based on current weight norms
            ci_values = []
            for Wi in self.coefs_:
                # Compute l-infinity norm (max absolute row sum)
                linf_norm = np.max(np.sum(np.abs(Wi), axis=1))
                ci_values.append(self.ci_scale * linf_norm)
            return ci_values
        elif np.isscalar(self.softplus_ci):
            return [self.softplus_ci] * len(self.coefs_)
        else:
            return self.softplus_ci
    
    def _normalize_weights(self):
        """Apply Lipschitz normalization to weight matrices"""
        ci_values = self._get_ci_values()
        
        for i, (Wi, ci) in enumerate(zip(self.coefs_, ci_values)):
            # Compute l-infinity norm (max absolute row sum)
            absrowsum = np.sum(np.abs(Wi), axis=1, keepdims=True)
            # Scale to satisfy Lipschitz constraint
            scale = np.minimum(1.0, ci / np.maximum(absrowsum, 1e-10))
            self.coefs_[i] = Wi * scale
    
    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        """Override to add weight normalization after updates"""
        loss = super()._backprop(X, y, activations, deltas, coef_grads, intercept_grads)
        self._normalize_weights()
        return loss
    
    def fit(self, X, y):
        """Override fit to initialize ci on first call"""
        result = super().fit(X, y)
        
        # Store ci values after initialization for inspection
        if self._ci_values is None:
            self._ci_values = self._get_ci_values()
            print(f"Initialized ci values: {[f'{c:.2f}' for c in self._ci_values]}")
        
        self._normalize_weights()
        return result
    
    def partial_fit(self, X, y, classes=None):
        """Override to ensure normalization in online learning"""
        result = super().partial_fit(X, y, classes)
        self._normalize_weights()
        return result

if __name__ == "__main__":
    # Usage example
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=30, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # IMPORTANT: Scale your features first!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Standard MLP
    clf_standard = MLPClassifier(hidden_layer_sizes=(50, 30), 
                                 random_state=42,
                                 max_iter=500)
    clf_standard.fit(X_train, y_train)

    # Lipschitz MLP with per-layer constants
    clf_lipschitz = LipschitzMLPClassifier(
        hidden_layer_sizes=(50, 30),
        softplus_ci='auto',
        ci_scale=10.0,
        random_state=42,
        max_iter=500
    )
    clf_lipschitz.fit(X_train, y_train)

    print(f"Standard MLP score: {clf_standard.score(X_test, y_test):.3f}")
    print(f"Lipschitz MLP score: {clf_lipschitz.score(X_test, y_test):.3f}")



    '''
    # Example 1: Auto-initialize ci based on weight norms
    print("=" * 60)
    print("AUTO INITIALIZATION (recommended)")
    print("=" * 60)
    clf_auto = LipschitzMLPClassifier(
        hidden_layer_sizes=(50, 30),
        softplus_ci='auto',      # Auto-initialize
        ci_scale=2.0,            # ci = 2.0 * ||Wi||_inf
        max_iter=500,
        random_state=42
    )
    clf_auto.fit(X_train_scaled, y_train)
    print(f"Score: {clf_auto.score(X_test_scaled, y_test):.3f}")

    # Example 2: Fixed large value
    print("\n" + "=" * 60)
    print("FIXED LARGE VALUE")
    print("=" * 60)
    clf_fixed = LipschitzMLPClassifier(
        hidden_layer_sizes=(50, 30),
        softplus_ci=50.0,        # Fixed value
        max_iter=500,
        random_state=42
    )
    clf_fixed.fit(X_train_scaled, y_train)
    print(f"Score: {clf_fixed.score(X_test_scaled, y_test):.3f}")

    # Example 3: Per-layer values
    print("\n" + "=" * 60)
    print("PER-LAYER VALUES")
    print("=" * 60)
    clf_per_layer = LipschitzMLPClassifier(
        hidden_layer_sizes=(50, 30),
        softplus_ci=[100.0, 50.0, 20.0],  # Different per layer
        max_iter=500,
        random_state=42
    )
    clf_per_layer.fit(X_train_scaled, y_train)
    print(f"Score: {clf_per_layer.score(X_test_scaled, y_test):.3f}")

    # Compare with standard MLP
    print("\n" + "=" * 60)
    print("STANDARD MLP (baseline)")
    print("=" * 60)
    clf_standard = MLPClassifier(
        hidden_layer_sizes=(50, 30),
        max_iter=500,
        random_state=42
    )
    clf_standard.fit(X_train_scaled, y_train)
    print(f"Score: {clf_standard.score(X_test_scaled, y_test):.3f}")

    clf_standard = MLPClassifier(
        hidden_layer_sizes=(50, 30),
        max_iter=500,
        random_state=42
    )
    clf_standard.fit(X_train, y_train)
    print(f"Unscaled Score: {clf_standard.score(X_test, y_test):.3f}")'''


    import torch
    from torchLipMLP import LipschitzLinear, LipschitzMLP # 
    from torch import nn
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    input_size = X_train.shape[1]  # 20
    hidden_sizes = [50, 30]
    output_size = 2  # binary classification
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    model = LipschitzMLP(layer_sizes, lipschitz_lambda=0.001)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 500
    batch_size = 32

    for epoch in range(epochs):
        model.train()
        
        # Mini-batch training
        permutation = torch.randperm(X_train_tensor.size(0))
        epoch_loss = 0
        
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # Classification loss
            loss = criterion(outputs, batch_y)
            
            # Add Lipschitz regularization
            lip_reg = model.lipschitz_regularization()
            total_loss = loss + lip_reg
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_tensor)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_test_tensor).float().mean()
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Test Acc: {accuracy:.4f}')

    # Final evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).float().mean()
        print(f'\nFinal Test Accuracy: {accuracy:.4f}')