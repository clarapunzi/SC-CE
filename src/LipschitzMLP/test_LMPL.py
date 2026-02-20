import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
from LipMLP import LipschitzMLPClassifier
from torchLipMLP import LipschitzMLP

# Generate data
X, y = make_classification(n_samples=1000, n_features=30, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# ============================================================================
# EXPERIMENT 1: Standard PyTorch MLP (baseline)
# ============================================================================
print("=" * 70)
print("STANDARD PYTORCH MLP (no Lipschitz constraint)")
print("=" * 70)

class StandardMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

layer_sizes = [30, 50, 30, 2]
model_standard = StandardMLP(layer_sizes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_standard.parameters(), lr=0.001)

epochs = 500
batch_size = 32

for epoch in range(epochs):
    model_standard.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
        
        optimizer.zero_grad()
        outputs = model_standard(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Evaluate
model_standard.eval()
with torch.no_grad():
    outputs = model_standard(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")

# ============================================================================
# COMPARE WITH SKLEARN
# ============================================================================
print("\n" + "=" * 70)
print("SKLEARN MLP (baseline)")
print("=" * 70)

clf_sklearn = MLPClassifier(
    hidden_layer_sizes=layer_sizes[1:-1],  # [50, 30]
    max_iter=500,
    random_state=42
)
clf_sklearn.fit(X_train_scaled, y_train)
print(f"Test Accuracy: {clf_sklearn.score(X_test_scaled, y_test):.4f}")



print("\n" + "=" * 70)
print("SKLEARN MLP (with Lipschitz constraint λ=0, auto ci init)")
print("=" * 70)

clf_sklearn = LipschitzMLPClassifier(
    hidden_layer_sizes=tuple(layer_sizes[1:-1]),  # (50, 30)
    softplus_ci='auto',      # This initializes based on weight norms
    ci_scale=2.0,            # ci = 2.0 * initial_norm
    max_iter=500,
    random_state=42
)
clf_sklearn.fit(X_train_scaled, y_train)
print(f"Test Accuracy: {clf_sklearn.score(X_test_scaled, y_test):.4f}")


# ============================================================================
# EXPERIMENT 2: Lipschitz MLP with LEARNABLE ci (paper's method)
# ============================================================================
print("\n" + "=" * 70)
print("LIPSCHITZ MLP with LEARNABLE ci (auto-initialized)")
print("=" * 70)

model_lipschitz = LipschitzMLP(
    layer_sizes=layer_sizes,
    init_cis='auto',           # Auto-initialize based on weight norms
    lipschitz_lambda=0.001     # Regularization weight
)

print(f"Initial ci values: {model_lipschitz.get_lipschitz_constants()}")
print(f"Initial Lipschitz bound: {model_lipschitz.get_lipschitz_bound():.2f}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_lipschitz.parameters(), lr=0.001)

for epoch in range(epochs):
    model_lipschitz.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
        
        optimizer.zero_grad()
        outputs = model_lipschitz(batch_x)
        
        # Classification loss + Lipschitz regularization
        loss = criterion(outputs, batch_y)
        lip_reg = model_lipschitz.lipschitz_regularization()
        total_loss = loss + lip_reg
        
        total_loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        model_lipschitz.eval()
        with torch.no_grad():
            outputs = model_lipschitz(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == y_test_tensor).float().mean()
            print(f"Epoch {epoch+1}: Test Acc = {acc:.4f}, "
                  f"ci values = {[f'{c:.1f}' for c in model_lipschitz.get_lipschitz_constants()]}, "
                  f"Lip bound = {model_lipschitz.get_lipschitz_bound():.1f}")

# Final evaluation
model_lipschitz.eval()
with torch.no_grad():
    outputs = model_lipschitz(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Final ci values: {model_lipschitz.get_lipschitz_constants()}")
    print(f"Final Lipschitz bound: {model_lipschitz.get_lipschitz_bound():.2f}")

# ============================================================================
# EXPERIMENT 3: Different lambda values
# ============================================================================
print("\n" + "=" * 70)
print("TESTING DIFFERENT LAMBDA VALUES")
print("=" * 70)

for lam in [0.0, 0.0001, 0.001, 0.01]:
    model = LipschitzMLP(layer_sizes=layer_sizes, init_cis='auto', lipschitz_lambda=lam)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            lip_reg = model.lipschitz_regularization()
            total_loss = loss + lip_reg
            total_loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).float().mean()
        print(f"λ = {lam:.4f}: Acc = {accuracy:.4f}, Lip bound = {model.get_lipschitz_bound():.1f}")