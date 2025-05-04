import numpy as np

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

# Binary Cross-Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # Prevent log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Accuracy
def accuracy(y_true, y_pred):
    preds = y_pred > 0.5
    return np.mean(preds == y_true)

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 2)         # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 1).astype(int).reshape(-1, 1)  # Linearly separable target

# Network architecture
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 1000

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)

    # Loss
    loss = binary_cross_entropy(y, y_pred)

    # Backward pass
    dz2 = y_pred - y
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(z1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if (epoch + 1) % 100 == 0:
        acc = accuracy(y, y_pred)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

# Final weights and accuracy
print("\nFinal Weights and Biases:")
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)
print("\nFinal Accuracy:", accuracy(y, y_pred))
