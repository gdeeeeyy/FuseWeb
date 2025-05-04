import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)
def relu_deriv(x):
    return (x > 0).astype(float)

# Binary Cross-Entropy
def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Accuracy
def accuracy(y_true, y_pred):
    return np.mean((y_pred > 0.5) == y_true)

# Dataset
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
y = y.reshape(-1, 1)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network class
class SimpleNN:
    def __init__(self, input_dim, hidden_units, activation='relu', learning_rate=0.1):
        self.hidden_units = hidden_units
        self.lr = learning_rate

        self.W1 = np.random.randn(input_dim, hidden_units) * 0.1
        self.b1 = np.zeros((1, hidden_units))
        self.W2 = np.random.randn(hidden_units, 1) * 0.1
        self.b2 = np.zeros((1, 1))

        if activation == 'relu':
            self.act = relu
            self.act_deriv = relu_deriv
        elif activation == 'sigmoid':
            self.act = sigmoid
            self.act_deriv = sigmoid_deriv
        else:
            raise ValueError("Only 'relu' and 'sigmoid' supported")

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.act(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)  # Output layer uses sigmoid
        return self.a2

    def backward(self, X, y, output):
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, self.W2.T) * self.act_deriv(self.z1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000):
        acc_list, loss_list = [], []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = binary_cross_entropy(y, output)
            acc = accuracy(y, output)

            self.backward(X, y, output)

            acc_list.append(acc)
            loss_list.append(loss)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")

        return acc_list, loss_list

# Train with different hidden layer sizes
def experiment(hidden_units_list):
    for h in hidden_units_list:
        print(f"\nTraining with {h} hidden units:")
        model = SimpleNN(input_dim=2, hidden_units=h, activation='relu', learning_rate=0.1)
        acc, loss = model.train(X_train, y_train, epochs=1000)

        # Test accuracy
        test_output = model.forward(X_test)
        test_acc = accuracy(y_test, test_output)
        print(f"Test Accuracy: {test_acc:.4f}")

        # Plot
        plt.plot(acc, label=f"{h} units")

    plt.title("Training Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the experiment
experiment([2, 4, 8, 16])
