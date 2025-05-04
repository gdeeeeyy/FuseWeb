import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Loss function
def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def accuracy(y_true, y_pred):
    preds = y_pred > 0.5
    return np.mean(preds == y_true)

# Generate dataset
def generate_data():
    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = ((X[:, 0] + X[:, 1]) > 1).astype(int).reshape(-1, 1)  # Linearly separable
    return X, y

# Feedforward Neural Network class
class SimpleNN:
    def __init__(self, input_size, hidden_size, activation='sigmoid', lr=0.1):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1)
        self.b2 = np.zeros((1, 1))
        self.lr = lr
        self.activation = activation
        self.W1_history = []

        # Choose activation
        if activation == 'sigmoid':
            self.act = sigmoid
            self.act_deriv = sigmoid_derivative
        elif activation == 'relu':
            self.act = relu
            self.act_deriv = relu_derivative
        else:
            raise ValueError("Unsupported activation")

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.act(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)  # output layer is always sigmoid
        return self.a2

    def backward(self, X, y, output):
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, self.W2.T) * self.act_deriv(self.z1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=1000, visualize=True):
        acc_list, loss_list = [], []

        for epoch in range(epochs):
            output = self.forward(X)
            loss = binary_cross_entropy(y, output)
            acc = accuracy(y, output)

            self.backward(X, y, output)

            acc_list.append(acc)
            loss_list.append(loss)
            self.W1_history.append(self.W1.copy())

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

        if visualize:
            self.visualize_training(acc_list, loss_list)

    def visualize_training(self, acc_list, loss_list):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(loss_list, label='Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(acc_list, label='Accuracy', color='green')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.grid(True)
        plt.show()

        # Visualize how W1 evolves (first neuron only)
        plt.figure(figsize=(8, 4))
        weights_over_time = [w[:, 0] for w in self.W1_history]
        weights_over_time = np.array(weights_over_time)
        plt.plot(weights_over_time)
        plt.title("Evolution of W1[:, 0] over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Weight value")
        plt.legend(["Feature 1", "Feature 2"])
        plt.grid(True)
        plt.show()

# Run training with two different hidden sizes and activations
def run_experiments():
    X, y = generate_data()

    print("▶️ Sigmoid activation, 4 hidden units")
    model1 = SimpleNN(input_size=2, hidden_size=4, activation='sigmoid', lr=0.1)
    model1.train(X, y, epochs=1000)

    print("\n▶️ ReLU activation, 6 hidden units")
    model2 = SimpleNN(input_size=2, hidden_size=6, activation='relu', lr=0.1)
    model2.train(X, y, epochs=1000)

if __name__ == "__main__":
    run_experiments()
