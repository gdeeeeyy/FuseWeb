import numpy as np

# Define the Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Perceptron class definition
class Perceptron:
    def __init__(self, input_dim, learning_rate=0.1):
        self.learning_rate = learning_rate
        # Initialize weights with small random values and set bias to 0
        self.weights = np.random.randn(input_dim)
        self.bias = 0

    # Forward propagation
    def forward(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)

    # Training using the Perceptron learning rule (Gradient Descent)
    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                # Forward pass
                output = self.forward(X[i])
                
                # Calculate the error (difference between prediction and actual label)
                error = y[i] - output
                
                # Update weights and bias using the perceptron rule
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

            # Optionally, you can print the loss for monitoring
            if epoch % 10 == 0:
                loss = np.mean((y - self.forward(X))**2)
                print(f"Epoch {epoch}, Loss: {loss}")

    # Predict on new data
    def predict(self, X):
        predictions = self.forward(X)
        return np.round(predictions)

    # Evaluate the model accuracy
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100

# Define the OR/AND logic gate dataset
# Inputs for AND Gate (XOR can also be tested with similar modifications)
# For AND Gate, the output is 1 only if both inputs are 1
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])  # AND gate outputs

# Initialize Perceptron
perceptron = Perceptron(input_dim=2, learning_rate=0.1)

# Train the Perceptron on the AND dataset
perceptron.train(X_and, y_and, epochs=100)

# Evaluate accuracy
accuracy = perceptron.accuracy(X_and, y_and)
print(f"Accuracy: {accuracy}%")

# Test the model with new data
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = perceptron.predict(test_data)
print("Predictions on test data:", predictions)
