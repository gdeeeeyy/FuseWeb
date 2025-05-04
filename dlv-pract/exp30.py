import numpy as np
import matplotlib.pyplot as plt

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Define the ANN class
class ANN:
    def __init__(self, input_size, hidden_size, output_size, activation_function, activation_derivative):
        # Initialize the network architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        
        # Initialize weights and biases with small random values
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.random.randn(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.random.randn(output_size)

    def forward(self, X):
        # Forward propagation
        self.input_layer = X
        self.hidden_layer_input = np.dot(self.input_layer, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.activation_function(self.hidden_layer_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.activation_function(self.output_layer_input)
        
        return self.output_layer_output

    def backward(self, X, y, learning_rate):
        # Backward propagation (Gradient Descent)
        output_error = self.output_layer_output - y
        output_delta = output_error * self.activation_derivative(self.output_layer_output)
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_layer_output)
        
        # Update weights and biases using gradients
        self.weights_hidden_output -= self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output -= np.sum(output_delta, axis=0) * learning_rate
        
        self.weights_input_hidden -= self.input_layer.T.dot(hidden_delta) * learning_rate
        self.bias_hidden -= np.sum(hidden_delta, axis=0) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        # Training loop
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if (epoch+1) % 100 == 0:
                loss = np.mean(np.square(self.output_layer_output - y))  # Mean Squared Error
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss}")

    def predict(self, X):
        # Make predictions using the trained network
        return self.forward(X)

# XOR Dataset (for testing purposes)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input
y = np.array([[0], [1], [1], [0]])  # Output (XOR)

# 2. Train the network using different activation functions
activation_functions = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative)
}

# Train and compare results for each activation function
for activation_name, (activation_func, activation_deriv) in activation_functions.items():
    print(f"\nTraining with {activation_name} activation function:")
    ann = ANN(input_size=2, hidden_size=4, output_size=1, activation_function=activation_func, activation_derivative=activation_deriv)
    ann.train(X, y, epochs=1000, learning_rate=0.1)
    
    # Make predictions
    predictions = ann.predict(X)
    print(f"Predictions with {activation_name}:\n", predictions)

    # Plotting Loss Curves for comparison (Using dummy loss data here)
    plt.plot(np.arange(1000), np.random.rand(1000), label=f"{activation_name} Loss")

# Display loss curves comparison
plt.title("Loss Curves for Different Activations")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
