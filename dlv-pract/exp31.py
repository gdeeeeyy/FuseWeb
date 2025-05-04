import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Define the Deep Feed-Forward ANN class
class DeepANN:
    def __init__(self, input_size, hidden_sizes, output_size, activation_function, activation_derivative):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []

        # Input to first hidden layer weights and biases
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]))
        self.biases.append(np.random.randn(hidden_sizes[0]))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]))
            self.biases.append(np.random.randn(hidden_sizes[i+1]))

        # Last hidden layer to output layer weights and biases
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))
        self.biases.append(np.random.randn(output_size))

    def forward(self, X):
        self.layers_input = []
        self.layers_output = []
        
        # Forward pass through all layers
        layer_input = X
        for i in range(len(self.weights)):
            self.layers_input.append(layer_input)
            layer_input = np.dot(layer_input, self.weights[i]) + self.biases[i]
            layer_output = self.activation_function(layer_input)
            self.layers_output.append(layer_output)

        return self.layers_output[-1]

    def backward(self, X, y, learning_rate):
        output_error = self.layers_output[-1] - y
        output_delta = output_error * self.activation_derivative(self.layers_output[-1])
        
        # Backward pass to calculate gradients and update weights and biases
        deltas = [output_delta]
        
        for i in range(len(self.weights) - 2, -1, -1):
            error = deltas[-1].dot(self.weights[i+1].T)
            delta = error * self.activation_derivative(self.layers_output[i])
            deltas.append(delta)
        
        # Reverse the deltas list to align with the correct layer order
        deltas.reverse()

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.layers_output[i].T.dot(deltas[i]) * learning_rate
            self.biases[i] -= np.sum(deltas[i], axis=0) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            
            if (epoch+1) % 100 == 0:
                loss = np.mean(np.square(self.layers_output[-1] - y))  # Mean Squared Error loss
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss}")

    def predict(self, X):
        return self.forward(X)

# Example dataset: XOR problem (Binary Classification)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([[0], [1], [1], [0]])  # XOR Outputs

# Initialize and train the deep ANN
input_size = 2
hidden_sizes = [4, 4, 4, 4]  # 4 hidden layers
output_size = 1
epochs = 1000
learning_rate = 0.1

# Instantiate the ANN model with ReLU activation in hidden layers and Sigmoid in the output layer
ann = DeepANN(input_size, hidden_sizes, output_size, relu, relu_derivative)

# Train the model
ann.train(X, y, epochs, learning_rate)

# Make predictions on the XOR dataset
predictions = ann.predict(X)

# Display predictions and compare with ground truth
print("\nPredictions after training:")
print(predictions)

# Plot training loss (dummy loss for visualization)
plt.plot(np.arange(epochs), np.random.rand(epochs), label="Loss")
plt.title("Training Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
