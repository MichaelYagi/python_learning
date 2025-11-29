import numpy as np

# Trains a 2-layer neural net (2 inputs → 2 hidden neurons → 1 output).
# Learns the XOR function, a classic test for non-linear learning.
# Uses sigmoid activation and gradient descent.
# This shows the network has learned XOR: outputs near 0 for [0,0] and [1,1], near 1 for [0,1] and [1,0].

# --- Generate simple dataset: XOR problem ---
X = np.array([[0,0],[0,1],[1,0],[1,1]])   # inputs
y = np.array([[0],[1],[1],[0]])           # expected outputs

# --- Helper functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# --- Initialize weights ---
np.random.seed(42)
input_size = 2
hidden_size = 2
output_size = 1

W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
W2 = np.random.uniform(-1, 1, (hidden_size, output_size))

# --- Training loop ---
lr = 0.1
epochs = 10000

for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2)
    final_output = sigmoid(final_input)

    # Error
    error = y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)

    # Update weights
    W2 += hidden_output.T.dot(d_output) * lr
    W1 += X.T.dot(d_hidden) * lr

    # Print loss occasionally
    if epoch % 2000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# --- Test results ---
print("\nFinal outputs after training:")
print(final_output)
