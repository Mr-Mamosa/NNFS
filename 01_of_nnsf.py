# ----------------------------------------------------
# Implementing a Single Neuron (Manual Computation)
# ----------------------------------------------------

# The input will be either actual training data or the outputs from the previous layer
inputs = [1, 2, 3]

# Each input has a weight (trainable parameter)
weights = [0.2, 0.8, -0.5]

# Bias shifts the output — another trainable parameter
bias = 2

# Neuron output = weighted sum of inputs + bias
output1 = (inputs[0]*weights[0] +
           inputs[1]*weights[1] +
           inputs[2]*weights[2] + bias)

print("Output1 (3 inputs):", output1)


# ----------------------------------------------------
# Extending to 4 inputs (same logic)
# ----------------------------------------------------

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output2 = (inputs[0]*weights[0] +
           inputs[1]*weights[1] +
           inputs[2]*weights[2] +
           inputs[3]*weights[3] + bias)

print("Output2 (4 inputs):", output2)


# ----------------------------------------------------
# Fully Connected Layer with 3 Neurons (Manual)
# ----------------------------------------------------

# Inputs to the layer (same for all neurons)
inputs = [1, 2, 3, 2.5]

# Each neuron has its own set of weights and a bias
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

# Manually calculating the output for each neuron
output3 = [
    inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
    inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
    inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3
]

print("Output of 3 Neurons (manual):", output3)


# ----------------------------------------------------
# Same Layer as Above — Using Loop for Efficiency
# ----------------------------------------------------

weights_set = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2, 3, 0.5]

layer_output = []

# Iterate through each neuron’s weights and bias
for neuron_weights, neuron_bias in zip(weights_set, biases):
    neuron_output = 0
    for i, w in zip(inputs, neuron_weights):
        neuron_output += i * w
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print("Layer Output (loop version):", layer_output)


# ----------------------------------------------------
# Vector Multiplication & Dot Product (Manual)
# ----------------------------------------------------

# A dot product is a sum of products of corresponding elements in two vectors
a = [1.0, 2.0, 3.0]
b = [0.2, 0.8, -0.5]

dot_product = sum(i * j for i, j in zip(a, b))
print("Manual Dot Product:", dot_product)


# ----------------------------------------------------
# Single Neuron using NumPy
# ----------------------------------------------------

import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

output = np.dot(inputs, weights) + bias
print("Single Neuron Output (NumPy):", output)


# ----------------------------------------------------
# Full Layer using NumPy (Cleaner & Faster)
# ----------------------------------------------------

inputs = [1.0, 2.0, 3.0, 2.5]

weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2.0, 3.0, 0.5]

layer_output = np.dot(weights, inputs) + biases
print("Layer Output (NumPy):", layer_output)


# ----------------------------------------------------
# Matrix Transposition Example
# ----------------------------------------------------

M = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("Transpose of Matrix:\n", M.T)


# ----------------------------------------------------
# Implementing Transpose in Neural Layer Code
# ----------------------------------------------------

inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.17, 0.87],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2.0, 3.0, 0.5]

# Transposing weights to perform matrix multiplication
outputs = np.dot(inputs, np.array(weights).T) + biases

print("Batch Output (with Transposed Weights):", outputs)
