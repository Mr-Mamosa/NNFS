# Implementing a SINGLE NEURON from scratch 
#*****Output1******

# The input will be either actual training data or the outputs of neurons from the previous layer in the neural network.
inputs = [1,2,3]

# Each inputs data has WEIGHT associate with it.
# Weight are the adjustable PARAMETERS that will change the input.
# The values for weights and biases are what get “trained,”.
# This is what make a model actually work (or not work). We’ll start by making up weights for now.
weights = [0.2, 0.8, -0.5]

# Bias is tendency of a model to consistently make errors in a specific direction essentially 
# --making it unable to capture the true underlying relationship between inputs and outputs.
bias = 2


# Output
output1 = (inputs[0]*weights[0]+
          inputs[1]*weights[1]+
          inputs[2]*weights[2] + bias)

# Priting out the output.
print("Output1:", output1)

# But what happend If there are FOUR inputs rather then THREE.
# We will just add the value in the list.

#*****Output2******

inputs = [ 1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output2 = (inputs[0] * weights[0] +
          inputs[1] * weights[1] +
          inputs[2] * weights[2] +
          inputs[3] * weights[3] + bias)

print("Output of the First code:", output2)

# These are the example of SINGLE NEURON


##----- LAYERS OF NEURONS -----

# Let's say we have a scenario with THREE neurons in a layer and FOUR inputs.
# https://www.youtube.com/watch?v=Uvngs6sWyBg

#*****Output3******

inputs = [1 , 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output2 = [
    # Neuron 1
    inputs[0]*weights1[0] +
    inputs[1]*weights1[1] +
    inputs[2]*weights1[2] +
    inputs[3]*weights1[3] + bias1,

    # Neuron 2
    inputs[0]*weights2[0] +
    inputs[1]*weights2[1] +
    inputs[2]*weights2[2] +
    inputs[3]*weights2[3] + bias2,

    # Neuron 3
    inputs[0]*weights3[0] +
    inputs[1]*weights3[1] +
    inputs[2]*weights3[2] +
    inputs[3]*weights3[3] + bias3,
    
] 

print("Output of 3 neurons: ", output2)

"""
In this code, we have three sets of weights and three biases, which define three neurons. Each
neuron is “connected” to the same inputs. The difference is in the separate weights and bias
that each neuron applies to the input. This is called a fully connected neural network — every
in the current layer has connections to every neuron from the previous layer.
"""

#*****Output4******
# This method is not efficient but We have another one to make it.

inputs = [1, 2, 3, 2.5]
weightA = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# This will store the output of the Current layer.
layer_output = []
# This will iterate Weights and Biases into this variable(neuron_weights, neuron_bias).
for neuron_weights, neuron_bias in zip(weightA, biases):
   
    # Zero output of the given neuron.
    neuron_output = 0
    
    # Here iteraion the WEIGHTS and INPUT. 
    for n_input, weight in zip(inputs, neuron_weights):

        # Multiplying input and weight
        neuron_output += n_input*weight

    # Now adding up the bias.
    neuron_output += neuron_bias

    # Adding up the output tp the Nueron Variable.
    layer_output.append(neuron_output)
print(layer_output)

# ------------------------------------------
#  Vector Multiplication & Dot Product Notes
# ------------------------------------------

#  Vectors = 1D arrays (like Python lists)
#  We multiply inputs and weights element-wise
#  Dot product = cleaner, faster way to do this

#  Dot product = sum of input[i] * weight[i]
#     - Result: scalar (single number)
#     - Used heavily in neural networks for weighted sums

#  Cross product (less common) results in a vector
#  Tensors, vectors, arrays → all similar in code
#  Don't get confused by complex terms — it's just math!

# Example:
inputs = [1.0, 2.0, 3.0]
weights = [0.2, 0.8, -0.5]

# Manual dot product
dot = sum(i * w for i, w in zip(inputs, weights))
print("Dot Product:", dot)  # Output: 0.3

# ------------------------------------------
#  A Single neuron with numpy
# ------------------------------------------
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

output = np.dot(inputs, weights) + bias

print(output)

# ------------------------------------------
#  A Layer of neuron with numpy
# ------------------------------
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

layer_output = np.dot(weights, inputs) + biases

print(layer_output)

# ------------------------------------------
#  Example of transpose
# ------------------------------------------

import numpy as np

M = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(M.T)

output = [[1 4]
          [2 5]
          [3 6]]

# ------------------------------------------
#  Implimenting transpostion to code
# ------------------------------------------

import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights =[[0.2, 0.8, -0.5, 1.0],
          [0.5, -0.91, 0.17, 0.87],
          [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

outputs = np.dot(inputs, np.array(weights).T) + biases

print(outputs)









