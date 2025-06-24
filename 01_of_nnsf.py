# Implementing a SINGLE NEURON from scratch 

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

output = (inputs[0]*weights[0]+
          inputs[1]*weights[1]+
          inputs[2]*weights[2] + bias)

# Priting out the output.
print("Output of the First code:", output)

# But what happend If there are FOUR inputs rather then THREE.
# We will just add the value in the list.

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

# This method is not efficient but We have another one to make it.

inputs = [1, 2, 3, 2.5]
weightA = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# Output of the layer will store in this.
layer_output = []
# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zerod output of the given neuron
    neuron_output = 0

    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output
