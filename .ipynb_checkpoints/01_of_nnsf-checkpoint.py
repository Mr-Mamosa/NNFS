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







