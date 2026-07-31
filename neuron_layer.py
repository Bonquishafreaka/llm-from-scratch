# A LAYER of neurons, by hand. Several neurons, same inputs, each with its own weights.

inputs = [2.0, 1.0, 3.0]   # 3 inputs, shared by all neurons in the layer

# This layer has 4 neurons. Each neuron has its OWN set of 3 weights and 1 bias.
layer_weights = [
    [0.5, -1.0, 0.2],    # neuron 0's weights
    [1.0,  0.0, -0.5],   # neuron 1's weights
    [-0.3, 0.8, 0.1],    # neuron 2's weights
    [0.2,  0.2, 0.2],    # neuron 3's weights
]
layer_biases = [1.0, 0.0, -0.5, 0.5]   # one bias per neuron

def relu(x):
    return x if x > 0 else 0.0

# Compute every neuron's output
layer_outputs = []
for n in range(len(layer_weights)):          # for each neuron
    weights = layer_weights[n]
    bias = layer_biases[n]
    weighted_sum = 0.0
    for i in range(len(inputs)):             # weighted sum over its inputs
        weighted_sum += inputs[i] * weights[i]
    output = relu(weighted_sum + bias)
    layer_outputs.append(output)
    print(f"neuron {n}: weighted_sum={weighted_sum:.2f} + bias={bias} -> ReLU -> {output:.2f}")

print(f"\nLAYER OUTPUT: {layer_outputs}")