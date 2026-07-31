# TWO LAYERS connected = a tiny neural network.
# Layer 1's outputs become Layer 2's inputs.

def relu(x):
    return x if x > 0 else 0.0

def layer(inputs, weights, biases):
    """Compute a whole layer: returns one output per neuron."""
    outputs = []
    for n in range(len(weights)):
        weighted_sum = sum(inputs[i] * weights[n][i] for i in range(len(inputs)))
        outputs.append(relu(weighted_sum + biases[n]))
    return outputs

# --- The network's input ---
inputs = [2.0, 1.0, 3.0]

# --- Layer 1: 3 inputs -> 4 neurons ---
layer1_weights = [
    [0.5, -1.0, 0.2],
    [1.0,  0.0, -0.5],
    [-0.3, 0.8, 0.1],
    [0.2,  0.2, 0.2],
]
layer1_biases = [1.0, 0.0, -0.5, 0.5]

# --- Layer 2: 4 inputs -> 2 neurons ---
# (its inputs = layer 1's 4 outputs, so each neuron here needs 4 weights)
layer2_weights = [
    [0.3, -0.2, 0.5, 0.1],   # neuron 0
    [-0.1, 0.4, 0.2, -0.3],  # neuron 1
]
layer2_biases = [0.0, 1.0]

# --- Forward pass: data flows layer 1 -> layer 2 ---
layer1_out = layer(inputs, layer1_weights, layer1_biases)
print(f"Layer 1 output (4 numbers): {[round(x, 3) for x in layer1_out]}")

layer2_out = layer(layer1_out, layer2_weights, layer2_biases)
print(f"Layer 2 output (2 numbers): {[round(x, 3) for x in layer2_out]}")

print(f"\nNETWORK OUTPUT: {[round(x, 3) for x in layer2_out]}")