# ONE NEURON, built by hand in pure Python.
# A neuron: multiply each input by its weight, sum them, add bias, activate.

# The inputs coming into the neuron (3 inputs)
inputs = [2.0, 1.0, 3.0]

# The neuron's weights -- one per input. These are the "learnable" numbers.
weights = [0.5, -1.0, 0.2]

# The neuron's bias -- one extra number added on.
bias = 1.0

# --- Step 1: the weighted sum (multiply each input by its weight, add them up) ---
weighted_sum = 0.0
for i in range(len(inputs)):
    weighted_sum += inputs[i] * weights[i]
    print(f"  input {inputs[i]} * weight {weights[i]} = {inputs[i] * weights[i]}")

print(f"weighted sum = {weighted_sum}")

# --- Step 2: add the bias ---
z = weighted_sum + bias
print(f"after adding bias {bias}: {z}")

# --- Step 3: the activation function (ReLU: keep positives, turn negatives to 0) ---
def relu(x):
    return x if x > 0 else 0.0

output = relu(z)
print(f"after ReLU activation: {output}")

print(f"\nNEURON OUTPUT: {output}")