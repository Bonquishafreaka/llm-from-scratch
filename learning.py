# A single neuron LEARNING to produce a target output.
# This is gradient descent, by hand.

inputs = [2.0, 1.0, 3.0]
weights = [0.5, -1.0, 0.2]   # starting weights (random-ish)
bias = 1.0

target = 5.0                 # we WANT the neuron to output this
learning_rate = 0.01         # how big each nudge is

def forward():
    # weighted sum + bias (no activation here, to keep the math clean)
    return sum(inputs[i] * weights[i] for i in range(len(inputs))) + bias

# --- Training loop ---
for step in range(50):
    output = forward()
    error = output - target          # how far off we are (positive = too high)
    loss = error ** 2                # squared error -- always positive, our "wrongness"

    # How to nudge each weight to reduce the error:
    # if error is positive (output too high), decrease weights that had positive input.
    # the nudge for each weight is proportional to error * that weight's input.
    for i in range(len(weights)):
        gradient = 2 * error * inputs[i]        # how much this weight affects the loss
        weights[i] -= learning_rate * gradient  # step downhill
    bias -= learning_rate * 2 * error           # nudge bias too

    if step % 5 == 0:
        print(f"step {step:2d} | output={output:.3f} | target={target} | loss={loss:.4f}")

print(f"\nfinal output: {forward():.3f}  (target was {target})")
print(f"final weights: {[round(w, 3) for w in weights]}")