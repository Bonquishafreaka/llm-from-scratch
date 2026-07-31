# Read the real text file
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Build the vocabulary from the actual data
chars = sorted(set(text))
vocab_size = len(chars)

# The two lookup tables
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# The translator functions
def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return "".join([itos[i] for i in ids])

import torch

# Encode the ENTIRE dataset into one long tensor of numbers
data = torch.tensor(encode(text), dtype=torch.long)

# Split: first 90% for training, last 10% held back for validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Demo output -- only runs when you run tokenizer.py DIRECTLY, not on import
if __name__ == "__main__":
    print("length of text in characters:", len(text))
    print("vocab size:", vocab_size)
    print("vocabulary:", "".join(chars))
    print()
    print("first 200 characters of the data:")
    print(text[:200])
    print()
    print("those same 200 characters encoded as numbers:")
    print(encode(text[:200]))
    print()
    print("full dataset as a tensor:", data.shape, data.dtype)
    print("training set size:", len(train_data))
    print("validation set size:", len(val_data))