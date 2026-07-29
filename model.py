import torch
import torch.nn as nn

# Pull in the vocabulary size and the data from your tokenizer
from tokenizer import vocab_size, train_data, val_data

# --- A setting we get to choose ---
n_embd = 32   # embedding dimension: each token becomes a list of 32 numbers

block_size = 8   # how many tokens the model looks at at once (context length)

# --- The embedding layer ---
# A lookup table: one learnable vector (length n_embd) for every token in the vocab.
token_embedding_table = nn.Embedding(vocab_size, n_embd)

position_embedding_table = nn.Embedding(block_size, n_embd)

sample = train_data[:block_size]     # grab block_size token ids
print("input token ids:", sample)
print("input shape:", sample.shape)

tok_emb = token_embedding_table(sample)                          # what each token is
pos_emb = position_embedding_table(torch.arange(block_size))     # where each position is
x = tok_emb + pos_emb                                            # combine them

print()
print("token embeddings shape:", tok_emb.shape)
print("position embeddings shape:", pos_emb.shape)
print("combined shape:", x.shape)

import torch.nn.functional as F

# --- One head of self-attention ---
head_size = 16   # size of the query/key/value vectors

# Three linear layers: each turns a token's vector into a query, key, or value
key   = nn.Linear(n_embd, head_size, bias=False)
query = nn.Linear(n_embd, head_size, bias=False)
value = nn.Linear(n_embd, head_size, bias=False)

# x is our combined embeddings from before, shape [block_size, n_embd]
k = key(x)     # [block_size, head_size]  -- what each token offers
q = query(x)   # [block_size, head_size]  -- what each token seeks
v = value(x)   # [block_size, head_size]  -- what each token hands over

# Compare every query against every key -> relevance scores
scores = q @ k.transpose(-2, -1)          # [block_size, block_size]
scores = scores / (head_size ** 0.5)      # scale (keeps numbers stable)

# Mask: block each token from seeing tokens AFTER it
tril = torch.tril(torch.ones(block_size, block_size))
scores = scores.masked_fill(tril == 0, float('-inf'))

# Turn scores into weights that sum to 1 (softmax)
weights = F.softmax(scores, dim=-1)

# Blend the values using those weights
out = weights @ v                          # [block_size, head_size]

print("\nattention weights shape:", weights.shape)
print("output shape:", out.shape)
print("\nattention weights (row = each token, columns = who it attends to):")
print(weights)