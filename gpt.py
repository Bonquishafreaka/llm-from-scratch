import torch
import torch.nn as nn
import torch.nn.functional as F
from block import Block

# --- hyperparameters (our settings) ---
vocab_size = 65      # from your tokenizer
n_embd = 32
block_size = 8
head_size = 32


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block = Block(n_embd, head_size, block_size)   # one block for now
        self.ln_f = nn.LayerNorm(n_embd)                    # final normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)        # -> one score per character

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                       # [B, T, n_embd]
        pos_emb = self.position_embedding_table(torch.arange(T))        # [T, n_embd]
        x = tok_emb + pos_emb                                           # combine
        x = self.block(x)                                               # refine
        x = self.ln_f(x)
        logits = self.lm_head(x)                                        # [B, T, vocab_size]

        if targets is None:
            loss = None
        else:
            # reshape so cross_entropy can compare prediction vs actual next char
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]          # keep last block_size tokens
            logits, loss = self(idx_cond)            # predict
            logits = logits[:, -1, :]                # focus on the LAST position
            probs = F.softmax(logits, dim=-1)        # scores -> probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # sample one
            idx = torch.cat((idx, idx_next), dim=1)  # append it
        return idx

# --- quick test ---
if __name__ == "__main__":
    torch.manual_seed(1337)

    # bring in the real data
    from tokenizer import train_data, val_data

    batch_size = 32          # how many chunks we train on at once
    max_iters = 3000         # how many training steps
    eval_interval = 300      # how often to print progress
    learning_rate = 1e-3     # how big each weight nudge is

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        # pick batch_size random starting points
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    model = GPTLanguageModel()

    # the optimizer: applies the weight nudges
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        xb, yb = get_batch('train')          # grab a batch of real text
        logits, loss = model(xb, yb)         # predictions + how wrong
        optimizer.zero_grad(set_to_none=True)  # clear old nudges
        loss.backward()                       # backprop: figure out the nudges
        optimizer.step()                      # apply the nudges

        if step % eval_interval == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    print(f"final loss: {loss.item():.4f}")

    # --- generate some text ---
    from tokenizer import decode
    context = torch.zeros((1, 1), dtype=torch.long)   # start with token 0 (newline)
    generated = model.generate(context, max_new_tokens=300)
    print("\n----- GENERATED TEXT -----")
    print(decode(generated[0].tolist()))