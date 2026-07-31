import torch
from gpt import GPTLanguageModel, device, block_size
from tokenizer import encode, decode

# --- 1. Load the conversation data ---
with open("conversations.txt", "r", encoding="utf-8") as f:
    conv_text = f.read()

data = torch.tensor(encode(conv_text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"conversation data: {len(data)} characters")

# --- 2. Load the existing (Shakespeare-trained) model ---
model = GPTLanguageModel()
model = model.to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
print("loaded Shakespeare model.pt as starting point")

# --- 3. Fine-tuning settings ---
batch_size = 64
max_iters = 3000
eval_interval = 300
learning_rate = 1e-4     # smaller than before -- gentle nudges, don't wreck what it knows

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- 4. The fine-tuning loop (same shape as training) ---
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % eval_interval == 0:
        print(f"step {step:4d} | loss {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")

# --- 5. Save the fine-tuned model (overwrites model.pt) ---
torch.save(model.state_dict(), "model_chat.pt")
print("saved fine-tuned model to model_chat.pt")

# --- 6. Quick test: does it respond to a prompt? ---
prompt = "User: Hello\nAI:"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=100)
print("\n----- TEST GENERATION -----")
print(decode(generated[0].tolist()))