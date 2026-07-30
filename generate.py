import torch
from gpt import GPTLanguageModel, device
from tokenizer import decode, encode

# 1. Build an empty model with the SAME architecture as training
model = GPTLanguageModel()
model = model.to(device)

# 2. Load the saved weights into it
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()   # tells the model we're using it, not training
print("loaded model.pt")

# 3. Generate from a prompt
prompt = "ROMEO:"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=300)
print("\n----- GENERATED -----")
print(decode(generated[0].tolist()))