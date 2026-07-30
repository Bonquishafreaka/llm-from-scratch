import torch
from gpt import GPTLanguageModel, device
from tokenizer import decode, encode

# Load the model once, when this file is imported
model = GPTLanguageModel()
model = model.to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()


def get_reply(prompt, max_new_tokens=200):
    """Take the user's text, return the model's continuation."""
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=max_new_tokens)
    full_text = decode(generated[0].tolist())
    # strip the prompt off the front so we return only the NEW text
    reply = full_text[len(prompt):]
    return reply