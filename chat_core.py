import torch
from gpt import GPTLanguageModel, device, block_size
from tokenizer import decode, encode

# Load the FINE-TUNED conversational model
model = GPTLanguageModel()
model = model.to(device)
model.load_state_dict(torch.load("model_chat.pt", map_location=device))
model.eval()


def get_reply(user_message, max_new_tokens=200):
    # Format the input the way the model was trained: "User: ...\nAI:"
    # Capitalize first letter to match training format ("User: Hello")
    user_message = user_message[0].upper() + user_message[1:] if user_message else user_message
    prompt = f"User: {user_message}\nAI:"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    # Generate one character at a time, stopping if the model starts a new "User:" turn
    for _ in range(max_new_tokens):
        context_cond = context[:, -block_size:]
        logits, _ = model(context_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_id), dim=1)

        # Check if we've generated "\nUser:" — if so, the AI's turn is done
        text_so_far = decode(context[0].tolist())
        if text_so_far.endswith("\nUser:"):
            break

    full_text = decode(context[0].tolist())
    # Extract just the AI's reply (after "AI:" and before any next "User:")
    reply = full_text.split("AI:")[1].split("User:")[0].strip()
    return reply