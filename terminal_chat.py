from chat_core import get_reply

print("=== Tiny GPT Chat (type 'quit' to exit) ===\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    reply = get_reply(user_input)
    print(f"Model:{reply}\n")