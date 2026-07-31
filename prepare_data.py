from datasets import load_dataset
from tokenizer import chars   # your existing 65-character vocabulary

# Load the dataset
dataset = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k")
conversations = dataset["train_sft"]

# The set of characters our tokenizer knows (for filtering)
known_chars = set(chars)

# Replace common "fancy" characters with plain equivalents our vocab has
def clean_text(text):
    replacements = {
        '\u2019': "'",   # curly apostrophe '
        '\u2018': "'",   # curly open quote '
        '\u201c': '',   # curly open double quote -> remove
        '\u201d': '',   # curly close double quote -> remove
        '\u2014': '-',   # em-dash —
        '\u2013': '-',   # en-dash –
        '\u2026': '...', # ellipsis …
    }
    for fancy, plain in replacements.items():
        text = text.replace(fancy, plain)
    return text

# --- Format each conversation into a text string ---
def format_conversation(messages):
    text = ""
    for turn in messages:
        content = clean_text(turn["content"])
        if turn["role"] == "user":
            text += "User: " + content + "\n"
        else:  # assistant
            text += "AI: " + content + "\n"
    text += "\n"   # blank line separates one conversation from the next
    return text

# Build the full training text, skipping any conversation with unknown characters
all_text = ""
kept = 0
skipped = 0
for conv in conversations:
    formatted = format_conversation(conv["messages"])
    # only keep it if EVERY character is in our vocabulary
    if all(c in known_chars for c in formatted):
        all_text += formatted
        kept += 1
    else:
        skipped += 1

print(f"kept {kept} conversations, skipped {skipped} (had unknown characters)")
print(f"total characters: {len(all_text)}")

# Save it to a file, like input.txt but for conversations
with open("conversations.txt", "w", encoding="utf-8") as f:
    f.write(all_text)
print("saved conversations.txt")

# Show a sample so we can see what it looks like
print("\n----- SAMPLE -----")
print(all_text[:500])