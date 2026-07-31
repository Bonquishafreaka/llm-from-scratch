# LLM From Scratch

A character-level GPT (transformer) built entirely from scratch in PyTorch, then
fine-tuned into a small conversational chatbot — with terminal and web interfaces.
Built to understand how language models work at every level, from the neuron math
up to a working chat application.

## What's here

**The model (built piece by piece):**
- `tokenizer.py` — character-level tokenizer (text ↔ numbers)
- `head.py` — one head of self-attention
- `feedforward.py` — the feed-forward network
- `block.py` — a full transformer block (attention + feed-forward, with residuals + LayerNorm)
- `gpt.py` — the complete GPT model + training loop

**Data + fine-tuning:**
- `prepare_data.py` — downloads and formats conversation data from HuggingFace
- `finetune.py` — fine-tunes the trained model on conversation data

**Using the model:**
- `chat_core.py` — shared logic: loads the model, generates replies
- `terminal_chat.py` — chat with the model in the terminal
- `web_chat.py` — chat with the model in a browser (Gradio)
- `generate.py` — generate text from a prompt

**Learning the internals by hand (pure Python, no PyTorch):**
- `neuron_from_scratch.py` — a single neuron
- `neuron_layer.py` — a layer of neurons
- `neuron_layers_connected.py` — two layers = a tiny network
- `learning.py` — a neuron learning via gradient descent

## Setup

pip install -r requirements.txt

Then download the training data:

curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## How it was built

1. Trained a character-level GPT on Shakespeare to learn language structure.
2. Fine-tuned it on everyday-conversation data to make it respond conversationally.
3. Wrapped it in terminal and web chat interfaces.

Training was done on a GPU (Google Colab). The model is small (~a few million
parameters), so it responds but is intentionally rough — the goal was
understanding, not a production chatbot.

## Notes

Because the tokenizer's vocabulary was built from Shakespeare, the model can't
represent characters it never saw (most digits, some symbols), so conversation
data is filtered to fit that vocabulary.