import gradio as gr
from chat_core import get_reply

def respond(message, history):
    reply = get_reply(message)
    return reply

demo = gr.ChatInterface(
    fn=respond,
    title="Tiny GPT Chat",
    description="A from-scratch language model. Type something and it continues.",
)

if __name__ == "__main__":
    demo.launch(share=True)