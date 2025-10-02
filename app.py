from transformers import pipeline
import gradio as gr

# Load summarization model
model = pipeline("summarization")

def predict(prompt):
    summary = model(prompt)[0]["summary_text"]
    return summary

# Simple Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="Enter text block to summarize", lines=4),
    outputs="text",
    title="Text Summarizer"
)

if __name__ == "__main__":
    demo.launch()
