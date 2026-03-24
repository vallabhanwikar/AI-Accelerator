import gradio as gr
from transformers import pipeline
analysizer = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")

def sentiment(text):
    result = analysizer(text)[0]
    label = result['label']
    score = round(result['score'], 4)
    return f"Sentiment: {label}"

demo_textbox = gr.Interface(
    fn=sentiment,
    inputs=gr.Textbox(label="Enter some text"),
    outputs=gr.Textbox(label="Sentiment Analysis Result"),
    title="Sentiment Analysis Example"
)

demo_textbox.launch()