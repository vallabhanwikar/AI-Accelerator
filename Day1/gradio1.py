import gradio as gr

def greet(name):
    return "Hello, " + name + "!"

# Create the interface
# inputs='text' specifies a textbox input
# outputs='text' specifies a textbox output
demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# Launch the interface
# The launch() method starts a local server
# You can access the interface at the displayed URL
demo.launch()