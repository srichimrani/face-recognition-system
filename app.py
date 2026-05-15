import gradio as gr

def greet(name):
    return "Hello updated app"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
