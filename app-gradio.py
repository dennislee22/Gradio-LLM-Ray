import gradio as gr
import requests
import json

with open("RAY_HEAD_IP") as fp:
    RAY_HEAD_IP=fp.read()
    
# Define the function that will be called when the user clicks the "Submit" button
def query_llm(prompt, max_tokens):
    """
    Sends a prompt to the local LLM API and returns the model's response.

    Args:
        prompt (str): The text prompt to send to the model.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        str: The generated text from the model, or an error message.
    """
    # The URL of your local LLM API endpoint
    url = f"http://{RAY_HEAD_IP}:8081/v1/completions"

    # The headers for the request
    headers = {
        "Content-Type": "application/json"
    }

    # The data payload for the request
    data = {
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": 0
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["text"]
        else:
            return "Error: Could not find generated text in the response. Full response: " + response.text

    except requests.exceptions.RequestException as e:
        # Handle connection errors, timeouts, etc.
        return f"Error: Could not connect to the API. Please ensure the local server is running. Details: {e}"
    except Exception as e:
        # Handle other potential errors
        return f"An unexpected error occurred: {e}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Local LLM Chat 🤖
        Enter a prompt and select the maximum number of tokens to generate a response from your local LLM.
        """
    )
    with gr.Row():
        prompt_input = gr.Textbox(label="Enter your prompt here", lines=5, placeholder="e.g., Singapore is a")
    
    with gr.Row():
        max_tokens_slider = gr.Slider(minimum=10, maximum=512, value=50, step=1, label="Max Tokens")

    with gr.Row():
        submit_button = gr.Button("Submit", variant="primary")

    with gr.Row():
        output_text = gr.Textbox(label="Model Response", lines=10, interactive=False)

    # Define the action for the submit button
    submit_button.click(
        fn=query_llm,
        inputs=[prompt_input, max_tokens_slider],
        outputs=output_text
    )

if __name__ == "__main__":
    # To make it accessible on your network, use `share=True`
    # demo.launch(share=True) 
    demo.queue().launch(
        server_port=8100,
        server_name="localhost",
    )
