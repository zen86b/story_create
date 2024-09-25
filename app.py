import gradio as gr
import openai
from utils.text_chunk import split_into_chunks
import replicate
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel

class PromptResponse(BaseModel):
    prompt: str

def generate_prompt(chunk):
    response = openai.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a story summary assistant and an professional image prompt creator. You need to create a prompt based on what I say for me to create a picture with it."
            },
            {
                "role": "user",
                "content": chunk
            }],
        max_tokens=512,
        response_format=PromptResponse,
    )
    prompt = response.choices[0].message.parsed.prompt
    return prompt

def generate_image(prompt,model="schnell",aspect_ratio="1:1"):
    input = {
        "prompt": "((Hyper Realistic, In Real Life))" +prompt,
        "go_fast": False,
        "num_outputs": 1,
        "aspect_ratio": aspect_ratio,
        "output_format": "png",
        "output_quality": 100,
        "guidance":5,
        "num_inference_steps":50
    }

    output = replicate.run(
        f"black-forest-labs/flux-{model}",
        input=input
    )
    return output[0]

def process_input(file,model,aspect_ratio):
    if file is None:
        return "No file provided."
    
    with open(file.name, "r") as f:
        content = f.read()
        chunks = split_into_chunks(content)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_prompt, chunk) for chunk in chunks]
        prompts = [future.result() for future in futures]
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_image, prompt,model,aspect_ratio) for prompt in prompts]
        images = [future.result() for future in futures]
        
    return list(zip(chunks, prompts, images)),images

with gr.Blocks() as iface:
    gr.Markdown("# STORY CREATOR")
    gr.Markdown("Upload a text file to split it into chunks and generate images for each chunk using FLUX.")
    
    file_input = gr.File(file_types=[".txt"], label="Upload Text File")
    model_dropdown = gr.Dropdown(choices=["schnell", "dev", "pro"], label="Select Model", value="schnell")
    aspect_ratio = gr.Dropdown(choices=["16:9", "4:3", "1:1"], label="Aspect Ratio", value="1:1")

    
    confirm_button = gr.Button("Confirm")
    output_table = gr.Dataframe(headers=["Chunk", "Prompt", "Image URL"], elem_id="output-table")
    output_gallery = gr.Gallery(label="Generated Images")
    
    confirm_button.click(
        fn=process_input,
        inputs=[file_input, model_dropdown],
        outputs=[output_table,output_gallery]
    )

if __name__ == "__main__":
    iface.launch(share=True)