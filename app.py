import gradio as gr
import os
from dotenv import load_dotenv
import torch
from transformers import BitsAndBytesConfig
from models import get_openai_model ,  get_hf_model
from generator import (
    generate_prompt,
    generate_openai,
    generate_hf
)

from huggingface_hub import login


load_dotenv(override=True)


hf_token = os.getenv('HF_TOKEN')

login(token=hf_token)

OUTPUT_FILE = "outputs/generated_dataset.csv"

os.makedirs("outputs", exist_ok=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

hf_model , hf_tokenizer = get_hf_model(quant_config)

gemini_base_url = os.getenv('GEMINI_BASE_URL')
gemini_api_key = os.getenv('GEMINI_API_KEY')
gemini_name = os.getenv('GEMINI_NAME')

gpt_base_url = os.getenv('GROQ_BASE_URL')
gpt_api_key = os.getenv('GROQ_API_KEY')
gpt_name = os.getenv('GROQ_GPT_OSS_20B_NAME')


gpt_model = get_openai_model(gpt_api_key,gpt_base_url)
gemini_model = get_openai_model(gemini_api_key,gemini_base_url)

def generate_dataset(
        provider,
        purpose,
        column_names,
        example_data,
        sample_size
):
    messages = [
        {
            'role':'user',
            'content':generate_prompt(purpose,column_names,example_data,sample_size)
        }
    ]
    if provider == 'HuggingFace':
        return generate_hf(
            hf_model,hf_tokenizer,messages,OUTPUT_FILE
        )
    elif provider == 'GPT-OSS-20B':
        return generate_openai(
            gpt_model,gpt_name,messages,OUTPUT_FILE
        )
    else:
        return generate_openai(
            gemini_model,gemini_name,messages,OUTPUT_FILE
        )


with gr.Blocks(title='Synthetic Dataset Generator') as demo:
    gr.Markdown("# Synthetic Dataset Generator")

    provider = gr.Dropdown(
        choices=['HuggingFace','GPT-OSS-20B','Gemini'],
        value='GPT-OSS-20B',
        label='Provider'
        )
    
    purpose = gr.Textbox(
        label='DataSet Purpose',
        lines=8
    )

    column_names = gr.Textbox(
        label='Column Names',
        placeholder="name, age, salary",
        lines=8
    )

    example_data = gr.Textbox(
        label="Example Data",
        lines=8,
        placeholder="""
[
  {"name":"Alice","age":20,"salary":3000}
]
""",
    )

    sample_size = gr.Slider(
        minimum=5,
        maximum=500,
        value=50,
        step=5,
        label='Number of Samples'
    )

    btn = gr.Button('Generate DataSet')

    output_file = gr.File(label='CSV')

    btn.click(
        fn=generate_dataset,
        inputs=[
            provider,
            purpose,
            column_names,
            example_data,
            sample_size
        ],
        outputs=output_file
    )

demo.launch()

