from transformers import AutoTokenizer ,  AutoModelForCausalLM
from openai import OpenAI
device = 'cuda'

def get_hf_model(quant_config):
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    if quant_config:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
        ).to(device)
    tokenizer = get_hf_tokenizer(model_name)
    return model , tokenizer


def get_hf_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer



def get_openai_model(api_key,base_url):
    model = OpenAI(api_key=api_key,base_url=base_url)
    return model

