import torch
import json
import pandas as pd
from openai import OpenAI
device = 'cuda'
def generate_tokens_hf(tokenizer,messeges):
    input_ids = tokenizer.apply_chat_template(messeges,
                                              return_tensors='pt',
                                              add_generation_prompt = True).to(device)
    attention_mask = torch.ones_like(input_ids,dtype=torch.long,device=device)
    return input_ids,attention_mask


def export_csv(text:str,output_file):
    # Extract JSON in case the model adds extra text
    start = text.find('[')
    end = text.rfind(']') + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in model output.")
    json_text = text[start:end]

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON generated:\n{e}")
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    return output_file
    



def generate_hf(model,tokenizer,messeges,output_file,max_new_tokens=10000):
    input_ids , attention_mask = generate_tokens_hf(tokenizer,messeges)
    outputs = model.generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             max_new_tokens=max_new_tokens)
    generated_tokens = outputs[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated_tokens,
                            skip_special_tokens = True)
    try:
        csv_path = export_csv(text,output_file)
        return csv_path
    except Exception as e:
        return str(e)


def generate_openai(model : OpenAI,model_name : str,messeges,output_file):
    output = model.chat.completions.create(
        model=model_name,
        messages=messeges,
    )
    try:
        csv_path = export_csv(output.choices[0].message.content,output_file)
        return csv_path
    except Exception as e:
        return str(e)

def generate_prompt(purpose, column_names, example_data, sample_size):
    return f"""
You are a synthetic dataset generator.

Task:
Generate {sample_size} new records for the following purpose:

{purpose}

Columns:
{column_names}

Example records:
{example_data}

Requirements:
- Generate exactly {sample_size} objects.
- Follow the same schema and data types as the examples.
- Generate realistic and diverse values.
- Every object must contain exactly these columns:
{column_names}
- Do not add extra fields.
- Do not omit any fields.
- Return ONLY a valid JSON array.
- The first character of the response must be '['.
- The last character of the response must be ']'.
- Do not write explanations.
- Do not write markdown.
- Do not use ```json.
- Do not write any text before or after the JSON array.
"""