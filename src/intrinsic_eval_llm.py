import json
import torch
import random
import argparse
import numpy as np
import transformers
import pandas as pd
import argparse

from vllm import LLM, SamplingParams
from pdb import set_trace
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def enforce_reproducibility(seed=1000):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_code = 'mistralai/Mistral-7B-Instruct-v0.3'

PROMPT = """
You are an intelligent and logical assistant. Your job is to read a sentence and a trope. You need to decide if the sentence is favor of or against the trope. If the sentence is a paraphrase of the trope or supports it, you should classify it as "Favor". When it is disagreeing with the trope, you should classify it as "Against". If the sentence is neutral, you should classify it as "Neutral".

<format>
The format of the output should be as a json file that looks follows:
{
    "Explanation": "<Why>"
    "Decision": "<Decision>",
}
"Decision" one of "Favor", "Against", "Neutral".
</format>

 And now for the task:

    input:
        Trope: "<Trope>"
        Sentence: "<Sentence>"
    output:
"""

sampling_params = {}

def predict_stance(test_file_path)-> None:
        df = pd.read_csv(test_file_path)
        model = LLM(model_code, tokenizer_mode='mistral', dtype='half', max_model_len=8096)
        sampling_params = SamplingParams(max_tokens=6000, temperature=0.2)

        for index, row in df.iterrows():
            trope = row['distilled_trope']
            sentence = row['sentences']

            prompt = PROMPT.replace("<Trope>", trope)
            prompt = prompt.replace("<Sentence>", sentence)

            messages = [{"role": "user", "content": prompt}]

            res = model.chat(messages, sampling_params)
            text_prompt = res[0].outputs[0].text
            
            try:
                ans_dict = json.loads(text_prompt)
                ruling = ans_dict["Decision"]
                explanation = ans_dict["Explanation"]
            except Exception as e:
                print(f"Error in decoding: {text_prompt}")
                print(ans_dict)
                ruling = "None"
                explanation = "None"
        
            df.at[index, 'prediction'] = ruling
            df.at[index, 'explanation'] = explanation
        df.to_csv("./data/tropes/data_with_tropes_entailed.csv", index=False)

if __name__ == '__main__':
    enforce_reproducibility()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tropes_csv", type=str, help="CSV with the tropes", required=True)

    args = parser.parse_args()
    predict_stance(args.tropes_csv)