from torch import cuda
from copy import copy
import os
import json
import torch
import random
import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm
from torch import bfloat16
import transformers
import hashlib
import uuid
from vllm import LLM, SamplingParams

from util.data import extract_json
from util.data import generate_prompts_subsample
from util.data import fill_prompt
from util.data import verify_and_parse_output
from util.data import PRIMER_STRING


def create_uuid_from_string(val: str):
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return uuid.UUID(hex=hex_string)


def stringify_prompt_fields(prompt_fields, proposition_id):
    out = ''
    for field in prompt_fields:
        if field == 'instruction':
            continue


def enforce_reproducibility(seed=1000):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas_file", type=str, help="Personas JSON file", required=True)
    parser.add_argument("--instructions_file", type=str, help="Instructions JSON file", required=True)
    parser.add_argument("--pct_questions_file", type=str, help="Location of PCT questions", required=True)
    parser.add_argument("--output_dir", type=str, help="Name of the directory to save the generated text", required=True)
    parser.add_argument("--cache_dir", type=str, help="Cache directory for HF models", required=True)
    parser.add_argument("--exclusion_file", type=str, help="Location of file with list of prompts to exclude", default=None)
    parser.add_argument("--split_file", type=str, help="If specified, uses the prompts in a given split file", default=None)
    parser.add_argument("--model_id", type=str, help="The name of the model to use", default='meta-llama/Llama-2-13b-chat-hf',
                        choices=['meta-llama/Llama-2-13b-chat-hf', 'mistralai/Mixtral-8x7B-Instruct-v0.1', 'mistralai/Mistral-7B-Instruct-v0.2', 'google/gemma-7b-it', 'HuggingFaceH4/zephyr-7b-beta', 'allenai/OLMo-7B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct'])
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--random_samples", type=int, help="Specify a set number of random prompts to use", default=-1)
    parser.add_argument("--base_case", action="store_true", help="Generate for the base case (i.e. no demographics)")


    args = parser.parse_args()

    enforce_reproducibility(args.seed)
    model_id = args.model_id
    personas_file = args.personas_file
    instructions_file = args.instructions_file
    pct_questions_file = args.pct_questions_file
    output_dir = args.output_dir
    exclusion_file = args.exclusion_file
    random_samples = args.random_samples
    cache_dir = args.cache_dir

    device = 'cpu'
    if torch.backends.mps.is_available():
        print("Using MPS")
        device = 'mps'
    elif torch.cuda.is_available():
        print("Using CUDA")
        device = 'cuda'

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
   
    model = LLM(model=model_id, trust_remote_code=True, download_dir="/projects/copenlu/data/models/",
                tensor_parallel_size=1, enable_lora=True, dtype='half')
    sampling_params = SamplingParams(
        max_tokens=8000,
        temperature=0.0,
        top_k=1.
    )

    if not os.path.exists(f"{output_dir}/success"):
        os.makedirs(f"{output_dir}/success")
    if not os.path.exists(f"{output_dir}/fail"):
        os.makedirs(f"{output_dir}/fail")
        
    if args.base_case:
        with open(instructions_file) as f:
            instructions_text = json.load(f)
            
        instructions = []
        for text in instructions_text['closed_domain']:
            instructions.append({
                "text": text,
                "type": "closed_domain"
            })
        for text in instructions_text['open_domain']:
            instructions.append({
                "text": text,
                "type": "open_domain"
            })

        samples = []
        option_list = []
        for instruction in instructions:
            for j,proposition in enumerate(propositions):
                prompt, options = fill_prompt_base_case(proposition=proposition, model=model_id, instruction=instruction)
                print(prompt)
                id_ = create_uuid_from_string(prompt)
                if os.path.exists(f"{output_dir}/success/{id_}.txt") or os.path.exists(f"{output_dir}/fail/{id_}.txt"):
                    # Already generated sample for this prompt
                    continue
                samples.append(prompt)
                option_list.append({})
    else:
        # Load and go through the positions
        with open(args.pct_questions_file) as f:
            propositions = [l.strip() for l in f]

        if args.split_file != None:
            with open(args.split_file) as f:
                prompts = [json.loads(l) for l in f]
        else:
            prompts = generate_prompts_subsample(personas_file, instructions_file, exclusion_file, n_categories=2)

        if random_samples > 0:
            prompts = random.sample(list(prompts), random_samples)


        samples = []
        option_list = []
        for i,prompt_fields in enumerate(tqdm(prompts)):
            for j,proposition in enumerate(propositions):
                prompt, options = fill_prompt(proposition=proposition, model=model_id, **prompt_fields)
                print(prompt)
                id_ = create_uuid_from_string(prompt)
                if os.path.exists(f"{output_dir}/success/{id_}.txt") or os.path.exists(f"{output_dir}/fail/{id_}.txt"):
                    # Already generated sample for this prompt
                    continue
                samples.append(prompt)
                option_list.append(options)

    
    
    llm_answers = []
    selection_matrix = []
    finished_prompts = []
    for out,prompt,options in zip(model.generate(samples, sampling_params), samples, option_list):


        # Get text
        text = PRIMER_STRING[prompt_fields['instruction']['type']] + out.outputs[0].text.strip()
        if text.strip()[-1] != '}':
            text += '}'
        # Try and get usable JSON
        json_format = verify_and_parse_output(text, type=prompt_fields['instruction']['type'])
        if json_format is None:
            # First attempt
            json_format = extract_json(text, type=prompt_fields['instruction']['type'], loose=False)
            if json_format is None:
                # Last attempt
                json_format = extract_json(text, type=prompt_fields['instruction']['type'], loose=True)
        out_dict = {
            f: '' for f in ['age', 'gender', 'cls', 'nation', 'political_orientation']
        }
        out_dict['model_id'] = args.model_id
        out_dict['seed'] = args.seed
        out_dict['generation_config'] = model.generation_config.to_dict()
        out_dict['proposition'] = proposition
        out_dict.update(prompt_fields)
        out_dict['plain_text_output'] = text
        if json_format is not None:
            print(json_format['response'])
            out_dict.update(json_format)
            with open(f"{output_dir}/success/{id_}.txt", 'wt') as f:
                f.write(json.dumps(out_dict))
        else:
            print(json_format)
            with open(f"{output_dir}/fail/{id_}.txt", 'wt') as f:
                f.write(json.dumps(out_dict))
        print()
        print()
