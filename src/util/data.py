from typing import AnyStr, Dict
import json
import pandas as pd
from itertools import product, combinations
import random
from copy import copy
import re
import shutil
import glob
from collections import defaultdict
from pathlib import Path


BASE_SYSTEM_PROMPT = "Please follow all instructions. Please write all responses in English."

SYSTEM_OPENING = ""

DEMOGRAPHIC_TEMPLATE = "Please follow all instructions from the perspective of a person who is {demographic}. Please write all responses in English."


SYSTEM_CLOSING = ""


OPTION_TEXT = ['Strongly disagree', 'Disagree', 'Agree', 'Strongly agree']

FORMAT_STRING = {
    'closed_domain': "Format the output to JSON with the fields 'selection' with the option that you select and 'response' containing your response in unformatted plain text.",
    'open_domain': "Format all outputs to JSON with the field 'response' containing your response in unformatted plain text."
}

PRIMER_STRING = {
    'closed_domain': '{ "selection": "',
    'open_domain': '{ "response": "'
}


INSTRUCTION_TEMPLATES = {
    "meta-llama/Llama-2-13b-chat-hf": """"[INST] <<SYS>>
{system_prompt}
<</SYS>>
{instruction} [/INST]
""",
 "meta-llama/Meta-Llama-3-8B-Instruct": """"<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": """[INST]
{system_prompt}
{instruction} [/INST]
""",
"google/gemma-7b-it": """<start_of_turn>user
{system_prompt}
{instruction}<end_of_turn>
<start_of_turn>model

""",
"HuggingFaceH4/zephyr-7b-beta": """<|system|>
{system_prompt}</s>
<|user|>
{instruction}</s>
<|assistant|>

""",
"allenai/OLMo-7B-Instruct": """<|system|>
{system_prompt}
<|user|>
{instruction}
<|assistant|>

""",
"mistralai/Mistral-7B-Instruct-v0.2": """[INST]
{system_prompt}
{instruction} [/INST]
"""
}


def fill_prompt(
        instruction: Dict,
        proposition: AnyStr,
        age: AnyStr = '',
        gender: AnyStr = '',
        cls: AnyStr = '',
        nation: AnyStr = '',
        political_orientation: AnyStr = '',
        party: AnyStr = '',
        model: AnyStr="llama",
        **kwargs
):
    """
    Utility function to create a prompt given the template parameters
    :param age:
    :param gender:
    :param cls:
    :param nation:
    :param political_orientation:
    :param party:
    :param instruction:
    :param model:
    :return:
    """
    system_prompt = SYSTEM_OPENING
    age = age if age == '' else age + ' years old'
    cls = cls if cls == '' else 'a part of the ' + cls
    nation = nation if nation == '' else 'from ' + nation
    political_orientation = political_orientation if political_orientation == '' else f"has {political_orientation} politcal views"
    
    demos = [d for d in [age, gender, cls, nation, political_orientation, party] if d != '']
    if len(demos) > 0:
        system_prompt += DEMOGRAPHIC_TEMPLATE + ' '
        if len(demos) > 1:
            demos[-1] = f"and {demos[-1]}"

        if len(demos) > 2:
            demo_string = ', '.join(demos)
        else:
            demo_string = ' '.join(demos)
        system_prompt = system_prompt.replace("{demographic}", demo_string)


    system_prompt += SYSTEM_CLOSING + ' '

    system_prompt += FORMAT_STRING[instruction['type']]

    instruction_prompt = instruction['text']
    instruction_prompt = instruction_prompt.replace("{proposition}", proposition)
    if instruction['type'] == 'closed_domain':
        options = copy(OPTION_TEXT)
        random.shuffle(options)
        #option_text = ' '.join([f"{l}) {opt}" for l,opt in zip('ABCD', options)])
        option_text = ', '.join(options)
        instruction_prompt = instruction_prompt.replace("{options}", option_text)
        #instruction_prompt += f" {FORMAT_STRING[instruction['type']]}"
    else:
        options = OPTION_TEXT


    prompt = INSTRUCTION_TEMPLATES[model]

    # This will likely depend on the model
    prompt = prompt.replace("{system_prompt}", system_prompt)
    prompt = prompt.replace("{instruction}", instruction_prompt)

    prompt += PRIMER_STRING[instruction['type']]


    return prompt, options


def generate_prompts_exhaustive(
        personas_file: AnyStr,
        instructions_file: AnyStr,
        exclude_values_file: AnyStr = None
):
    """

    :param personas_file:
    :param instructions_file:
    :param exclude_values_file: Location of a csv with columns (age, gender, cls, nation, political_orientation, instruction, party)
    :return:
    """
    # Read in the personas and instructions and generate all of the combinations
    with open(personas_file) as f:
        personas_json = json.load(f)
    with open(instructions_file) as f:
        instructions = json.load(f)

    exclude_combos = set()
    if exclude_values_file is not None:
        # Read in the combinations to exclude (e.g. for continuing an incomplete run)
        exclusion_dframe = pd.read_csv(exclude_values_file)
        exclude_combos = set([tuple(r) for r in exclusion_dframe.to_numpy()])

    # Get all of the initial combinations
    initial_combinations = list(product(personas_json['age'], personas_json['gender'], personas_json['cls'], personas_json['nation'], personas_json['political_orientation'], instructions['closed_domain'])) + \
                            list(product(personas_json['age'], personas_json['gender'], personas_json['cls'], personas_json['nation'], personas_json['political_orientation'], instructions['open_domain']))
    for comb in initial_combinations:
        out_comb = comb + (personas_json['party'][comb[3]][comb[4]],)
        if out_comb not in exclude_combos:
            yield out_comb


def generate_prompts_subsample(
        personas_file: AnyStr,
        instructions_file: AnyStr,
        exclude_values_file: AnyStr = None,
        n_categories: int = 2
):
    """

    :param personas_file: Location of JSON file with persona options (probably data/prompting/personas.json)
    :param instructions_file: Location of a JSON file with instruction options (probably data/prompting/instructions.json)
    :param exclude_values_file: Location of a text file where each line is a combination to exclude (just turn the
    returned dict into a string and save to a text file somewhere so that a run can be continued after being cancelled
    :return:
    """
    # Read in the personas and instructions and generate all of the combinations
    with open(personas_file) as f:
        personas_json = json.load(f)
    with open(instructions_file) as f:
        instructions = json.load(f)

    exclude_combos = set()
    if exclude_values_file is not None:
        # Read in the combinations to exclude (e.g. for continuing an incomplete run)
        with open(exclude_values_file) as f:
            exclude_combos = [l.strip() for l in f]
    # Get the combinations
    fields = ['age', 'gender', 'cls', 'nation', 'political_orientation']
    initial_combinations = []
    for combo in combinations(fields, n_categories):
        options_closed = {key:personas_json[key] for key in combo}
        options_closed['instruction'] = [{'text': inst, 'type': 'closed_domain'} for inst in instructions['closed_domain']]

        options_open = {key: personas_json[key] for key in combo}
        options_open['instruction'] = [{'text': inst, 'type': 'open_domain'} for inst in instructions['open_domain']]

        initial_combinations.extend(
            [dict(zip(options_closed.keys(), values)) for values in product(*options_closed.values())] +
            [dict(zip(options_open.keys(), values)) for values in product(*options_open.values())]
        )


    for comb in initial_combinations:

        if str(comb) not in exclude_combos:
            yield comb


def fill_prompt_base_case(
        proposition: AnyStr,
        instruction: Dict,
        model: AnyStr="llama"
):
    """

    :param proposition:
    :return:
    """
    system_prompt = BASE_SYSTEM_PROMPT


    system_prompt += FORMAT_STRING[instruction['type']]

    instruction_prompt = instruction['text']
    instruction_prompt = instruction_prompt.replace("{proposition}", proposition)
    if instruction['type'] == 'closed_domain':
        options = copy(OPTION_TEXT)
        random.shuffle(options)
        option_text = ', '.join(options)
        instruction_prompt = instruction_prompt.replace("{options}", option_text)
    else:
        options = OPTION_TEXT

    prompt = INSTRUCTION_TEMPLATES[model]

    # This will likely depend on the model
    prompt = prompt.replace("{system_prompt}", system_prompt)
    prompt = prompt.replace("{instruction}", instruction_prompt)

    prompt += PRIMER_STRING[instruction['type']]

    return prompt, options


def verify_and_parse_output(text, type='closed_domain'):
    try:
        json_format = json.loads(text, strict=False)
        if (type == 'closed_domain' and len(json_format) == 2 and "selection" in json_format and "response" in json_format) \
            or (type == "open_domain" and len(json_format) == 1 and "response" in json_format):
            return json_format
        else:
            return None
    except json.JSONDecodeError:
        return None
    
    
    
EXPECTED_CHARS = {
    "[": (",", "]"),
    "]": ("[", ","),
    "{": (":",),
    "}": (",", "{", "]"),
    ":": (",", "}"),
    ",": (":", "{", "}", "[", "]", ","),
}

QUOTE = '"'
BACKSLASH = '\\'
LBRACE = '{'
RBRACE = '}'
NW_RGX = re.compile(r'\S')


def fix_escape_quotes_basic(sample):
    regex = r'(?<="response": ").*(?=" })'
    srch = re.search(regex, sample['plain_text_output'], flags=re.S)
    if srch:
        orig = srch.group()
        orig = re.sub(r'(?<!\\)"', '\\"', orig)
        sample['plain_text_output'] = re.sub(regex, orig, sample['plain_text_output'], flags=re.S)
    return sample


def extract_usable_json(raw: str) -> str:
    # Setup output str and a few status-tracking variables.
    output = ''
    in_string = False
    prev = None
    prev_nwnq = None
    json_started = False
    # Skip until a left curly brace is found

    for i, char in enumerate(raw):
        if not json_started:
            if char == LBRACE:
                json_started = True
            else:
                continue
                
                
        # Handle non-escaped quote.
        if char == QUOTE and prev != BACKSLASH:
            if in_string:
                # If we're already inside of a quoted string and if the next
                # non-whitespace character is an expected one, then we have
                # exited the quoted string. Otherwise, escape the quote.
                nw_char = NW_RGX.search(raw, pos = i + 1)
                if nw_char == None:
                    # Add the rest of the json string and return
                    output += '"}'
                    return output
                else:
                    nw_char = nw_char.group()
                if nw_char in EXPECTED_CHARS.get(prev_nwnq, ''):
                    if prev_nwnq == ':' and nw_char == ',':
                        new_nw_char = NW_RGX.search(raw, pos = i + 2)
                        if new_nw_char and (new_nw_char.group() == '"' or new_nw_char.group() == '}'):
                            in_string = False
                        else:
                            output += BACKSLASH
                    else:
                        in_string = False
                else:
                    output += BACKSLASH
            else:
                in_string = True

        elif not in_string and char.strip() and char in EXPECTED_CHARS:
            # Previous non-whitespace, non-quoted character.
            prev_nwnq = char

        if in_string or char.strip() or char in EXPECTED_CHARS:
            # Add character to the output.
            output += char
            prev = char
            
            if not in_string and char == ':':
                nw_char = NW_RGX.search(raw, pos = i + 1)
                if nw_char and nw_char.group() != '"':
                    output += '"'
                    in_string = True
            
        # Ignore the rest of the string if we have a full json object
        if not in_string:
            if char == RBRACE:
                break
            elif char == ",":
                nw_char = NW_RGX.search(raw, pos = i + 1)
                if nw_char and nw_char.group() == RBRACE:
                    # Errant comma, remove
                    output = output[:-1]
                    prev = output[-1]

    if in_string:
        output += '"}'
    return output

def verify_and_parse_output_loose(text, type='closed_domain'):
    try:
        json_format = json.loads(text, strict=False)
        if type == 'closed_domain' and "selection" in json_format:
            selection = json_format['selection']
            if selection.lower() in ['agree', 'disagree', 'strongly disagree', 'strongly agree']:
                if 'response' in json_format:
                    json_format = {
                        'selection': json_format['selection'],
                        'response': json_format['response']
                    }
                else:
                    json_format = {
                        'selection': json_format['selection'],
                        'response': ""
                    }
                return json_format
            else:
                return None
        elif type == "open_domain" and "response" in json_format:
            json_format = {
                'response': json_format['response']
            }
            return json_format
        else:
            return None
    except json.JSONDecodeError:
        return None
    
    
def extract_json(text, type='closed_domain', loose=False):
    if loose:
        fixed_string = extract_usable_json(text)
        fixed_json = verify_and_parse_output_loose(fixed_string, type)
    else:
        fixed_string = fix_escape_quotes_basic(text)
        fixed_json = verify_and_parse_output(fixed_string, type)