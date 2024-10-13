from torch import cuda
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import bfloat16
import transformers
from typing import List, Dict
import json
from openai import OpenAI

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
SYSTEM_PROMPT_PATH = "tropes/system_prompt.txt"
USER_PROMPT_PATH = "tropes/user_prompt.txt"
ASSISTANT_PROMPT_PATH = "tropes/assistant_prompt.txt"


def enforce_reproducibility(seed: int = 1000) -> None:
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


class QueryModel(object):
    def __init__(
        self,
        model_id: str = MODEL_ID,
    ):
        enforce_reproducibility(1000)
        self.model_id = model_id

    def query(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """
        A function to query the model
        """
        pass


class OpenAIModel(QueryModel):
    def __init__(self, model_id: str, openai_credentials: Dict[str, str]):
        super().__init__(model_id)
        self._validate_credentials(openai_credentials)
        self.model_id = model_id
        self.client = OpenAI(**openai_credentials)

    def _validate_credentials(self, openai_credentials: Dict[str, str]) -> None:
        """
        Validates the format of the OpenAI credentials
        """
        if openai_credentials is None:
            raise ValueError("OpenAI credentials should be provided")
        if "api_key" not in openai_credentials:
            raise ValueError("api_key should be provided")
        if "organization" not in openai_credentials:
            raise ValueError("organization should be provided")
        if "project" not in openai_credentials:
            raise ValueError("project should be provided")
        return True
        
    def query(
        self, messages: List[Dict[str, str]], parameters: Dict = {}
    ) -> Dict[str, str]:
        """
        A function to query the OpenAI client
        """
        completion = self.client.chat.completions.create(model=self.model_id, messages=messages, **parameters)
        raw_result = completion.choices[0].message.content
        raw_result = raw_result.strip()
        return raw_result


class LocalModel(QueryModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,  # 4-bit quantization
            bnb_4bit_quant_type="nf4",  # Normalized float 4
            bnb_4bit_use_double_quant=True,  # Second quantization after the first
            bnb_4bit_compute_dtype=bfloat16,  # Computation type
        )
        max_memory = {i: "30000MB" for i in range(torch.cuda.device_count())}
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            quantization_config=self.bnb_config,
            device_map="auto",
            max_memory=max_memory,
        )
        self.model.eval()
        self.pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            max_new_tokens=300,
            repetition_penalty=1.1,
            do_sample=False,
        )

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        prompt = f'[INST]\n{messages[0]["content"]}\n\n{messages[1]["content"]}\n[/INST]\n{messages[2]["content"]}'
        return prompt

    def query(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """
        A function to query the model
        """
        prompt = self._format_messages(messages)
        raw_result = self.pipeline(prompt)[0]["generated_text"]
        raw_result = raw_result.strip()
        result = {messages[2]["content"]} + result
        return raw_result


class TropeFilter(object):
    """
    A class that filters sentences based whether they contain reasoning or not
    """

    def __init__(self, args):
        self.model_id = args.filtration_model_id
        if not args.local_filter:
            credentials = json.load(open(args.open_ai_credentials_path, "r"))
            self.query_model = OpenAIModel(model_id=self.model_id, openai_credentials=credentials)
        else:
            self.query_model = LocalModel(model_id=self.model_id)

        self.system_prompt = open(SYSTEM_PROMPT_PATH, "r").read()
        self.user_prompt = open(USER_PROMPT_PATH, "r").read()
        self.assistant_prompt = open(ASSISTANT_PROMPT_PATH, "r").read()

    def _construct_messages(
        self, proposition: str, sentence: str
    ) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_prompt.replace("<Comment>", sentence).replace(
                    "<Statement>", proposition
                ),
            },
            {"role": "assistant", "content": self.assistant_prompt},
        ]
        return messages

    def filter_tropes(self, sentences: List[Dict[str, str]], verbose: bool = True):
        """
        A function to filter sentences based on whether they contain reasoning or not
        @param tropes: List of senetences to be filtered, should be in the format:
            [
                {
                    "sentence": [TROPE CANDIDATE]
                    "proposition": [ONE OF THE 62 PROPOSITIONS]
                    "id": [ID OF THE SENTENCE]
                }
            ]
        """
        filtered_tropes = []  # list of sentences that contain reasoning
        for item in sentences:
            sentence = item["sentence"]
            proposition = item["proposition"]
            if type(sentence) != str:
                continue
            if len(sentence) < 10:
                continue

            messages = self._construct_messages(proposition, sentence)
            result = self.query_model.query(messages)
            result = result[: result.find("}") + 1]
                
            # decoding the result
            try:
                as_dict = json.loads(result)
            except json.JSONDecodeError:
                print(f"Error in decoding: {result}")
                continue
            try:
                ruling = as_dict["Decision"].strip()
                explanation = as_dict["Explanation"].strip()
            except KeyError:
                print(f"Error in decoding: {result}")
                print(as_dict)
                continue

            if verbose:
                print("==" * 20)
                print(f"ID: {item['id']}")
                print(f"Proposition: {proposition}")
                print(f"Sentence: {sentence}")
                print(f"Ruling: {ruling}")
                print(f"Explanation: {explanation}")
                print("==" * 20)

            if ruling != "No argument":
                filtered_tropes.append(item)
        return filtered_tropes
