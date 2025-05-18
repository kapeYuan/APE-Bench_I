"""DO NOT rename this file!"""
import os
import re
import json
import textwrap
import sys

import openai

from tqdm import tqdm


class Submission:
    """A submission template. """

    def __init__(self, output_file: str):
        """You need to specify the following arguments."""

        self.output_file = output_file

        self.task = "Auto_Formalization"    # [Auto_Formalization, Auto_Informalization]
        self.phase = "development"          # [development, final]

        self.base_url = "http://120.77.8.29:12345/v1/"  # The base url of the model server
        # If you are using OpenAI API or have set API key for
        # your own model, please fill in your API key
        self.api_key = "EMPTY"
        self.model = "./Mistral-7B-Instruct-v0.2"       # Your own model path, or GPTs
        self.prompt = textwrap.dedent("""
            You are a math expert and familar with Lean 3 formal language. 
            Now please translate the following statement and solution of a math 
            word problem into Lean 3 formal solution. Please note that the 
            informal solution and the formal solution need to be identical.
            # Problem: {{informal_statement}}
            # Solution: {{informal_proof}}
            # Formal solution in Lean 3: 
            """)

        # custom generation parameters
        self.max_tokens = 256
        self.temperature = 0.9
        self.top_p = 0.7
        self.frequency_penalty = 0.0

    def generate(self, prompt):
        """We DO NOT recommend modifying this function, as 
        it will be used to test if the model is accessable"""

        openai.api_key = self.api_key
        openai.base_url = self.base_url

        messages = [
            {"role": "user", "content": prompt},
        ]

        completion = openai.chat.completions.create(
            model=self.model, messages=messages, max_tokens=self.max_tokens,
            temperature=self.temperature, top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
        )

        return completion.choices[0].message.content

    def post_process(self, model_output: str):
        """You can post-process the model output here, 
        such as extracting the formal proof from the model output."""

        formal_proof = re.findall(r'```[\S\s]*```', model_output)
        if formal_proof == []:
            formal_proof = re.findall(r'```[\S\s]*', model_output)
        if formal_proof == []:
            formal_proof = [model_output]
        formal_proof = formal_proof[-1].strip()

        lean_code = "\n".join(formal_proof.strip().split("\n")[1:-1])  # remove ```lean ```
        lean_code = re.sub(pattern=r'line [0-9]* ', repl='', string=lean_code)  # remove line *

        return lean_code

    def run(self, input_data: str):
        """Run your model on the given input data, and store the 
        predictions into the output file."""

        with open(input_data, 'r', encoding="utf8") as f:
            datas = json.load(f)

        outputs = []
        for data in tqdm(datas[:10], file=sys.stdout):
            input_text = self.prompt.format(
                informal_statement=data["informal_statement"],
                informal_proof=data["informal_proof"]
            )

            output = self.generate(prompt=input_text)
            outputs.append(dict(
                name=data["name"],
                formal_proof=self.post_process(output),
            ))

        if not os.path.exists(self.output_file):
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf8') as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)
