#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import time
import logging
import uuid
import openai
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random, wait_combine, retry_if_exception_type

# Import diff_repair from the APE-Bench project
from src.apebench.inference.utils.diff_repair import DiffRepair, apply_diff, generate_diff
from src.utils.lean_utils import remove_lean_comments


class Submission:
    """APE-Bench submission for patch generation"""

    def __init__(self, output_file: str):
        # Constants
        self.output_file = output_file
        self.model_name = "deepseek-v3-250324"
        self.api_base_url = "https://api.deepseek.com"
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "your-api-key-here")
        
        # Prompt templates
        self.system_prompt = """You are given a set of **Task Descriptions**, each specifying modifications to an existing Lean 4 codebase (which may be optional or only partially provided). Your goal is to generate a **unified diff patch** that implements **only** the specified changes in **Lean 4 syntax**, ensuring strict adherence to Lean 4 conventions.

Follow these steps:

### **Step 1: Identify Key Proving Strategies**
- For each Task Description, **analyze and summarize** the key strategies involved, such as:
  - Lemma rewriting
  - Data structure modification
  - Function renaming
  - Introducing new theorems or lemmas
  - Other conceptual or syntactical transformations
- Highlight any specialized proof techniques or high-level ideas guiding your modifications.

### **Step 2: Declaration Inventory**
- List all **relevant declarations** (definitions, lemmas, theorems, data types) to be **added, removed, or modified**.
- For new Lean 4 declarations:
  - Provide **concise, academic-style statements** or descriptions.
  - Explain how they integrate into the overall codebase.

### **Step 3: Determine Modification Locations**
- Identify **where each modification should be applied** within the given Lean 4 codebase.
- Quote relevant **original Lean code** where applicable, indicating:
  - **Insertion points** for new definitions, lemmas, or theorems.
  - **Lines to be modified**, specifying which parts require updates.
  - **Removals**, justifying why specific lines or declarations should be deleted.

### **Step 4: Unified Diff Patch (Lean 4)**
- Present the **final patch** in **unified diff format** with **at least three lines of context before and after** each modified hunk.
- Ensure the patch contains **only** the specified changes—no extraneous edits.
- **Strictly enforce Lean 4 syntax**:
  - Check that all modifications are **Lean 4-compliant** and follow best practices.
  - Avoid deprecated Lean 3 syntax or tactics.
  - Ensure consistency with **Lean 4's module system and proof style**.
- All code must be valid **Lean 4 syntax**, with **no** placeholders (`sorry`, `admit`).
- Do **not** interleave commentary within the diff—explanations belong in Steps 1–3.

### **Response Format**

#### **Step 1: Key Strategies**
[Summarize the main strategies for each Task Description.]

#### **Step 2: Declaration Inventory**
[List modified, removed, or added declarations, providing concise descriptions for new ones.]

#### **Step 3: Modification Locations**
[Identify and quote the relevant Lean code where changes should be made. Specify insertion points, modifications, and removals.]

#### **Step 4: Unified Diff Patch (Lean 4)**
- **Overall Explanation of the Changes:**
  - [Provide a structured natural-language overview of the modifications.]
- **Lean 4 Compliance Reminder:**
  - Clearly highlight how the diff strictly adheres to **Lean 4 syntax**, avoiding **Lean 3 syntax or tactics**.
  - Emphasize key changes in **Lean 4 module system, proof tactics, and syntax adaptations**.
- **Final Patch in Unified Diff Format:**
```diff
[Present the final patch in unified diff format, with at least three lines of context before and after each diff hunk. Ensure strict Lean 4 compliance.]
```"""

        self.input_prompt_with_code = """# Lean4 Code Modification Task

## Task Requirements

{instructions}

## Source Codebase: {filename}

```lean
{lean_code}
```

Please generate a unified diff patch that implements all specified requirements while ensuring strict adherence to Lean4 syntax and conventions."""

        self.input_prompt_without_code = """# Lean4 Code Creation Task

## Task Requirements

{instructions}

## Source Codebase Status

This task requires creating a new file for {filename}. No existing code is provided.

Please generate a unified diff patch that creates this file with all specified requirements while ensuring strict adherence to Lean4 syntax and conventions."""

        # Set up logging
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def generate_logid(self) -> str:
        """Generate a unique log ID for request tracking"""
        return str(uuid.uuid4())

    def create_deepseek_client(self, api_key: Optional[str] = None):
        """Create an OpenAI client for DeepSeek API"""
        if api_key is None:
            api_key = self.api_key
        
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("API key not set. Please set the DEEPSEEK_API_KEY environment variable or provide it directly.")
        
        return openai.OpenAI(
            api_key=api_key,
            base_url=self.api_base_url
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_combine(
            wait_exponential(multiplier=1, min=1, max=60),
            wait_random(0, 2)
        ),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def execute_completion(self, client, params):
        """Execute request with retry logic and jitter"""
        try:
            return client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}")
            raise

    def call_deepseek_api(self, prompt: str, system_prompt: Optional[str] = None, 
                          temperature: float = 0.0, max_tokens: int = 8000,
                          api_key: Optional[str] = None) -> Dict[str, Any]:
        """Call the DeepSeek API to generate a patch"""
        logid = self.generate_logid()
        client = self.create_deepseek_client(api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_headers": {"X-TT-LOGID": logid}
        }
        
        start_time = time.time()
        try:
            completion = self.execute_completion(client, params)
            result = completion.model_dump()
            result['inference_params'] = params
            
            response_time = time.time() - start_time
            self.logger.info(f"API call completed in {response_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Request failed [logid: {logid}]: {str(e)}")
            raise

    def parse_diff_from_response(self, response_text: str) -> Optional[str]:
        """Extract diff code from the response text"""
        diff_match = re.search(r'```diff(.*?)```', response_text, re.DOTALL)
        if diff_match:
            return diff_match.group(1).strip()
        return None

    def process_repair_patch(self, original_code: str, diff_text: str, 
                             strict_match_threshold: float = 0.5) -> Tuple[Optional[str], Optional[str]]:
        """Process and repair a patch using DiffRepair"""
        if not diff_text:
            return None, None
        
        try:
            repairer = DiffRepair(
                original_code, 
                diff_text, 
                strict_match_threshold=strict_match_threshold, 
                max_context_lines=3, 
                exact_match=False
            )
            return repairer.repair()
        except Exception as e:
            self.logger.error(f"Error repairing patch: {str(e)}")
            return None, None

    def generate_patch(self, instruction: str, content_before: str, file_path: str, 
                       temperature: float = 0.0, max_tokens: int = 8000) -> Dict[str, Any]:
        """Generate a patch for a given task"""
        if content_before:
            prompt = self.input_prompt_with_code.format(
                instructions=instruction, 
                lean_code=content_before, 
                filename=file_path
            )
        else:
            prompt = self.input_prompt_without_code.format(
                instructions=instruction, 
                filename=file_path
            )
        
        try:
            api_response = self.call_deepseek_api(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_text = api_response["choices"][0]["message"]["content"]
            diff_text = self.parse_diff_from_response(response_text)
            
            result = {
                "gen_patch": diff_text,
                "gen_content_from_scratch": None,
                "gen_patch_after_repair": None,
                "gen_content_after_repair": None,
                "best_gen_patch_comment_free": None,
                "raw_response": response_text,
                "model": self.model_name,
                "usage": api_response.get("usage", {})
            }
            
            # Create comment-free version of content_before
            content_before_comment_free = remove_lean_comments(content_before) if content_before else ''
            
            if diff_text:
                if not content_before:
                    try:
                        result["gen_content_from_scratch"] = apply_diff("", diff_text)
                        # Generate comment-free version
                        content_after_comment_free = remove_lean_comments(result["gen_content_from_scratch"])
                        result["best_gen_patch_comment_free"] = generate_diff(content_before_comment_free, content_after_comment_free)
                    except Exception as e:
                        self.logger.error(f"Error applying diff for new file: {str(e)}")
                else:
                    repaired_patch, full_new_content = self.process_repair_patch(content_before, diff_text)
                    
                    if full_new_content is not None:
                        result["gen_content_after_repair"] = full_new_content
                        repaired_diff = generate_diff(content_before, full_new_content)
                        result["gen_patch_after_repair"] = repaired_diff
                        
                        # Generate comment-free version
                        content_after_comment_free = remove_lean_comments(full_new_content)
                        result["best_gen_patch_comment_free"] = generate_diff(content_before_comment_free, content_after_comment_free)
                    elif repaired_patch is not None:
                        try:
                            repaired_content = apply_diff(content_before, repaired_patch)
                            actual_diff = generate_diff(content_before, repaired_content)
                            result["gen_content_after_repair"] = repaired_content
                            result["gen_patch_after_repair"] = actual_diff
                            
                            # Generate comment-free version
                            content_after_comment_free = remove_lean_comments(repaired_content)
                            result["best_gen_patch_comment_free"] = generate_diff(content_before_comment_free, content_after_comment_free)
                        except Exception as e:
                            self.logger.error(f"Error applying repaired patch: {str(e)}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error generating patch: {str(e)}")
            return {
                "error": str(e),
                "gen_patch": None,
                "gen_content_from_scratch": None,
                "gen_patch_after_repair": None,
                "gen_content_after_repair": None,
                "best_gen_patch_comment_free": None,
                "model": self.model_name
            }

    def load_input_data(self, input_file: str) -> pd.DataFrame:
        """Load input data from a JSONL or Parquet file"""
        if input_file.endswith('.jsonl'):
            df = pd.read_json(input_file, lines=True)
        elif input_file.endswith('.parquet'):
            df = pd.read_parquet(input_file)
        else:
            raise ValueError(f"Unsupported input file format: {input_file}. Must be .jsonl or .parquet")
        
        self.logger.info(f"Loaded {len(df)} entries from {input_file}")
        return df

    def save_output_data(self, data: List[Dict[str, Any]], output_file: Optional[str] = None):
        """Save output data to a JSONL file"""
        if output_file is None:
            output_file = self.output_file
            
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        self.logger.info(f"Saved {len(data)} results to {output_file}")

    def run(self, input_file: str, temperature: float = 0.0, max_tokens: int = 8000):
        """Run the patch generation on all tasks in the input file"""
        start_time = time.time()
        self.logger.info("Starting APE-Bench submission for patch generation")
        
        df = self.load_input_data(input_file)
        
        results = []
        for idx, row in df.iterrows():
            self.logger.info(f"Processing task {idx+1}/{len(df)}")
            
            instruction = row.get('full_instruction', row.get('instruction', ''))
            content_before = row.get('content_before', '')
            file_path = row.get('file_path_after', row.get('file_path', ''))
            
            if not instruction:
                self.logger.error(f"Task {idx+1} missing instruction, skipping")
                continue
            
            result = self.generate_patch(
                instruction=instruction,
                content_before=content_before,
                file_path=file_path,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result.update(**row)
            
            if "gen_patch_after_repair" in result and result["gen_patch_after_repair"]:
                result["best_gen_patch"] = result["gen_patch_after_repair"]
            elif "gen_patch" in result and result["gen_patch"]:
                result["best_gen_patch"] = result["gen_patch"]
            
            if "gen_content_after_repair" in result and result["gen_content_after_repair"]:
                result["best_gen_content"] = result["gen_content_after_repair"]
            elif "gen_content_from_scratch" in result and result["gen_content_from_scratch"]:
                result["best_gen_content"] = result["gen_content_from_scratch"]
            
            results.append(result)
        
        self.save_output_data(results, self.output_file)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Completed processing {len(df)} tasks in {elapsed_time:.2f} seconds")
        
        return results
