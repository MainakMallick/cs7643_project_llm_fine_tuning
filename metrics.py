import torch
import json
import datasets
import os
import types
import doctest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from fine_tune import MODEL_TO_OUTPUT

# ----------------------------
# Load HumanEval Dataset
# ----------------------------
def load_humaneval_data():
    print("Downloading dataset openai_humaneval...")
    try:
        dataset = datasets.load_dataset("openai_humaneval")
        return list(dataset["test"])
    except Exception:
        print("Falling back to local: dataset/humaneval.json")
        if not os.path.exists("dataset/humaneval.json"):
            raise FileNotFoundError("Local fallback file dataset/humaneval.json does not exist.")
        with open("dataset/humaneval.json", "r") as f:
            raw_data = json.load(f)
            if isinstance(raw_data[0], str):
                return [json.loads(item) for item in raw_data]
            return raw_data

# ----------------------------
# Load Base Model and Tokenizer
# ----------------------------
def load_base_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# ----------------------------
# Load Fine-Tuned Model (LoRA or IA3)
# ----------------------------
def load_finetuned_model(model_name, peft_path):
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, peft_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(peft_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# ----------------------------
# Generate Code
# ----------------------------
def generate_code(model, tokenizer, prompt, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------------------
# Run Doctest on Generated Code
# ----------------------------
def run_doctest_on_generated_code(code: str) -> bool:
    import types
    import doctest
    from io import StringIO
    import sys

    try:
        # Step 1: Truncate trailing usage examples or broken lines
        lines = code.strip().splitlines()
        clean_lines = []
        for line in lines:
            if line.strip().startswith("def ") or clean_lines:
                clean_lines.append(line)
            if line.strip().startswith("if __name__") or line.strip().startswith("print("):
                break  # Stop at example/test code

        # Step 2: Inject typing support if needed
        cleaned_code = "from typing import *\n" + "\n".join(clean_lines)

        # Step 3: Execute safely
        local_env = {}
        exec(cleaned_code, local_env)

        # Step 4: Run doctest for each function
        for name, obj in local_env.items():
            if isinstance(obj, types.FunctionType):
                # Redirect stdout to silence output
                stdout_backup = sys.stdout
                sys.stdout = StringIO()
                try:
                    doctest.run_docstring_examples(obj, local_env, name=name, verbose=False)
                    sys.stdout = stdout_backup
                    return True  # Passed
                except doctest.DocTestFailure:
                    sys.stdout = stdout_backup
                    return False
                except Exception:
                    sys.stdout = stdout_backup
                    return False

        return False  # No valid function found

    except Exception as e:
        print("⚠️ Doctest failed:", e)
        return False

# ----------------------------
# HumanEval Evaluation
# ----------------------------
def evaluate_model_on_humaneval(model, tokenizer, data, label=""):
    correct = 0
    total = len(data)

    for example in tqdm(data, desc=f"Evaluating {label}"):
        prompt = example["prompt"]
        generated_code = generate_code(model, tokenizer, prompt)

        if run_doctest_on_generated_code(generated_code):
            correct += 1
        else:
            print("\n--- Failed Test ---")
            print("Prompt:\n", prompt)
            print("Generated Code:\n", generated_code)

    return correct / total if total > 0 else 0.0

# ----------------------------
# Compare Base and Finetuned
# ----------------------------
def compare_base_and_finetuned(model_name, peft_path, test_ratio=0.1):
    base_model, base_tokenizer = load_base_model(model_name)
    ft_model, ft_tokenizer = load_finetuned_model(model_name, peft_path)

    humaneval_data = load_humaneval_data()
    humaneval_data = humaneval_data[:int(test_ratio * len(humaneval_data))]

    base_score = evaluate_model_on_humaneval(base_model, base_tokenizer, humaneval_data, label="Base Model")
    ft_score = evaluate_model_on_humaneval(ft_model, ft_tokenizer, humaneval_data, label="Fine-tuned Model")

    print("\n=== HumanEval Pass@1 Comparison ===")
    print(f"Base Model Score     : {base_score:.4f}")
    print(f"Fine-tuned Model Score: {ft_score:.4f}")

    return {
        "Base Model": base_score,
        "Fine-tuned Model": ft_score
    }

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    PEFT_PATH = MODEL_TO_OUTPUT[MODEL_NAME]

    results = compare_base_and_finetuned(MODEL_NAME, PEFT_PATH)
    print(results)
