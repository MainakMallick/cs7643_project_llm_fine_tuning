import torch
import evaluate
import json
import datasets
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from typing import List
from codebleu import calc_codebleu  # Correct import for CodeBLEU
from fine_tune import MODEL_TO_OUTPUT, format_example

# Function to download datasets from Hugging Face
def download_dataset(dataset_name, default_path):
    print(f"Downloading dataset {dataset_name}...")
    try:
        dataset = datasets.load_dataset(dataset_name)
        data = dataset["train"] if "train" in dataset else dataset["test"]
        return data
    except Exception:
        print(f"Dataset {dataset_name} not found. Please provide the dataset manually at {default_path}.")
        return None

# Load model and tokenizer
def load_model(model_name, lora_dir):
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Generate code from a given prompt
def generate_code(model, tokenizer, prompt, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluate Pass@1 using HumanEval
# Should we use pass-k 
def evaluate_humaneval(model, tokenizer, humaneval_data):
    correct = 0
    total = len(humaneval_data)
    for example in tqdm(humaneval_data, desc="Evaluating HumanEval"):
        prompt = example["prompt"]
        expected_output = example["canonical_solution"]
        generated_code = generate_code(model, tokenizer, prompt)
        
        try:
            exec_env = {}
            exec(generated_code, exec_env)
            if exec_env.get("output") == expected_output:
                correct += 1
        except Exception:
            pass
    return correct / total

# Evaluate correctness for LiveCodeBench
# TODO: This need to be fixed
def evaluate_livecodebench(model, tokenizer, livecodebench_data):
    correct = 0
    total = len(livecodebench_data)
    for example in tqdm(livecodebench_data, desc="Evaluating LiveCodeBench"):
        prompt = example[""]
        expected_output = example["expected_output"]
        generated_code = generate_code(model, tokenizer, prompt)
        
        try:
            exec_env = {}
            exec(generated_code, exec_env)
            if exec_env.get("output") == expected_output:
                correct += 1
        except Exception:
            pass
    return correct / total

# Evaluate instruction adherence using CodeBLEU
def evaluate_python_instructions(model, tokenizer, instruction_data):
    correct = 0
    total = len(instruction_data)
    
    for example in tqdm(instruction_data, desc="Evaluating Instruction Adherence"):
        prompt = example["instruction"]
        expected_output = example["output"]
        generated_code = generate_code(model, tokenizer, prompt)
        
        if generated_code.strip() == expected_output.strip():
            correct += 1
    
    accuracy = correct / total
    return accuracy

# Main evaluation function
def evaluate_model(model_name, lora_dir, test_ratio=0.1):
    model, tokenizer = load_model(model_name, lora_dir)
    
    # Download datasets or use manual paths
    humaneval_data = download_dataset("openai_humaneval", "dataset/humaneval.json")
    if humaneval_data is None:
        humaneval_data = json.load(open("dataset/humaneval.json"))
    
    livecodebench_data = download_dataset("livecodebench/code_generation_lite", "dataset/livecodebench.json")
    if livecodebench_data is None:
        livecodebench_data = json.load(open("dataset/livecodebench.json"))
    
    instruction_data = download_dataset("iamtarun/python_code_instructions_18k_alpaca", "dataset/python_code_instructions_18k_alpaca.json")
    if instruction_data is None:
        instruction_data = json.load(open("dataset/python_code_instructions_18k_alpaca.json"))
    # Split data for evaluation
    humaneval_data = humaneval_data[:int(test_ratio * len(humaneval_data))]
    #livecodebench_data = livecodebench_data[:int(test_ratio * len(livecodebench_data))]
    instruction_data = instruction_data[:int(test_ratio * len(instruction_data))]

    # Evaluate
    pass_at_1 = evaluate_humaneval(model, tokenizer, humaneval_data)
    #livecodebench_acc = evaluate_livecodebench(model, tokenizer, livecodebench_data)
    code_bleu_score = evaluate_python_instructions(model, tokenizer, instruction_data)
    
    print("\n=== Evaluation Results ===")
    print(f"Pass@1 (HumanEval): {pass_at_1:.4f}")
    #print(f"Correctness (LiveCodeBench): {livecodebench_acc:.4f}")
    print(f"CodeBLEU Score (Instruction Adherence): {code_bleu_score:.4f}")
    
    return {
        "Pass@1": pass_at_1,
        # "LiveCodeBench Accuracy": livecodebench_acc,
        "CodeBLEU Score": code_bleu_score,
    }

# Example usage
if __name__ == "__main__":
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    results = evaluate_model(MODEL_NAME, MODEL_TO_OUTPUT[MODEL_NAME])
    print(results)
