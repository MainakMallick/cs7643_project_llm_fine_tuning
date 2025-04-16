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
from fine_tune import MODEL_TO_OUTPUT, format_example, extract_response

# Function to download datasets from Hugging Face
def download_dataset(dataset_name, default_path):
    if os.path.exists(default_path):
        print(f"Dataset found at {default_path}. Loading from file...")
        with open(default_path, "r") as f:
            dataset = datasets.load_dataset("json", data_files=f.name)
        data = dataset["test"] if "test" in dataset else dataset["train"]
        return data

    print(f"Dataset {dataset_name} not found at {default_path}. Downloading...")
    try:
        dataset = datasets.load_dataset(dataset_name)
        # Use "test" split if available, otherwise fallback to "train"
        data = dataset["test"] if "test" in dataset else dataset["train"]
        
        # Save the dataset to the specified default path
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        data.to_json(default_path)
        
        return data
    except Exception as e:
        print(f"Failed to download dataset {dataset_name}. Error: {e}")
        print(f"Please provide the dataset manually at {default_path}.")
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
# TODO: Should we use pass-k 
# Evaluate Pass@1 using HumanEval
def evaluate_humaneval(model, tokenizer, humaneval_data):
    correct = 0
    total = len(humaneval_data)
    results = []  # To store prompts and generated code

    for example in tqdm(humaneval_data, desc="Evaluating HumanEval"):
        prompt = example["prompt"]
        expected_output = example["canonical_solution"]
        generated_code = generate_code(model, tokenizer, prompt)
        
        # Save the prompt and generated code
        results.append({
            "prompt": prompt,
            "generated_code": generated_code,
            "expected_output": expected_output
        })

        try:
            exec_env = {}
            exec(generated_code, exec_env)
            if exec_env.get("output") == expected_output:
                correct += 1
        except Exception:
            pass

    # Save results to output/human_eval.json
    os.makedirs("output", exist_ok=True)
    with open("output/human_eval.json", "w") as f:
        json.dump(results, f, indent=4)

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
# def evaluate_python_instructions(model, tokenizer, instruction_data):
#     results = []
#     codebleu_scores = []

#     for example in tqdm(instruction_data, desc="Evaluating Instruction Adherence"):
#         try:
#             prompt = format_example(example, is_test=True)
#             expected_output = example["output"]
#             generated_code = generate_code(model, tokenizer, prompt)
#             response = extract_response(generated_code)

#             # Validate inputs
#             if not isinstance(expected_output, str):
#                 print(f"Skipping example ID {example.get('id', 'unknown')}: expected_output is not a string ({type(expected_output)})")
#                 codebleu_scores.append(0)
#                 continue
#             if not isinstance(response, str):
#                 print(f"Skipping example ID {example.get('id', 'unknown')}: response is not a string ({type(response)})")
#                 codebleu_scores.append(0)
#                 continue
#             if not expected_output.strip() or not response.strip():
#                 print(f"Skipping example ID {example.get('id', 'unknown')}: Empty expected_output or response")
#                 codebleu_scores.append(0)
#                 continue

#             # Save result
#             results.append({
#                 "instruction": prompt,
#                 "generated_code": generated_code,
#                 "extracted_response": response,
#                 "expected_output": expected_output
#             })

#             # Compute CodeBLEU
#             codebleu_result = calc_codebleu(
#                 references=[expected_output],
#                 predictions=[response],
#                 lang="python",
#                 weights=(0.25, 0.25, 0.25, 0.25)
#             )
#             # Extract the 'codebleu' score from the result dictionary
#             codebleu_score = codebleu_result['codebleu']
#             codebleu_scores.append(codebleu_score)

#         except Exception as e:
#             print(f"Error processing example ID {example.get('id', 'unknown')}: {str(e)}")
#             codebleu_scores.append(0)
#             continue

def evaluate_python_instructions(model, tokenizer, instruction_data):
    """
    Evaluate instruction adherence using CodeBLEU for a dataset of Python instructions.
    
    Args:
        model: The loaded language model.
        tokenizer: The tokenizer corresponding to the model.
        instruction_data: A dataset with instruction examples. Each example is expected to have
                          the necessary fields, e.g., an instruction prompt and the expected code
                          output (assumed here to be in the "output" key).
                          
    Returns:
        A float representing the CodeBLEU score across the evaluated examples.
    """
    candidates = []  # List to hold model-generated code responses
    references = []  # List to hold the expected reference code from the dataset
    
    for example in tqdm(instruction_data, desc="Evaluating CodeBLEU Instruction Adherence"):
        # Format the prompt. This could combine the instruction and any associated input.
        prompt = format_example(example)
        
        # Generate a response using the model.
        generated_text = generate_code(model, tokenizer, prompt)
        
        # Extract the actual code output using the helper function.
        candidate = extract_response(generated_text)
        
        # Get the reference code from the dataset.
        # Adjust the key if your dataset uses a different name.
        reference = example.get("output", "").strip()
        
        # Append to our collections; stripping extra whitespace for consistency.
        candidates.append(candidate.strip())
        references.append(reference)
    
    # Calculate the CodeBLEU score for the set of generated codes against the references.
    codebleu_score = calc_codebleu(candidates, references, lang='python')
    
    return codebleu_score
    # # Save all results
    # os.makedirs("output", exist_ok=True)
    # with open("output/python_instructions.json", "w") as f:
    #     json.dump(results, f, indent=4)

    # avg_score = sum(codebleu_scores) / len(codebleu_scores) if codebleu_scores else 0.0
    # return avg_score

# Main evaluation function
def evaluate_model(model_name, lora_dir):
    model, tokenizer = load_model(model_name, lora_dir)
    
    # Download datasets or use manual paths
    #humaneval_data = download_dataset("openai_humaneval", "dataset/humaneval.json")
    
    #livecodebench_data = download_dataset("livecodebench/code_generation_lite", "dataset/livecodebench.json")
    
    instruction_data = download_dataset("iamtarun/python_code_instructions_18k_alpaca", "dataset/python_code_instructions_18k_alpaca_test.json")
    
    # Split data for evaluation
    #livecodebench_data = livecodebench_data[:int(test_ratio * len(livecodebench_data))]

    # Evaluate
    #pass_at_1 = evaluate_humaneval(model, tokenizer, humaneval_data)
    #livecodebench_acc = evaluate_livecodebench(model, tokenizer, livecodebench_data)
    code_bleu_score = evaluate_python_instructions(model, tokenizer, instruction_data)
    
    print("\n=== Evaluation Results ===")
    #print(f"Pass@1 (HumanEval): {pass_at_1:.4f}")
    #print(f"Correctness (LiveCodeBench): {livecodebench_acc:.4f}")
    print(f"CodeBLEU Score (Instruction Adherence): {code_bleu_score:.4f}")
    
    return {
        #"Pass@1": pass_at_1,
        # "LiveCodeBench Accuracy": livecodebench_acc,
        "CodeBLEU Score": code_bleu_score,
    }

# Example usage
if __name__ == "__main__":
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    results = evaluate_model(MODEL_NAME, MODEL_TO_OUTPUT[MODEL_NAME])
    print(results)
