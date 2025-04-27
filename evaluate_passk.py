import torch
import json
import datasets
import os
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import IA3Config, PeftModel
from tqdm import tqdm
from fine_tune import MODEL_TO_OUTPUT, format_example, extract_response

def download_humaneval(dataset_name="openai_humaneval", default_path="dataset/humaneval_400.json", force_download=False):
    """
    Download or load HumanEval dataset with fixed 400 samples.
    Args:
        dataset_name: Name of the dataset to download
        default_path: Path to save/load the dataset
        force_download: If True, force download even if file exists
    """
    if os.path.exists(default_path) and not force_download:
        print(f"Loading fixed 400 samples from {default_path}...")
        with open(default_path, "r") as f:
            dataset = datasets.load_dataset("json", data_files=f.name)
        data = dataset["test"] if "test" in dataset else dataset["train"]
        print(f"Loaded {len(data)} samples")
        return data

    print(f"Downloading complete HumanEval dataset from {dataset_name}...")
    try:
        dataset = datasets.load_dataset(dataset_name)
        data = dataset["test"] if "test" in dataset else dataset["train"]
        print(f"Total available samples: {len(data)}")
        
        # Take first 400 samples
        data = data.select(range(min(100, len(data))))
        print(f"Selected {len(data)} samples")
        
        # Save the fixed dataset
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        data.to_json(default_path)
        print(f"Saved fixed samples to {default_path}")
        
        return data
    except Exception as e:
        print(f"Failed to download dataset {dataset_name}. Error: {e}")
        print(f"Please provide the dataset manually at {default_path}.")
        return None

def load_model(model_name, model_dir):
    """
    Load base and fine-tuned models.
    Args:
        model_name: Name of the base model
        model_dir: Directory containing the fine-tuned model (either LORA or IA3)
    """
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Determine if we're using LORA or IA3 based on directory name
    if "lora" in model_dir.lower():
        from peft import LoraConfig
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    else:  # IA3
        config = IA3Config(
            target_modules=["v_proj", "k_proj", "down_proj"],
            feedforward_modules=["down_proj"],
            init_ia3_weights=True,
            inference_mode=True,
            task_type="CAUSAL_LM"
        )
    
    model = PeftModel.from_pretrained(base_model, model_dir, config=config)
    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return base_model, model, tokenizer

def generate_code(model, tokenizer, prompt, num_samples=1, max_tokens=500):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate multiple samples if requested
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_return_sequences=num_samples,
        temperature=0.8,  # Add some randomness
        top_p=0.95,      # Nucleus sampling
        do_sample=True   # Enable sampling
    )
    
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def clean_generated_code(generated_code):
    """Clean the generated code by removing prompts, explanations, and comments."""
    import re
    
    # Remove the prompt if it's included in the output
    markers = [
        "You are a code generator",
        "PROBLEM:",
        "IMPORTANT INSTRUCTIONS:",
        "Generate ONLY the code solution below:",
        "Let me think",
        "Let's start",
        "Wait,",
        "So,"
    ]
    
    for marker in markers:
        if marker in generated_code:
            # Find the last occurrence of any marker and take everything after it
            last_pos = generated_code.rfind(marker)
            generated_code = generated_code[last_pos:].split("\n", 1)[-1]
    
    # Try to extract code blocks if present
    code_block_patterns = [
        r"```python\n(.*?)\n```",
        r"```\n(.*?)\n```",
        r"```(.*?)```"
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, generated_code, re.DOTALL)
        if matches:
            # Take the longest code block
            generated_code = max(matches, key=len).strip()
            break
    
    # Remove any remaining markdown or explanatory text
    lines = generated_code.split("\n")
    code_lines = []
    in_code = False
    
    for line in lines:
        # Skip empty lines at the start
        if not code_lines and not line.strip():
            continue
            
        # Skip lines that look like explanations
        if any(marker in line for marker in markers):
            continue
            
        # Skip markdown-style comments
        if line.strip().startswith(("#", ">")):
            continue
            
        # If we see a Python-like line, we're in code
        if re.match(r'^[\s]*(def|if|for|while|class|import|from|print|[a-zA-Z_][a-zA-Z0-9_]*\s*=)', line):
            in_code = True
            
        if in_code:
            code_lines.append(line)
            
    # If we didn't find any clear code markers, just take lines that look like Python code
    if not code_lines:
        code_lines = [line for line in lines if re.match(r'^[\s]*(def|if|for|while|class|import|from|print|[a-zA-Z_][a-zA-Z0-9_]*\s*=)', line)]
    
    return "\n".join(code_lines)

def evaluate_passk(model, tokenizer, data, k=1, num_samples=10):
    results = []
    correct = 0
    total = len(data)
    
    for example in tqdm(data, desc=f"Evaluating Pass@{k}"):
        # Get problem details
        prompt = example.get("prompt", "")
        entry_point = example.get("entry_point", "")
        
        # Extract test cases from the example
        test_cases = []
        test_fn = example.get("test", "")
        if test_fn:
            # Parse the test function to extract test cases
            import ast
            try:
                tree = ast.parse(test_fn)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assert):
                        # Extract test case from assert statement
                        test_case = {
                            "assert_code": ast.unparse(node),
                            "test_code": test_fn
                        }
                        test_cases.append(test_case)
            except Exception as e:
                print(f"Failed to parse test function: {e}")
        
        # Format prompt
        prompt = (
            f"Generate a Python function that solves this problem. Include ONLY the function code:\n\n"
            f"{prompt}\n"
            f"Function name should be: {entry_point}\n\n"
            f"GENERATE CODE ONLY:\n"
        )
        
        # Generate multiple samples
        generated_codes = generate_code(model, tokenizer, prompt, num_samples=num_samples)
        cleaned_codes = [clean_generated_code(code) for code in generated_codes]
        
        # Save the example results
        example_result = {
            "prompt": prompt,
            "generated_codes": cleaned_codes,
            "test_cases": test_cases,
            "passed_samples": 0,
            "total_samples": num_samples
        }
        
        # Test each generated sample
        passed_samples = 0
        for code in cleaned_codes:
            if not code.strip():
                continue
                
            try:
                # Create test environment
                test_env = {}
                exec(code, test_env)
                
                # Check if function exists
                if entry_point not in test_env:
                    continue
                    
                # Run test cases
                all_passed = True
                for test_case in test_cases:
                    try:
                        # Execute the full test code which includes the assert statements
                        test_env_copy = test_env.copy()
                        exec(test_case["test_code"], test_env_copy)
                        
                    except AssertionError:
                        all_passed = False
                        break
                    except Exception as e:
                        all_passed = False
                        example_result["test_error"] = str(e)
                        break
                
                if all_passed:
                    passed_samples += 1
                    
            except Exception as e:
                example_result["execution_error"] = str(e)
        
        example_result["passed_samples"] = passed_samples
        if passed_samples >= k:
            correct += 1
            
        results.append(example_result)
    
    return correct / total, results

def evaluate_model(model_name, lora_dir, k=1, num_samples=10):
    """
    Evaluate model on fixed 400 HumanEval samples.
    Args:
        model_name: Name of the base model
        lora_dir: Directory containing the fine-tuned model
        k: Pass@k parameter
        num_samples: Number of samples to generate per problem
    """
    # Create output directory
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Load models
    base_model, model, tokenizer = load_model(model_name, lora_dir)
    
    # Load fixed dataset
    humaneval_data = download_humaneval()
    if not humaneval_data:
        print("Error: Could not load dataset")
        return None
    
    num_examples = len(humaneval_data)
    print(f"\nEvaluating on {num_examples} fixed samples")
    
    # Store all results
    all_results = {
        "metadata": {
            "model_name": model_name,
            "lora_dir": lora_dir,
            "num_examples": num_examples,
            "k": k,
            "num_samples": num_samples,
            "evaluation_date": str(datetime.datetime.now())
        },
        "results": {}
    }
    
    # Evaluate base model
    print("\nEvaluating base model...")
    base_pass_k, base_results = evaluate_passk(base_model, tokenizer, humaneval_data, k, num_samples)
    
    # Evaluate fine-tuned model
    print("\nEvaluating fine-tuned model...")
    finetuned_pass_k, finetuned_results = evaluate_passk(model, tokenizer, humaneval_data, k, num_samples)
    
    # Store results
    all_results["results"]["base_model"] = {
        f"pass@{k}": float(base_pass_k),
        "examples": base_results
    }
    all_results["results"]["finetuned_model"] = {
        f"pass@{k}": float(finetuned_pass_k),
        "examples": finetuned_results
    }
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Base Model ({model_name}):")
    print(f"Pass@{k}: {base_pass_k:.4f}")
    print(f"\nFine-tuned Model ({lora_dir}):")
    print(f"Pass@{k}: {finetuned_pass_k:.4f}")
    
    # Save results with method name
    method = "lora" if "lora" in lora_dir.lower() else "ia3" if "ia3" in lora_dir.lower() else "other"
    output_file = f"evaluation_results/passk_results_{method}_{num_examples}_examples_k{k}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return all_results

if __name__ == "__main__":
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Evaluate LORA model
    print("\n=== Evaluating LORA Model ===")
    lora_dir = "IA3-deepseek-ai-1-5b-finetuned"  # Your LORA directory
    lora_results = evaluate_model(MODEL_NAME, lora_dir, k=1, num_samples=5)
    
